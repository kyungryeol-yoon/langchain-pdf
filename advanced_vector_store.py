"""
AdvancedVectorStore v2
──────────────────────
계층적·다중 임베딩·하이브리드 검색을 지원하는 고급 벡터 저장소 모듈.
PDFIngestionPipeline과 직접 연계하여 end-to-end RAG 파이프라인 구성 가능.

버그 수정:
  1. similarity_search_with_scroe  → similarity_search_with_score (오타)
  2. Document / Any 타입 미임포트  → 상단 import에 추가
  3. create_filteredretriever       → create_filtered_retriever (PEP8 네이밍)
  4. FilteredRetriever.get_relevant_documents → invoke() 추가 (LangChain v0.2+)
  5. weights 길이 검증 누락          → vectorstore_keys 길이와 자동 검증

개선 사항:
  1. PDFIngestionPipeline 직접 연계 — PipelineResult 한 번에 인덱싱
  2. 점수 정규화                    — 저장소별 점수 범위를 0~1로 통일
  3. MMR 검색 지원                  — 다양성 기반 검색 (중복 억제)
  4. 저장소 영속성 (save/load)      — FAISS 인덱스를 디스크에 저장·불러오기
  5. 배치 upsert                   — 문서를 청크 단위로 나눠 안전하게 추가
  6. 로깅 시스템                   — print() → logging 모듈
  7. 데이터클래스 결과              — SearchResult dataclass
  8. 컨텍스트 매니저               — with 문 지원
  9. 저장소 존재 여부 검증          — KeyError 대신 명확한 오류 메시지
 10. 통계 추적                     — 저장소별 문서 수·검색 횟수
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_openai import OpenAIEmbeddings

# PDF 파이프라인 연계 (선택적 임포트 — 없어도 독립 사용 가능)
try:
    from pdf_ingestion_pipeline import PipelineResult
    _PIPELINE_AVAILABLE = True
except ImportError:
    _PIPELINE_AVAILABLE = False

# ──────────────────────────────────────────────
# 로깅
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("AdvancedVectorStore")


# ──────────────────────────────────────────────
# 데이터 클래스
# ──────────────────────────────────────────────
@dataclass
class SearchResult:
    """단일 검색 결과"""
    document: Document
    score: float                    # 정규화된 유사도 (0~1, 높을수록 유사)
    source_store: str               # 어느 저장소에서 왔는지
    rank: int = 0

    @property
    def content_preview(self) -> str:
        return self.document.page_content[:120].replace("\n", " ") + "…"


@dataclass
class SearchReport:
    """검색 전체 결과 + 통계"""
    query: str
    results: List[SearchResult]
    elapsed_sec: float
    stores_searched: List[str]
    strategy: str                   # "hybrid" | "mmr" | "similarity"
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"[{self.strategy}] '{self.query[:40]}' → "
            f"{len(self.results)}개 결과 | "
            f"저장소: {self.stores_searched} | "
            f"{self.elapsed_sec:.3f}s"
        )

    def print_results(self, max_show: int = 5) -> None:
        print(f"\n▶ 검색 결과: {self.summary()}")
        for r in self.results[:max_show]:
            meta = r.document.metadata
            print(
                f"  [{r.rank}] score={r.score:.3f} | "
                f"store={r.source_store} | "
                f"strategy={meta.get('split_strategy', '-')} | "
                f"{r.content_preview}"
            )


# ──────────────────────────────────────────────
# 커스텀 검색기 (LangChain BaseRetriever 상속)
# ──────────────────────────────────────────────
class FilteredRetriever(BaseRetriever):
    """
    메타데이터 필터를 적용하는 커스텀 검색기.
    LangChain v0.2+ invoke() 규격 준수.
    """
    vectorstore: Any
    filter_fn: Callable[[Dict], bool]
    k: int = 5
    fetch_k: int = 20               # 필터링 전 후보 수

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        # ✅ Fix: invoke() 규격 → _get_relevant_documents 구현
        candidates = self.vectorstore.similarity_search(query, k=self.fetch_k)
        filtered   = [d for d in candidates if self.filter_fn(d.metadata)]
        return filtered[: self.k]

    # 편의 메서드
    def search(self, query: str) -> List[Document]:
        return self.invoke(query)


# ──────────────────────────────────────────────
# 메인 클래스
# ──────────────────────────────────────────────
class AdvancedVectorStore:
    """
    계층적·다중 임베딩·하이브리드·MMR 검색을 지원하는 벡터 저장소.

    사용 예시:
        with AdvancedVectorStore("openai") as store:
            store.ingest_from_pipeline(pipeline_result, domain="legal")
            report = store.hybrid_search("계약 조건", domain="legal")
            report.print_results()
    """

    SUPPORTED_EMBEDDINGS = ("openai", "openai_small", "huggingface")
    BATCH_SIZE           = 500      # upsert 배치 크기

    def __init__(
        self,
        embedding_type: str = "openai",
        persist_dir: Optional[str] = None,
    ) -> None:
        if embedding_type not in self.SUPPORTED_EMBEDDINGS:
            raise ValueError(
                f"지원하지 않는 임베딩 타입: '{embedding_type}'. "
                f"사용 가능: {self.SUPPORTED_EMBEDDINGS}"
            )
        self.embedding_type  = embedding_type
        self.persist_dir     = Path(persist_dir) if persist_dir else None
        self.embeddings      = self._init_embeddings(embedding_type)
        self._stores: Dict[str, FAISS] = {}
        self._stats: Dict[str, Dict[str, int]] = {}
        logger.info("AdvancedVectorStore 초기화 — 임베딩: %s", embedding_type)

    # ── 컨텍스트 매니저 ──────────────────────────
    def __enter__(self) -> "AdvancedVectorStore":
        return self

    def __exit__(self, *_) -> None:
        if self.persist_dir:
            self.save_all()
        logger.info("세션 통계: %s", self.get_stats())

    # ── 임베딩 초기화 ────────────────────────────
    @staticmethod
    def _init_embeddings(embedding_type: str):
        if embedding_type == "openai":
            return OpenAIEmbeddings(model="text-embedding-3-large")
        elif embedding_type == "openai_small":
            return OpenAIEmbeddings(model="text-embedding-3-small")
        elif embedding_type == "huggingface":
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

    # ── 저장소 유효성 검사 ───────────────────────
    def _require_store(self, key: str) -> FAISS:
        if key not in self._stores:
            available = list(self._stores.keys())
            raise KeyError(
                f"저장소 '{key}' 없음. 사용 가능한 저장소: {available}"
            )
        return self._stores[key]

    def _register_store(self, key: str, store: FAISS, doc_count: int) -> None:
        self._stores[key] = store
        self._stats[key]  = {"doc_count": doc_count, "search_count": 0}
        logger.info("저장소 등록: '%s' (%d개 문서)", key, doc_count)

    # ══════════════════════════════════════════
    # 1. PDFIngestionPipeline 직접 연계 (핵심)
    # ══════════════════════════════════════════
    def ingest_from_pipeline(
        self,
        result: "PipelineResult",
        domain: str,
        build_hierarchical: bool = True,
    ) -> Dict[str, FAISS]:
        """
        PipelineResult를 받아 벡터 저장소를 자동 구성.
        PDF 메타데이터(제목·로더·분할 전략)가 이미 청크에 포함된 상태로 인덱싱.

        Args:
            result            : PDFIngestionPipeline.run() 반환값
            domain            : 저장소 네임스페이스 (예: "legal_docs")
            build_hierarchical: True면 content_type별 분리 저장소도 생성

        Returns:
            생성된 저장소 딕셔너리
        """
        if not _PIPELINE_AVAILABLE:
            raise ImportError("pdf_ingestion_pipeline 모듈이 필요합니다.")

        logger.info(
            "파이프라인 연계 인제스트 — 도메인: %s, 청크: %d개, "
            "PDF: %s, 로더: %s, 분할 전략: %s",
            domain,
            result.chunk_count,
            Path(result.pdf_path).name,
            result.load_result.loader_used,
            result.split_result.strategy_used,
        )

        if build_hierarchical:
            return self.create_hierarchical_store(result.chunks, domain)
        else:
            return self.upsert(result.chunks, f"{domain}_all")

    def ingest_batch_from_pipeline(
        self,
        results: List["PipelineResult"],
        domain: str,
    ) -> Dict[str, FAISS]:
        """
        여러 PipelineResult를 하나의 도메인으로 통합 인덱싱.
        """
        all_chunks: List[Document] = []
        for r in results:
            all_chunks.extend(r.chunks)
        logger.info("배치 인제스트 — 도메인: %s, 파일: %d개, 청크: %d개",
                    domain, len(results), len(all_chunks))
        return self.create_hierarchical_store(all_chunks, domain)

    # ══════════════════════════════════════════
    # 2. 계층적 저장소 생성
    # ══════════════════════════════════════════
    def create_hierarchical_store(
        self, documents: List[Document], domain: str
    ) -> Dict[str, FAISS]:
        """
        content_type별 저장소 + 전체 통합 저장소를 함께 생성.
        빈 타입 그룹은 자동으로 건너뜀.
        """
        # content_type별 분류
        type_groups: Dict[str, List[Document]] = {}
        for doc in documents:
            dtype = doc.metadata.get("content_type", "general")
            type_groups.setdefault(dtype, []).append(doc)

        created: Dict[str, FAISS] = {}

        # 유형별 저장소
        for dtype, docs in type_groups.items():
            if not docs:
                continue
            key   = f"{domain}_{dtype}"
            store = self._build_store(docs)
            self._register_store(key, store, len(docs))
            created[key] = store

        # 통합 저장소
        all_key   = f"{domain}_all"
        all_store = self._build_store(documents)
        self._register_store(all_key, all_store, len(documents))
        created[all_key] = all_store

        logger.info(
            "계층적 저장소 생성 완료 — 도메인: %s, 저장소: %s",
            domain, list(created.keys()),
        )
        return created

    # ══════════════════════════════════════════
    # 3. 다중 임베딩 저장소
    # ══════════════════════════════════════════
    def create_multi_embedding_store(
        self,
        documents: List[Document],
        domain: str,
        models: Optional[List[str]] = None,
    ) -> Dict[str, FAISS]:
        """
        여러 임베딩 모델로 동일 문서를 인덱싱.
        실패한 모델은 경고만 출력하고 건너뜀.
        """
        models = models or ["openai", "openai_small", "huggingface"]
        created: Dict[str, FAISS] = {}

        for model_name in models:
            try:
                emb   = self._init_embeddings(model_name)
                key   = f"{domain}_{model_name}"
                store = self._build_store(documents, embeddings=emb)
                self._register_store(key, store, len(documents))
                created[key] = store
            except Exception as exc:
                logger.warning("[%s] 임베딩 실패, 건너뜀: %s", model_name, exc)

        return created

    # ══════════════════════════════════════════
    # 4. 배치 upsert
    # ══════════════════════════════════════════
    def upsert(
        self,
        documents: List[Document],
        store_key: str,
        batch_size: Optional[int] = None,
    ) -> Dict[str, FAISS]:
        """
        문서를 배치 단위로 분할하여 저장소에 안전하게 추가/갱신.
        store_key가 없으면 신규 생성, 있으면 merge_from으로 확장.
        """
        bsize = batch_size or self.BATCH_SIZE
        total = len(documents)
        logger.info("upsert 시작 — 키: %s, 총 %d개 (배치 크기: %d)", store_key, total, bsize)

        base_store: Optional[FAISS] = None

        for i in range(0, total, bsize):
            batch = documents[i : i + bsize]
            chunk_store = self._build_store(batch)
            if base_store is None:
                base_store = chunk_store
            else:
                base_store.merge_from(chunk_store)
            logger.debug("upsert 진행: %d/%d", min(i + bsize, total), total)

        if base_store:
            # 기존 저장소가 있으면 병합
            if store_key in self._stores:
                self._stores[store_key].merge_from(base_store)
                self._stats[store_key]["doc_count"] += total
                logger.info("기존 저장소에 병합: '%s' (+%d개)", store_key, total)
            else:
                self._register_store(store_key, base_store, total)

        return {store_key: self._stores[store_key]}

    # ══════════════════════════════════════════
    # 5. 필터링된 검색기
    # ══════════════════════════════════════════
    def create_filtered_retriever(  # ✅ Fix: create_filteredretriever → snake_case
        self,
        store_key: str,
        filters: Dict[str, Any],
        k: int = 5,
        fetch_k: int = 20,
    ) -> FilteredRetriever:
        """
        메타데이터 필터를 적용하는 LangChain 호환 검색기 반환.

        filters 예시:
            {"content_type": "text"}
            {"section_index": [0, 1, 2]}
            {"pdf_author": "홍길동", "content_type": ["text", "table"]}
        """
        store = self._require_store(store_key)

        def filter_fn(metadata: Dict[str, Any]) -> bool:
            for key, value in filters.items():
                if key not in metadata:
                    return False
                meta_val = metadata[key]
                if isinstance(value, list):
                    if meta_val not in value:
                        return False
                elif meta_val != value:
                    return False
            return True

        return FilteredRetriever(
            vectorstore=store,
            filter_fn=filter_fn,
            k=k,
            fetch_k=fetch_k,
        )

    # ══════════════════════════════════════════
    # 6. 유사도 검색
    # ══════════════════════════════════════════
    def similarity_search(
        self,
        query: str,
        store_key: str,
        k: int = 5,
    ) -> SearchReport:
        """단일 저장소 유사도 검색"""
        start = time.perf_counter()
        store = self._require_store(store_key)

        raw = store.similarity_search_with_score(query, k=k)
        results = [
            SearchResult(
                document=doc,
                score=self._normalize_score(score),
                source_store=store_key,
                rank=i + 1,
            )
            for i, (doc, score) in enumerate(raw)
        ]
        self._stats[store_key]["search_count"] += 1

        return SearchReport(
            query=query,
            results=results,
            elapsed_sec=round(time.perf_counter() - start, 3),
            stores_searched=[store_key],
            strategy="similarity",
        )

    # ══════════════════════════════════════════
    # 7. MMR 검색 (다양성 기반)
    # ══════════════════════════════════════════
    def mmr_search(
        self,
        query: str,
        store_key: str,
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ) -> SearchReport:
        """
        Maximal Marginal Relevance 검색.
        유사도와 다양성을 동시에 고려해 중복 결과를 억제.

        Args:
            lambda_mult: 1.0 = 유사도 최대화, 0.0 = 다양성 최대화
        """
        start = time.perf_counter()
        store = self._require_store(store_key)

        docs = store.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
        )
        results = [
            SearchResult(document=doc, score=1.0, source_store=store_key, rank=i + 1)
            for i, doc in enumerate(docs)
        ]
        self._stats[store_key]["search_count"] += 1

        return SearchReport(
            query=query,
            results=results,
            elapsed_sec=round(time.perf_counter() - start, 3),
            stores_searched=[store_key],
            strategy=f"mmr(λ={lambda_mult})",
        )

    # ══════════════════════════════════════════
    # 8. 하이브리드 검색
    # ══════════════════════════════════════════
    def hybrid_search(
        self,
        query: str,
        store_keys: List[str],
        weights: Optional[List[float]] = None,
        k: int = 5,
        domain: Optional[str] = None,
    ) -> SearchReport:
        """
        여러 저장소를 가중치로 결합하는 하이브리드 검색.
        점수는 저장소별로 정규화 후 가중치 적용.

        Args:
            query      : 검색 쿼리
            store_keys : 검색할 저장소 키 리스트.
                         domain을 지정하면 해당 도메인의 모든 저장소를 자동 선택.
            weights    : 각 저장소의 가중치 (None이면 균등)
            k          : 최종 반환 결과 수
            domain     : 지정 시 store_keys 대신 이 도메인의 모든 저장소 사용
        """
        # domain으로 자동 키 수집
        if domain:
            store_keys = [k_ for k_ in self._stores if k_.startswith(domain)]
            if not store_keys:
                raise KeyError(f"도메인 '{domain}'에 해당하는 저장소 없음")

        # ✅ Fix: weights 길이 검증
        if weights is None:
            weights = [1.0 / len(store_keys)] * len(store_keys)
        elif len(weights) != len(store_keys):
            raise ValueError(
                f"weights 길이({len(weights)})가 "
                f"store_keys 길이({len(store_keys)})와 다릅니다."
            )

        start      = time.perf_counter()
        candidates: List[Tuple[Document, float, str]] = []

        for key, weight in zip(store_keys, weights):
            if key not in self._stores:
                logger.warning("저장소 '%s' 없음, 건너뜀", key)
                continue
            try:
                # ✅ Fix: similarity_search_with_scroe → similarity_search_with_score
                raw = self._stores[key].similarity_search_with_score(query, k=k * 3)
                # 정규화 후 가중치 적용
                for doc, raw_score in raw:
                    weighted = self._normalize_score(raw_score) * weight
                    candidates.append((doc, weighted, key))
                self._stats[key]["search_count"] += 1
            except Exception as exc:
                logger.warning("[%s] 검색 실패, 건너뜀: %s", key, exc)

        # 점수 내림차순 정렬 + 중복 제거
        candidates.sort(key=lambda x: x[1], reverse=True)
        seen: set = set()
        results: List[SearchResult] = []

        for doc, score, src_key in candidates:
            h = hash(doc.page_content)
            if h not in seen:
                seen.add(h)
                results.append(
                    SearchResult(
                        document=doc,
                        score=round(score, 4),
                        source_store=src_key,
                        rank=len(results) + 1,
                    )
                )
                if len(results) >= k:
                    break

        return SearchReport(
            query=query,
            results=results,
            elapsed_sec=round(time.perf_counter() - start, 3),
            stores_searched=store_keys,
            strategy="hybrid",
            warnings=[] if results else ["검색 결과 없음"],
        )

    # ══════════════════════════════════════════
    # 9. 저장·불러오기
    # ══════════════════════════════════════════
    def save_all(self, base_dir: Optional[str] = None) -> None:
        """모든 저장소를 디스크에 저장"""
        save_path = Path(base_dir) if base_dir else self.persist_dir
        if not save_path:
            raise ValueError("persist_dir 또는 base_dir를 지정해야 합니다.")
        save_path.mkdir(parents=True, exist_ok=True)

        for key, store in self._stores.items():
            store_dir = save_path / key.replace("/", "_")
            store.save_local(str(store_dir))
            logger.info("저장소 저장: '%s' → %s", key, store_dir)

    def load_store(self, key: str, base_dir: Optional[str] = None) -> FAISS:
        """디스크에서 특정 저장소 불러오기"""
        load_path = Path(base_dir) if base_dir else self.persist_dir
        if not load_path:
            raise ValueError("persist_dir 또는 base_dir를 지정해야 합니다.")

        store_dir = load_path / key.replace("/", "_")
        if not store_dir.exists():
            raise FileNotFoundError(f"저장소 경로 없음: {store_dir}")

        store = FAISS.load_local(
            str(store_dir),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        self._register_store(key, store, store.index.ntotal)
        logger.info("저장소 불러오기: '%s' ← %s", key, store_dir)
        return store

    # ══════════════════════════════════════════
    # 10. 통계
    # ══════════════════════════════════════════
    def get_stats(self) -> Dict[str, Any]:
        """저장소별 문서 수·검색 횟수 통계 반환"""
        return {
            key: {**stat, "store_key": key}
            for key, stat in self._stats.items()
        }

    def list_stores(self) -> List[str]:
        """등록된 저장소 키 목록 반환"""
        return list(self._stores.keys())

    # ── 내부 유틸 ────────────────────────────────
    def _build_store(
        self,
        documents: List[Document],
        embeddings=None,
    ) -> FAISS:
        """FAISS 저장소 생성 헬퍼"""
        emb = embeddings or self.embeddings
        return FAISS.from_documents(documents, emb)

    @staticmethod
    def _normalize_score(raw_score: float) -> float:
        """
        FAISS L2 거리를 유사도(0~1)로 변환.
        score = 1 / (1 + distance)
        """
        return float(1.0 / (1.0 + max(raw_score, 0.0)))


# ──────────────────────────────────────────────
# end-to-end 파이프라인 예시
# ──────────────────────────────────────────────
if __name__ == "__main__":
    from pdf_ingestion_pipeline import PDFIngestionPipeline

    # ── 1. PDF → 청크 ─────────────────────────
    pipeline = PDFIngestionPipeline(chunk_size=1000, chunk_overlap=200)
    result   = pipeline.run("example.pdf")
    result.print_report()

    # ── 2. 청크 → 벡터 저장소 ─────────────────
    with AdvancedVectorStore(embedding_type="openai", persist_dir="./vectorstore") as store:

        # PDFIngestionPipeline 결과를 바로 인덱싱
        store.ingest_from_pipeline(result, domain="legal_docs")

        print("\n▶ 등록된 저장소:", store.list_stores())

        # ── 3. 유사도 검색 ──────────────────────
        sim_report = store.similarity_search("계약 조건", "legal_docs_all")
        sim_report.print_results()

        # ── 4. MMR 검색 (다양성 강화) ───────────
        mmr_report = store.mmr_search(
            "계약 조건", "legal_docs_all", k=5, lambda_mult=0.6
        )
        mmr_report.print_results()

        # ── 5. 하이브리드 검색 ──────────────────
        hybrid_report = store.hybrid_search(
            query="계약 조건",
            store_keys=["legal_docs_text", "legal_docs_all"],
            weights=[0.7, 0.3],
            k=5,
        )
        hybrid_report.print_results()

        # 또는 도메인 전체 자동 검색
        domain_report = store.hybrid_search("손해배상", domain="legal_docs")
        domain_report.print_results()

        # ── 6. 필터링 검색 ──────────────────────
        retriever = store.create_filtered_retriever(
            store_key="legal_docs_all",
            filters={"content_type": ["text", "table"], "section_index": [0, 1, 2]},
            k=5,
        )
        filtered_docs = retriever.invoke("계약 해지")
        print(f"\n▶ 필터링 검색 결과: {len(filtered_docs)}개")

        # ── 7. 통계 ─────────────────────────────
        print("\n▶ 저장소 통계")
        for key, stat in store.get_stats().items():
            print(f"  {key:<30}: 문서 {stat['doc_count']}개, "
                  f"검색 {stat['search_count']}회")

    # ── 8. 불러오기 예시 ──────────────────────
    # store2 = AdvancedVectorStore(persist_dir="./vectorstore")
    # store2.load_store("legal_docs_all")
    # report = store2.similarity_search("계약 조건", "legal_docs_all")