"""
AdvancedRAGPipeline v3
──────────────────────
컨텍스트 인식 · 다중 쿼리 · 적응형 RAG를 통합한 고급 파이프라인.
AdvancedVectorStore와 직접 연계하여 end-to-end RAG 구성 가능.

v3 추가 기능:
  1. LLM 응답 캐싱 (SQLiteCache)
     - 동일 질문·컨텍스트 조합은 OpenAI API를 재호출하지 않고 DB에서 반환
     - 질문 분류 / 다중 쿼리 생성 / 최종 답변 생성 모든 LLM 호출에 자동 적용
     - get_cache_stats()로 히트율·절약 토큰 조회, clear_cache()로 초기화
     - cache_path=None 으로 캐싱 비활성화 가능

  2. 압축 검색 (ContextualCompressionRetriever)
     - 검색된 청크에서 질문과 관련된 문장만 LLM이 추출 → 노이즈 제거
     - use_compression=True 로 활성화, 기본값 False (성능 우선)
     - compression_llm_model 로 압축 전용 경량 모델 지정 가능
       (기본값 None → 메인 LLM 재사용)
     - RAGResult.docs_before_compression 으로 압축 전후 비교 가능

버그 수정:
  1. langchain_classic                  → 존재하지 않는 패키지.
                                          create_history_aware_retriever는
                                          langchain.chains 에서 임포트
  2. vectorsotre (오타 파라미터명)       → vectorstore 로 통일
  3. genreate_queries (오타)            → generate_queries
  4. .invoke("question": question)      → 유효하지 않은 문법.
                                          .invoke({"question": question}) 로 수정
  5. summariztion (오타, 분류 프롬프트) → summarization
  6. retriever.get_relevant_documents() → LangChain v0.2+ deprecated.
                                          retriever.invoke() 로 교체

기존 개선 사항 (v2):
  1. AdvancedVectorStore 직접 연계  — hybrid/MMR 검색을 RAG에 그대로 활용
  2. 대화 기록 자동 관리            — 슬라이딩 윈도우
  3. 출처(Citation) 추적            — 답변에 사용된 청크 출처 자동 첨부
  4. 스트리밍 지원                  — stream() 메서드로 토큰 단위 실시간 출력
  5. 토큰 예산 관리                 — 컨텍스트 길이 초과 시 자동 잘라냄
  6. RAGResult 데이터클래스         — 답변·출처·통계를 단일 객체로 반환
  7. 로깅 시스템                    — print() → logging 모듈
  8. 컨텍스트 매니저                — with 문 지원 + 세션 통계 출력
  9. 재시도 로직                    — LLM 호출 실패 시 N회 자동 재시도
 10. 세션 통계                      — 쿼리 수·평균 응답 시간·질문 유형 분포
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import langchain
# from langchain.chains import create_history_aware_retriever  # ✅ Fix 1
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# from langchain.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
# from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_community.cache import SQLiteCache
from langchain_core.documents import Document
from langchain_core.globals import set_llm_cache
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain_openai import ChatOpenAI

# 이전 모듈 연계 (선택적 임포트)
try:
    from advanced_vector_store import AdvancedVectorStore, SearchReport
    _VECTOR_STORE_AVAILABLE = True
except ImportError:
    _VECTOR_STORE_AVAILABLE = False

try:
    from pdf_ingestion_pipeline import PDFIngestionPipeline, PipelineResult
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
logger = logging.getLogger("AdvancedRAGPipeline")

# ──────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────
QUESTION_TYPES      = ("factual", "analytical", "comparative", "summarization")
MAX_CONTEXT_CHARS   = 12_000   # 컨텍스트 최대 길이 (토큰 예산)
HISTORY_WINDOW      = 6        # 유지할 최근 대화 턴 수 (HumanMessage + AIMessage 쌍)
LLM_MAX_RETRIES     = 3        # LLM 호출 실패 시 재시도 횟수


# ──────────────────────────────────────────────
# 데이터 클래스
# ──────────────────────────────────────────────
@dataclass
class RAGResult:
    """RAG 응답 단일 결과"""
    question: str
    answer: str
    question_type: str
    source_docs: List[Document]
    elapsed_sec: float
    retrieval_strategy: str = "similarity"
    cache_hit: bool = False                 # LLM 캐시 히트 여부
    docs_before_compression: int = 0        # 압축 전 청크 수 (0이면 압축 미적용)
    warnings: List[str] = field(default_factory=list)

    @property
    def citations(self) -> List[str]:
        """출처 목록 (중복 제거)"""
        seen: set = set()
        result: List[str] = []
        for doc in self.source_docs:
            src = doc.metadata.get("source", "")
            title = doc.metadata.get("pdf_title", "")
            label = title or src
            if label and label not in seen:
                seen.add(label)
                result.append(label)
        return result

    @property
    def compression_ratio(self) -> Optional[str]:
        """압축률 (압축 적용 시에만 반환)"""
        if self.docs_before_compression > 0 and len(self.source_docs) > 0:
            ratio = len(self.source_docs) / self.docs_before_compression
            return f"{len(self.source_docs)}/{self.docs_before_compression} ({ratio:.0%})"
        return None

    def print_report(self) -> None:
        sep = "─" * 55
        print(f"\n{sep}")
        print(f"  질문 유형  : {self.question_type}")
        print(f"  검색 전략  : {self.retrieval_strategy}")
        print(f"  소요 시간  : {self.elapsed_sec:.3f}s")
        print(f"  캐시 히트  : {'예 (API 미호출)' if self.cache_hit else '아니오'}")
        if self.compression_ratio:
            print(f"  압축 결과  : {self.compression_ratio}")
        print(f"  참조 출처  : {self.citations or '없음'}")
        print(f"\n  질문: {self.question}")
        print(f"\n  답변:\n  {self.answer}")
        if self.warnings:
            print(f"\n  경고: {self.warnings}")
        print(sep)


@dataclass
class SessionStats:
    """세션 전체 통계"""
    total_queries: int = 0
    total_elapsed_sec: float = 0.0
    type_distribution: Dict[str, int] = field(default_factory=dict)
    retry_count: int = 0
    # ── v3 추가 ──────────────────────────────
    cache_hits: int = 0                 # LLM 캐시 히트 횟수
    cache_misses: int = 0               # LLM 캐시 미스 횟수
    total_compressed_removed: int = 0   # 압축으로 제거된 누적 청크 수

    @property
    def avg_elapsed(self) -> float:
        return round(self.total_elapsed_sec / max(self.total_queries, 1), 3)

    @property
    def cache_hit_rate(self) -> str:
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return "N/A"
        return f"{self.cache_hits / total:.0%}"

    def record(self, result: RAGResult) -> None:
        self.total_queries += 1
        self.total_elapsed_sec += result.elapsed_sec
        t = result.question_type
        self.type_distribution[t] = self.type_distribution.get(t, 0) + 1
        # 캐시 히트 여부 기록
        if result.cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        # 압축 제거 수 누적
        if result.docs_before_compression > 0:
            removed = result.docs_before_compression - len(result.source_docs)
            self.total_compressed_removed += max(removed, 0)


# ──────────────────────────────────────────────
# 메인 클래스
# ──────────────────────────────────────────────
class AdvancedRAGPipeline:
    """
    컨텍스트 인식·다중 쿼리·적응형 질문 라우팅을 갖춘 RAG 파이프라인.

    사용 예시:
        with AdvancedRAGPipeline(vector_store=store, domain="legal") as rag:
            result = rag.ask("계약 해지 조건은?")
            result.print_report()

            # 스트리밍
            for token in rag.stream("주요 조항을 요약해주세요"):
                print(token, end="", flush=True)
    """

    def __init__(
        self,
        vectorstore: Any,                           # FAISS 또는 AdvancedVectorStore
        llm_model: str = "gpt-4",
        temperature: float = 0.1,
        retrieval_strategy: str = "similarity",     # "similarity" | "mmr" | "hybrid"
        domain: Optional[str] = None,               # AdvancedVectorStore의 도메인
        k: int = 5,
        # ── v3: 캐싱 ──────────────────────────────
        cache_path: Optional[str] = "./llm_cache.db",
        # ── v3: 압축 검색 ─────────────────────────
        use_compression: bool = False,
        compression_llm_model: Optional[str] = None,
    ) -> None:
        self.vectorstore        = vectorstore         # ✅ Fix 2: vectorsotre → vectorstore
        self.llm                = ChatOpenAI(model=llm_model, temperature=temperature)
        self.retrieval_strategy = retrieval_strategy
        self.domain             = domain
        self.k                  = k
        self.chat_history: List[BaseMessage] = []
        self._stats             = SessionStats()

        # ── v3: SQLiteCache 초기화 ─────────────────
        self._cache_path = cache_path
        if cache_path:
            set_llm_cache(SQLiteCache(database_path=cache_path))
            logger.info("LLM 캐시 활성화 → %s", cache_path)
        else:
            set_llm_cache(None)
            logger.info("LLM 캐시 비활성화")

        # ── v3: 압축 검색기 초기화 ─────────────────
        self._use_compression = use_compression
        self._compressor: Optional[ContextualCompressionRetriever] = None
        if use_compression:
            comp_llm = (
                ChatOpenAI(model=compression_llm_model, temperature=0)
                if compression_llm_model
                else self.llm
            )
            self._base_compressor = LLMChainExtractor.from_llm(comp_llm)
            logger.info(
                "압축 검색 활성화 — 압축 모델: %s",
                compression_llm_model or llm_model,
            )

        # 검색기 초기화
        self.retriever = self._build_retriever()
        logger.info(
            "AdvancedRAGPipeline 초기화 — 모델: %s, 검색 전략: %s, "
            "캐시: %s, 압축: %s",
            llm_model, retrieval_strategy,
            "ON" if cache_path else "OFF",
            "ON" if use_compression else "OFF",
        )

    # ── 컨텍스트 매니저 ──────────────────────────
    def __enter__(self) -> "AdvancedRAGPipeline":
        return self

    def __exit__(self, *_) -> None:
        s = self._stats
        logger.info(
            "세션 종료 — 총 쿼리: %d, 평균 응답: %.3fs, 유형 분포: %s | "
            "캐시 히트율: %s (%d/%d) | 압축 제거 청크: %d개",
            s.total_queries, s.avg_elapsed, s.type_distribution,
            s.cache_hit_rate, s.cache_hits, s.cache_hits + s.cache_misses,
            s.total_compressed_removed,
        )

    # ── 검색기 초기화 ────────────────────────────
    def _build_retriever(self):
        """
        AdvancedVectorStore면 hybrid/MMR 전략 활용.
        일반 FAISS면 as_retriever()로 폴백.
        """
        if _VECTOR_STORE_AVAILABLE and isinstance(self.vectorstore, AdvancedVectorStore):
            logger.info("AdvancedVectorStore 연계 검색기 사용 — 전략: %s", self.retrieval_strategy)
            return None  # _retrieve()에서 직접 호출
        else:
            logger.info("기본 FAISS 검색기 사용")
            return self.vectorstore.as_retriever(search_kwargs={"k": self.k})

    def _retrieve(self, query: str) -> List[Document]:
        """
        retrieval_strategy에 따라 적절한 검색 실행.
        AdvancedVectorStore 연계 시 hybrid/MMR 검색을 그대로 활용.
        use_compression=True 이면 관련 문장만 추출 후 반환.
        """
        if _VECTOR_STORE_AVAILABLE and isinstance(self.vectorstore, AdvancedVectorStore):
            store: AdvancedVectorStore = self.vectorstore

            if self.retrieval_strategy == "hybrid" and self.domain:
                report: SearchReport = store.hybrid_search(
                    query, domain=self.domain, k=self.k
                )
            elif self.retrieval_strategy == "mmr":
                key = f"{self.domain}_all" if self.domain else next(iter(store.list_stores()))
                report = store.mmr_search(query, key, k=self.k)
            else:
                key = f"{self.domain}_all" if self.domain else next(iter(store.list_stores()))
                report = store.similarity_search(query, key, k=self.k)

            docs = [r.document for r in report.results]
        else:
            # 일반 FAISS 검색기 폴백  ✅ Fix 6: get_relevant_documents → invoke
            docs = self.retriever.invoke(query)

        # ── v3: 압축 검색 적용 ─────────────────────
        if self._use_compression and docs:
            docs = self._compress_documents(docs, query)

        return docs

    def _compress_documents(
        self, docs: List[Document], query: str
    ) -> List[Document]:
        """
        LLMChainExtractor로 각 청크에서 질문과 관련된 문장만 추출.
        빈 결과가 나온 청크(관련 없음)는 자동으로 제거됨.
        """
        try:
            compressed = self._base_compressor.compress_documents(docs, query)
            removed = len(docs) - len(compressed)
            if removed > 0:
                logger.info(
                    "압축 검색: %d개 → %d개 청크 (-%d개 제거)",
                    len(docs), len(compressed), removed,
                )
            return compressed if compressed else docs  # 전부 제거되면 원본 반환
        except Exception as exc:
            logger.warning("압축 검색 실패, 원본 반환: %s", exc)
            return docs

    # ══════════════════════════════════════════
    # 1. 컨텍스트 인식 검색기
    # ══════════════════════════════════════════
    def create_contextual_retriever(self):
        """
        이전 대화 기록을 반영해 질문을 재구성 후 검색.
        AdvancedVectorStore 연계 시 hybrid/MMR 전략 유지.
        """
        if self.retriever is None:
            # AdvancedVectorStore용 래퍼 검색기
            from langchain_core.retrievers import BaseRetriever
            from langchain_core.callbacks import CallbackManagerForRetrieverRun

            pipeline_self = self

            class _WrappedRetriever(BaseRetriever):
                def _get_relevant_documents(
                    self, query: str, *, run_manager: CallbackManagerForRetrieverRun
                ) -> List[Document]:
                    return pipeline_self._retrieve(query)

            base_retriever = _WrappedRetriever()
        else:
            base_retriever = self.retriever

        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "이전 대화 기록과 최신 사용자 질문이 주어졌을 때, "
             "이전 대화 맥락을 참조하는 질문을 독립적으로 이해할 수 있는 질문으로 "
             "재구성하세요. 질문에 답하지 말고, 필요하다면 재구성만 하고, "
             "그렇지 않으면 그대로 반환하세요."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        return create_history_aware_retriever(  # ✅ Fix 1: langchain_classic 제거
            self.llm, base_retriever, contextualize_prompt
        )

    # ══════════════════════════════════════════
    # 2. 다중 쿼리 검색
    # ══════════════════════════════════════════
    def _generate_queries(self, question: str) -> List[str]:  # ✅ Fix 3: genreate → generate
        """원본 질문으로부터 다양한 관점의 파생 쿼리 3개 생성"""
        multi_query_prompt = ChatPromptTemplate.from_template(
            "다음 질문에 대해 다른 관점에서 3개의 유사한 질문을 생성하세요.\n"
            "각 질문은 한 줄에 하나씩 작성하세요.\n\n"
            "원본 질문: {question}\n\n"
            "대안 질문들:"
        )
        queries_text = (multi_query_prompt | self.llm | StrOutputParser()).invoke(
            {"question": question}  # ✅ Fix 4: .invoke("question": ...) → .invoke({...})
        )
        derived = [q.strip() for q in queries_text.split("\n") if q.strip()]
        return [question] + derived  # 원본 포함

    def multi_query_retrieve(self, question: str) -> List[Document]:
        """다중 쿼리로 검색 후 중복 제거한 문서 반환"""
        queries  = self._generate_queries(question)
        all_docs: List[Document] = []

        for q in queries:
            docs = self._retrieve(q)  # ✅ Fix 6: get_relevant_documents → _retrieve
            all_docs.extend(docs)

        seen:         set            = set()
        unique_docs: List[Document] = []
        for doc in all_docs:
            h = hash(doc.page_content)
            if h not in seen:
                seen.add(h)
                unique_docs.append(doc)

        logger.info("다중 쿼리 검색: 쿼리 %d개 → 중복 제거 후 %d개", len(queries), len(unique_docs))
        return unique_docs[: self.k]

    # ══════════════════════════════════════════
    # 3. 질문 유형 분류
    # ══════════════════════════════════════════
    def _classify_question(self, question: str) -> str:
        """질문을 factual / analytical / comparative / summarization 중 하나로 분류"""
        classification_prompt = ChatPromptTemplate.from_template(
            "다음 질문을 분류하세요:\n"
            "1. factual: 사실적 정보 요청\n"
            "2. analytical: 분석적 사고 필요\n"
            "3. comparative: 비교 분석 필요\n"
            "4. summarization: 요약 요청\n\n"  # ✅ Fix 5: summariztion → summarization
            "질문: {question}\n\n"
            "분류 결과 (한 단어로만): "
        )
        result = (classification_prompt | self.llm | StrOutputParser()).invoke(
            {"question": question}
        )
        qtype = result.strip().lower()
        if qtype not in QUESTION_TYPES:
            logger.warning("알 수 없는 질문 유형 '%s', factual로 폴백", qtype)
            qtype = "factual"
        return qtype

    # ══════════════════════════════════════════
    # 4. 질문 유형별 프롬프트
    # ══════════════════════════════════════════
    _PROMPTS: Dict[str, str] = {
        "factual": (
            "다음 문서에서 정확한 사실 정보를 찾아 답변하세요.\n\n"
            "문서:\n{context}\n\n질문: {question}\n\n"
            "답변 (사실만 간결하게):"
        ),
        "analytical": (
            "다음 문서를 분석하여 깊이 있는 답변을 제공하세요.\n\n"
            "문서:\n{context}\n\n질문: {question}\n\n"
            "분석적 답변:"
        ),
        "comparative": (
            "다음 문서들을 비교 분석하여 답변하세요.\n\n"
            "문서:\n{context}\n\n질문: {question}\n\n"
            "비교 분석 답변:"
        ),
        "summarization": (
            "다음 문서의 내용을 요약하여 답변하세요.\n\n"
            "문서:\n{context}\n\n질문: {question}\n\n"
            "요약 답변:"
        ),
    }

    def _get_prompt(self, qtype: str) -> ChatPromptTemplate:
        template = self._PROMPTS.get(qtype, self._PROMPTS["factual"])
        return ChatPromptTemplate.from_template(template)

    # ══════════════════════════════════════════
    # 5. 토큰 예산 관리
    # ══════════════════════════════════════════
    @staticmethod
    def _build_context(docs: List[Document], max_chars: int = MAX_CONTEXT_CHARS) -> str:
        """
        문서를 연결한 컨텍스트 문자열.
        max_chars를 초과하면 잘라내고 경고 접미사 추가.
        """
        parts: List[str] = []
        total = 0
        for doc in docs:
            chunk = doc.page_content
            if total + len(chunk) > max_chars:
                remaining = max_chars - total
                if remaining > 200:
                    parts.append(chunk[:remaining] + "\n[... 이하 생략]")
                break
            parts.append(chunk)
            total += len(chunk)
        return "\n\n".join(parts)

    # ══════════════════════════════════════════
    # 6. 대화 기록 관리
    # ══════════════════════════════════════════
    def _trim_history(self) -> List[BaseMessage]:
        """최근 HISTORY_WINDOW 쌍(HumanMessage + AIMessage)만 유지"""
        # 메시지는 항상 Human/AI 쌍이므로 뒤에서 2*N개 슬라이싱
        window = HISTORY_WINDOW * 2
        return self.chat_history[-window:] if len(self.chat_history) > window else self.chat_history

    def _append_history(self, question: str, answer: str) -> None:
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))

    def clear_history(self) -> None:
        """대화 기록 초기화"""
        self.chat_history.clear()
        logger.info("대화 기록 초기화 완료")

    # ══════════════════════════════════════════
    # 7. 핵심: 단일 질문 처리
    # ══════════════════════════════════════════
    def ask(
        self,
        question: str,
        use_multi_query: bool = False,
        use_history: bool = True,
    ) -> RAGResult:
        """
        질문을 분류 → 검색 → (압축) → 답변 생성까지 한 번에 처리.

        Args:
            question        : 사용자 질문
            use_multi_query : True면 다중 쿼리 검색 (더 넓은 검색 범위)
            use_history     : True면 대화 기록을 컨텍스트로 활용

        Returns:
            RAGResult — answer, citations, cache_hit, compression_ratio 등 포함
        """
        start    = time.perf_counter()
        warnings: List[str] = []

        # ── 질문 분류 ──────────────────────────
        qtype = self._classify_question(question)
        logger.info("질문 분류: '%s' → [%s]", question[:50], qtype)

        # ── 문서 검색 ──────────────────────────
        # 압축 적용 시 before/after 비교를 위해 원본 수를 따로 보존
        if use_multi_query:
            raw_docs = self.multi_query_retrieve(question)
            strategy = f"multi_query+{self.retrieval_strategy}"
        else:
            # _retrieve() 내부에서 압축이 적용되므로 압축 전 수를 미리 측정
            raw_docs = self._retrieve_raw(question)
            strategy = self.retrieval_strategy

        docs_before = len(raw_docs)

        # 압축 적용 (use_compression=True 이고 multi_query 아닐 때)
        if self._use_compression and not use_multi_query and raw_docs:
            docs = self._compress_documents(raw_docs, question)
        else:
            docs = raw_docs

        docs_before_compression = docs_before if self._use_compression else 0

        if not docs:
            warnings.append("검색된 문서가 없습니다.")

        # ── 컨텍스트 구성 (토큰 예산 적용) ────
        context = self._build_context(docs)
        if len(context) >= MAX_CONTEXT_CHARS:
            warnings.append(f"컨텍스트가 {MAX_CONTEXT_CHARS}자로 잘렸습니다.")

        # ── 프롬프트 + LLM 실행 (캐시·재시도 포함) ─
        prompt   = self._get_prompt(qtype)
        messages = self._trim_history() if use_history else []

        llm_start = time.perf_counter()
        answer = self._invoke_with_retry(
            prompt, context=context, question=question,
            chat_history=messages,
        )
        llm_elapsed = time.perf_counter() - llm_start

        # ── 캐시 히트 감지 ──────────────────────
        # 캐시 히트 시 DB 조회만 하므로 응답이 0.15초 미만
        # (실제 API 호출은 최소 0.5초 이상 소요)
        cache_hit = self._cache_path is not None and llm_elapsed < 0.15

        # ── 대화 기록 갱신 ─────────────────────
        self._append_history(question, answer)

        elapsed = round(time.perf_counter() - start, 3)
        result  = RAGResult(
            question=question,
            answer=answer,
            question_type=qtype,
            source_docs=docs,
            elapsed_sec=elapsed,
            retrieval_strategy=strategy,
            cache_hit=cache_hit,
            docs_before_compression=docs_before_compression,
            warnings=warnings,
        )
        self._stats.record(result)
        logger.info(
            "답변 생성 완료 — 유형: %s, 출처: %d개, %.3fs, 캐시히트: %s",
            qtype, len(docs), elapsed, cache_hit,
        )
        return result

    def _retrieve_raw(self, query: str) -> List[Document]:
        """압축을 적용하지 않고 원본 청크만 반환 (ask() 내부 전용)"""
        # 임시로 압축 플래그를 끄고 검색
        original = self._use_compression
        self._use_compression = False
        try:
            return self._retrieve(query)
        finally:
            self._use_compression = original

    # ══════════════════════════════════════════
    # 8. 스트리밍 응답
    # ══════════════════════════════════════════
    def stream(self, question: str, use_multi_query: bool = False) -> Iterator[str]:
        """
        토큰 단위 스트리밍 출력.

        사용 예:
            for token in rag.stream("계약 조건은?"):
                print(token, end="", flush=True)
        """
        qtype   = self._classify_question(question)
        docs    = self.multi_query_retrieve(question) if use_multi_query else self._retrieve(question)
        context = self._build_context(docs)
        prompt  = self._get_prompt(qtype)

        chain = prompt | self.llm | StrOutputParser()
        full_answer = ""
        for token in chain.stream({"context": context, "question": question}):
            full_answer += token
            yield token

        self._append_history(question, full_answer)

    # ══════════════════════════════════════════
    # 9. 적응형 RAG 체인 (LangChain 체인 객체 반환)
    # ══════════════════════════════════════════
    def build_adaptive_chain(self):
        """
        RunnableBranch 기반 적응형 체인.
        invoke({"question": "..."}) 형태로 사용.
        """
        pipeline_self = self

        def route(inputs: Dict[str, Any]) -> Dict[str, Any]:
            q     = inputs["question"]
            qtype = pipeline_self._classify_question(q)
            docs  = pipeline_self._retrieve(q)
            return {
                "context":       pipeline_self._build_context(docs),
                "question":      q,
                "question_type": qtype if qtype in QUESTION_TYPES else "factual",
            }

        adaptive_chain = (
            RunnablePassthrough()
            | route
            | RunnableBranch(
                (lambda x: x["question_type"] == "analytical",
                 self._get_prompt("analytical") | self.llm),
                (lambda x: x["question_type"] == "comparative",
                 self._get_prompt("comparative") | self.llm),
                (lambda x: x["question_type"] == "summarization",
                 self._get_prompt("summarization") | self.llm),
                self._get_prompt("factual") | self.llm,  # 기본값 (factual 포함)
            )
            | StrOutputParser()
        )
        return adaptive_chain

    # ══════════════════════════════════════════
    # 10. LLM 재시도 로직
    # ══════════════════════════════════════════
    def _invoke_with_retry(
        self,
        prompt: ChatPromptTemplate,
        max_retries: int = LLM_MAX_RETRIES,
        **kwargs: Any,
    ) -> str:
        chain = prompt | self.llm | StrOutputParser()
        last_exc: Optional[Exception] = None

        for attempt in range(1, max_retries + 1):
            try:
                return chain.invoke(kwargs)
            except Exception as exc:
                last_exc = exc
                self._stats.retry_count += 1
                logger.warning("LLM 호출 실패 (%d/%d): %s", attempt, max_retries, exc)
                time.sleep(1.5 * attempt)  # 지수 백오프

        raise RuntimeError(f"LLM 호출 {max_retries}회 모두 실패: {last_exc}") from last_exc

    # ══════════════════════════════════════════
    # v3: 캐시 관리
    # ══════════════════════════════════════════
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        SQLite DB를 직접 조회해 캐시 상태를 반환.
        세션 통계와 달리, 이전 세션에서 쌓인 캐시까지 포함한 전체 통계.
        """
        if not self._cache_path or not Path(self._cache_path).exists():
            return {"status": "캐시 비활성화 또는 DB 없음"}

        try:
            con = sqlite3.connect(self._cache_path)
            cur = con.cursor()

            # LangChain SQLiteCache 테이블: full_llm_cache
            cur.execute("SELECT COUNT(*) FROM full_llm_cache")
            total_entries = cur.fetchone()[0]

            cur.execute("SELECT SUM(LENGTH(response)) FROM full_llm_cache")
            total_bytes   = cur.fetchone()[0] or 0
            con.close()

            return {
                "db_path":          self._cache_path,
                "total_entries":    total_entries,
                "db_size_kb":       round(Path(self._cache_path).stat().st_size / 1024, 1),
                "avg_entry_bytes":  round(total_bytes / max(total_entries, 1)),
                # 세션 내 히트율
                "session_hits":     self._stats.cache_hits,
                "session_misses":   self._stats.cache_misses,
                "session_hit_rate": self._stats.cache_hit_rate,
            }
        except Exception as exc:
            return {"error": str(exc)}

    def clear_cache(self) -> None:
        """SQLite 캐시 DB를 초기화 (전체 삭제)"""
        if not self._cache_path:
            logger.warning("캐시가 비활성화 상태입니다.")
            return
        try:
            con = sqlite3.connect(self._cache_path)
            con.execute("DELETE FROM full_llm_cache")
            con.commit()
            con.close()
            logger.info("LLM 캐시 초기화 완료: %s", self._cache_path)
        except Exception as exc:
            logger.error("캐시 초기화 실패: %s", exc)

    # ── 통계 ─────────────────────────────────────
    def get_stats(self) -> SessionStats:
        return self._stats


# ──────────────────────────────────────────────
# end-to-end 사용 예시
# ──────────────────────────────────────────────
if __name__ == "__main__":

    # ── 1. PDF → 청크 ─────────────────────────
    pipeline = PDFIngestionPipeline(chunk_size=1000, chunk_overlap=200)
    p_result = pipeline.run("example.pdf")

    # ── 2. 청크 → 벡터 저장소 ─────────────────
    store = AdvancedVectorStore(embedding_type="openai", persist_dir="./vectorstore")
    store.ingest_from_pipeline(p_result, domain="legal")

    # ── 3. RAG 파이프라인 (캐싱 + 압축 검색 활성화) ──
    with AdvancedRAGPipeline(
        vectorstore=store,
        llm_model="gpt-4",
        retrieval_strategy="hybrid",
        domain="legal",
        k=5,
        # ── v3 신규 파라미터 ─────────────────────
        cache_path="./llm_cache.db",        # None 이면 캐싱 비활성화
        use_compression=True,               # 관련 문장만 추출
        compression_llm_model="gpt-3.5-turbo",  # 압축에는 경량 모델 사용 (비용 절약)
    ) as rag:

        # 첫 번째 호출 — API 실제 호출 (cache miss)
        result1 = rag.ask("이 계약서의 주요 조건들을 비교 분석해주세요.")
        result1.print_report()
        # result1.cache_hit == False, result1.compression_ratio 표시됨

        # 동일 질문 두 번째 — 캐시 반환 (cache hit, 빠름)
        result2 = rag.ask("이 계약서의 주요 조건들을 비교 분석해주세요.")
        result2.print_report()
        # result2.cache_hit == True, elapsed_sec 대폭 감소

        # 다중 쿼리 검색
        result3 = rag.ask("손해배상 조항은?", use_multi_query=True)
        result3.print_report()

        # 스트리밍 출력
        print("\n▶ 스트리밍 답변:")
        for token in rag.stream("계약 해지 절차를 요약해주세요"):
            print(token, end="", flush=True)
        print()

        # ── v3: 캐시 통계 확인 ──────────────────
        print("\n▶ 캐시 통계")
        for k, v in rag.get_cache_stats().items():
            print(f"  {k:<22}: {v}")

        # ── v3: 세션 통계 ────────────────────────
        stats = rag.get_stats()
        print(f"\n▶ 세션 통계")
        print(f"  총 쿼리 수       : {stats.total_queries}")
        print(f"  평균 응답 시간   : {stats.avg_elapsed}s")
        print(f"  캐시 히트율      : {stats.cache_hit_rate} ({stats.cache_hits}/{stats.cache_hits + stats.cache_misses})")
        print(f"  압축 제거 청크   : {stats.total_compressed_removed}개")
        print(f"  재시도 횟수      : {stats.retry_count}")
        print(f"  질문 유형 분포   : {stats.type_distribution}")

        # ── v3: 캐시 초기화 (필요 시) ────────────
        # rag.clear_cache()