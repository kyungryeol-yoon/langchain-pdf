"""
AdvancedPDFProcessor v2
───────────────────────
다양한 LangChain PDF 로더를 상황에 맞게 자동 선택·폴백하는 고급 PDF 처리 모듈.

개선 사항:
  1. 점수 기반(Scoring) 로더 선택  — 단순 if/else → 다중 지표 합산
  2. 자동 폴백(Fallback)           — 로더 실패 시 우선순위 순서로 재시도
  3. 전체 페이지 샘플링 분석       — 첫 페이지만 → 최대 N페이지 균등 샘플링
  4. 파일 유효성 검사              — 존재·확장자·암호화 여부 사전 체크
  5. 분석 결과 캐싱               — 동일 파일 재분석 비용 제거 (lru_cache)
  6. 배치 처리                    — 여러 PDF를 병렬(ThreadPool) 처리
  7. 로깅 시스템                  — print() → logging 모듈
  8. 컨텍스트 매니저              — with 문으로 리소스 안전 관리
  9. 통계 추적                    — 로더별 사용 횟수·성공률 기록
 10. 타입 힌트 강화               — dataclass, Optional, Tuple 활용
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import PyPDF2
from langchain_community.document_loaders import (
    PDFMinerLoader,
    PDFPlumberLoader,
    PyMuPDFLoader,
    PyPDFLoader,
    UnstructuredPDFLoader,
)
from langchain_core.documents import Document

# ──────────────────────────────────────────────
# 로깅 설정
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("AdvancedPDFProcessor")


# ──────────────────────────────────────────────
# 데이터 클래스
# ──────────────────────────────────────────────
@dataclass
class PDFAnalysis:
    """PDF 구조 분석 결과"""
    path: str
    page_count: int
    is_encrypted: bool
    has_images: bool
    has_tables: bool
    text_density: float           # 샘플 페이지당 평균 글자 수 / 1000
    scanned_pages_ratio: float    # 텍스트가 거의 없는 페이지 비율 (스캔본 추정)
    recommended_loader: str
    loader_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class LoadResult:
    """PDF 로드 결과"""
    documents: List[Document]
    loader_used: str
    elapsed_sec: float
    fallback_used: bool = False
    error_log: List[str] = field(default_factory=list)


# ──────────────────────────────────────────────
# 메인 클래스
# ──────────────────────────────────────────────
class AdvancedPDFProcessor:
    """
    다중 로더 자동 선택 + 폴백 기능을 갖춘 PDF 처리기.

    사용 예시:
        with AdvancedPDFProcessor() as processor:
            result = processor.load("report.pdf")
            meta   = processor.extract_metadata("report.pdf")
    """

    # 로더 폴백 우선순위
    LOADER_PRIORITY: List[str] = [
        "pypdf", "plumber", "pymupdf", "pdfminer", "unstructured"
    ]

    # 분석 임계값
    DENSITY_LOW          = 0.05  # 이 미만 → 스캔본·이미지 중심 PDF
    DENSITY_HIGH         = 2.5   # 이 초과 → 고밀도 텍스트 PDF
    TABLE_PIPE_THRESHOLD = 8     # '|' 개수 기준 (표 감지)
    SAMPLE_PAGES         = 5     # 분석에 사용할 최대 페이지 수

    def __init__(self, max_workers: int = 4) -> None:
        self._loaders: Dict[str, type] = {
            "pypdf":        PyPDFLoader,
            "unstructured": UnstructuredPDFLoader,
            "plumber":      PDFPlumberLoader,
            "pymupdf":      PyMuPDFLoader,
            "pdfminer":     PDFMinerLoader,
        }
        self._max_workers = max_workers
        self._stats: Dict[str, Dict[str, int]] = {
            name: {"success": 0, "failure": 0} for name in self._loaders
        }
        logger.info("AdvancedPDFProcessor 초기화 완료 (로더 %d개)", len(self._loaders))

    # ── 컨텍스트 매니저 ──────────────────────────
    def __enter__(self) -> "AdvancedPDFProcessor":
        return self

    def __exit__(self, *_) -> None:
        logger.info("세션 통계: %s", self.get_stats())

    # ── 파일 유효성 검사 ─────────────────────────
    @staticmethod
    def _validate_path(pdf_path: str) -> Path:
        """파일 존재·확장자 검사 후 Path 반환"""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없음: {pdf_path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"PDF 파일이 아님: {pdf_path}")
        return path

    # ── 핵심 분석 (캐싱 적용) ───────────────────
    def analyze(self, pdf_path: str) -> PDFAnalysis:
        """
        PDF 구조를 분석하고 최적 로더를 추천.
        동일 경로는 캐싱되어 재분석 비용 없음.
        """
        abs_path = str(self._validate_path(pdf_path).resolve())
        return self._cached_analyze(abs_path)

    @lru_cache(maxsize=128)
    def _cached_analyze(self, abs_path: str) -> PDFAnalysis:
        """절대 경로 기준으로 분석 결과 캐싱"""
        logger.info("PDF 분석 시작: %s", abs_path)

        with open(abs_path, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)

            # 암호화된 PDF는 분석 없이 unstructured 추천
            if reader.is_encrypted:
                logger.warning("암호화된 PDF 감지: %s", abs_path)
                return PDFAnalysis(
                    path=abs_path, page_count=0, is_encrypted=True,
                    has_images=False, has_tables=False,
                    text_density=0.0, scanned_pages_ratio=0.0,
                    recommended_loader="unstructured",
                )

            page_count = len(reader.pages)
            sample_indices = self._sample_page_indices(page_count)

            texts: List[str] = [
                reader.pages[i].extract_text() or ""
                for i in sample_indices
            ]

        # ── 지표 계산 ──────────────────────────
        all_text      = "\n".join(texts)
        avg_density   = (len(all_text) / max(len(texts), 1)) / 1000
        # 텍스트가 20자 미만인 페이지 → 스캔본으로 추정
        scanned_ratio = sum(1 for t in texts if len(t.strip()) < 20) / max(len(texts), 1)
        has_tables    = (
            "table" in all_text.lower()
            or all_text.count("|")  > self.TABLE_PIPE_THRESHOLD
            or all_text.count("\t") > 20
        )
        has_images    = scanned_ratio > 0.3

        # ── 점수 기반 로더 선택 ────────────────
        scores      = self._score_loaders(avg_density, has_tables, has_images, scanned_ratio)
        recommended = max(scores, key=scores.__getitem__)

        analysis = PDFAnalysis(
            path=abs_path,
            page_count=page_count,
            is_encrypted=False,
            has_images=has_images,
            has_tables=has_tables,
            text_density=round(avg_density, 4),
            scanned_pages_ratio=round(scanned_ratio, 4),
            recommended_loader=recommended,
            loader_scores={k: round(v, 2) for k, v in scores.items()},
        )
        logger.info(
            "분석 완료 — 페이지: %d, 밀도: %.3f, 추천 로더: %s",
            page_count, avg_density, recommended,
        )
        return analysis

    def _sample_page_indices(self, page_count: int) -> List[int]:
        """전체 페이지에서 균등 샘플 인덱스 반환"""
        if page_count <= self.SAMPLE_PAGES:
            return list(range(page_count))
        step = page_count // self.SAMPLE_PAGES
        return [i * step for i in range(self.SAMPLE_PAGES)]

    def _score_loaders(
        self,
        density: float,
        has_tables: bool,
        has_images: bool,
        scanned_ratio: float,
    ) -> Dict[str, float]:
        """
        각 로더에 점수를 부여해 최적 로더를 결정.

        로더별 강점:
          pypdf       — 일반 텍스트 PDF에 빠르고 안정적 (기본값)
          plumber     — 표·레이아웃 보존 우수
          pymupdf     — 이미지 포함 PDF, 처리 속도 우수
          pdfminer    — 고밀도 텍스트 정밀 파싱
          unstructured— 스캔본·복잡 레이아웃의 최후 수단
        """
        scores: Dict[str, float] = {
            "pypdf": 1.0, "plumber": 0.0,
            "pymupdf": 0.0, "pdfminer": 0.0, "unstructured": 0.0,
        }

        if has_tables:
            scores["plumber"]      += 2.0
            scores["pymupdf"]      += 0.5

        if has_images:
            scores["pymupdf"]      += 2.0
            scores["unstructured"] += 1.0

        if scanned_ratio > 0.5:          # 스캔본 비율 높음
            scores["unstructured"] += 3.0
            scores["pymupdf"]      += 1.0

        if density > self.DENSITY_HIGH:  # 고밀도 텍스트
            scores["pdfminer"]     += 2.0
            scores["pymupdf"]      += 1.0

        if density < self.DENSITY_LOW:   # 저밀도 (이미지 위주)
            scores["pymupdf"]      += 1.5
            scores["unstructured"] += 1.5

        # 일반 텍스트 PDF 보너스
        if (
            not has_tables and not has_images
            and self.DENSITY_LOW <= density <= self.DENSITY_HIGH
        ):
            scores["pypdf"] += 1.5

        return scores

    # ── PDF 로드 (자동 폴백 포함) ───────────────
    def load(self, pdf_path: str) -> LoadResult:
        """
        분석 결과에 따라 최적 로더로 PDF 로드.
        실패 시 LOADER_PRIORITY 순서로 자동 폴백.
        """
        analysis  = self.analyze(pdf_path)
        order     = self._fallback_order(analysis.recommended_loader)
        error_log: List[str] = []
        start     = time.perf_counter()

        for attempt, loader_name in enumerate(order):
            try:
                docs    = self._run_loader(loader_name, pdf_path)
                elapsed = time.perf_counter() - start
                self._stats[loader_name]["success"] += 1
                logger.info(
                    "로드 성공 — 로더: %s, 문서: %d개, %.2fs",
                    loader_name, len(docs), elapsed,
                )
                return LoadResult(
                    documents=docs,
                    loader_used=loader_name,
                    elapsed_sec=round(elapsed, 3),
                    fallback_used=(attempt > 0),
                    error_log=error_log,
                )
            except Exception as exc:
                msg = f"[{loader_name}] 실패: {exc}"
                error_log.append(msg)
                self._stats[loader_name]["failure"] += 1
                logger.warning(msg)

        raise RuntimeError(
            f"모든 로더 실패: {pdf_path}\n" + "\n".join(error_log)
        )

    def _fallback_order(self, recommended: str) -> List[str]:
        """추천 로더를 앞에 두고, 나머지를 우선순위 순으로 정렬"""
        return [recommended] + [n for n in self.LOADER_PRIORITY if n != recommended]

    def _run_loader(self, loader_name: str, pdf_path: str) -> List[Document]:
        return self._loaders[loader_name](pdf_path).load()

    # ── 메타데이터 추출 ──────────────────────────
    def extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """PDF 메타데이터 추출 (None 안전 처리 포함)"""
        path = self._validate_path(pdf_path)

        with open(path, "rb") as fh:
            reader   = PyPDF2.PdfReader(fh)
            raw_meta = reader.metadata or {}

            def safe(key: str) -> str:
                val = raw_meta.get(key, "")
                return str(val).strip() if val else ""

            return {
                "title":             safe("/Title"),
                "author":            safe("/Author"),
                "subject":           safe("/Subject"),
                "creator":           safe("/Creator"),
                "producer":          safe("/Producer"),
                "creation_date":     safe("/CreationDate"),
                "modification_date": safe("/ModDate"),
                "page_count":        len(reader.pages),
                "is_encrypted":      reader.is_encrypted,
                "file_size_kb":      round(path.stat().st_size / 1024, 2),
            }

    # ── 배치 처리 ────────────────────────────────
    def load_batch(
        self,
        pdf_paths: List[str],
        parallel: bool = True,
    ) -> List[Tuple[str, Optional[LoadResult], Optional[str]]]:
        """
        여러 PDF를 한 번에 처리.

        Args:
            pdf_paths: PDF 파일 경로 리스트
            parallel:  True이면 ThreadPoolExecutor로 병렬 처리

        Returns:
            [(경로, LoadResult or None, 에러 메시지 or None), ...]
        """
        results: List[Tuple[str, Optional[LoadResult], Optional[str]]] = []

        if parallel:
            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                future_map = {executor.submit(self.load, p): p for p in pdf_paths}
                for future in as_completed(future_map):
                    path = future_map[future]
                    try:
                        results.append((path, future.result(), None))
                    except Exception as exc:
                        results.append((path, None, str(exc)))
        else:
            for path in pdf_paths:
                try:
                    results.append((path, self.load(path), None))
                except Exception as exc:
                    results.append((path, None, str(exc)))

        return results

    # ── 통계 ─────────────────────────────────────
    def get_stats(self) -> Dict[str, Any]:
        """로더별 성공/실패 통계 반환"""
        stats: Dict[str, Any] = {}
        for name, counts in self._stats.items():
            total = counts["success"] + counts["failure"]
            stats[name] = {
                **counts,
                "total":        total,
                "success_rate": f"{counts['success'] / total:.0%}" if total else "N/A",
            }
        return stats


# ──────────────────────────────────────────────
# 사용 예시
# ──────────────────────────────────────────────
if __name__ == "__main__":
    PDF = "example.pdf"

    with AdvancedPDFProcessor() as processor:

        # 1. 구조 분석
        analysis = processor.analyze(PDF)
        print("\n▶ PDF 분석 결과")
        print(f"  페이지 수       : {analysis.page_count}")
        print(f"  텍스트 밀도     : {analysis.text_density}")
        print(f"  표 포함 여부    : {analysis.has_tables}")
        print(f"  이미지 포함     : {analysis.has_images}")
        print(f"  스캔본 추정 비율: {analysis.scanned_pages_ratio:.0%}")
        print(f"  로더 점수       : {analysis.loader_scores}")
        print(f"  추천 로더       : {analysis.recommended_loader}")

        # 2. 로드 (자동 폴백 포함)
        result = processor.load(PDF)
        print("\n▶ 로드 결과")
        print(f"  사용된 로더     : {result.loader_used}")
        print(f"  폴백 발생 여부  : {result.fallback_used}")
        print(f"  소요 시간       : {result.elapsed_sec}s")
        print(f"  문서 수         : {len(result.documents)}")

        # 3. 메타데이터
        meta = processor.extract_metadata(PDF)
        print("\n▶ 메타데이터")
        for k, v in meta.items():
            print(f"  {k:<22}: {v}")

        # 4. 배치 처리 예시
        # batch = processor.load_batch(["a.pdf", "b.pdf", "c.pdf"])
        # for path, res, err in batch:
        #     print(path, res.loader_used if res else err)

        # 5. 세션 통계 (with 블록 종료 시 자동 출력)