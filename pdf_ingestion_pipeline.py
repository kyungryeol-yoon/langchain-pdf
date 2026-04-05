"""
PDFIngestionPipeline
────────────────────
AdvancedPDFProcessor(로드·분석) + IntelligentTextSplitter(분할)를
단일 파이프라인으로 연결하는 통합 모듈.

연계 핵심:
  - PDFAnalysis 결과(has_tables, has_images, scanned_ratio)를
    Splitter 전략 선택에 직접 주입 → 중복 분석 0회
  - PDF 메타데이터(제목·저자·페이지 등)를 모든 청크에 자동 삽입
  - run() 한 번으로 로드 → 분석 → 분할 → 메타데이터 enrichment 완료
  - 배치(run_batch)도 ThreadPoolExecutor로 병렬 처리
  - PipelineResult 단일 객체로 모든 결과·통계 반환

의존 모듈:
  advanced_pdf_processor.py   — PDF 로드·구조 분석
  intelligent_text_splitter.py — 텍스트 분할 전략
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

# 두 모듈 임포트
from advanced_pdf_processor import AdvancedPDFProcessor, LoadResult, PDFAnalysis
from intelligent_text_splitter import IntelligentTextSplitter, SplitResult

# ──────────────────────────────────────────────
# 로깅
# ──────────────────────────────────────────────
logger = logging.getLogger("PDFIngestionPipeline")


# ──────────────────────────────────────────────
# 데이터 클래스
# ──────────────────────────────────────────────
@dataclass
class PipelineResult:
    """
    파이프라인 전체 결과.
    PDF 분석 / 로드 / 분할 / 메타데이터를 한 객체로 제공.
    """
    # ── 입력 ──
    pdf_path: str

    # ── PDF 분석 ──
    pdf_analysis: PDFAnalysis
    pdf_metadata: Dict[str, Any]

    # ── 로드 ──
    load_result: LoadResult

    # ── 분할 ──
    split_result: SplitResult

    # ── 타이밍 ──
    total_elapsed_sec: float

    # ── 경고 ──
    warnings: List[str] = field(default_factory=list)

    # ── 편의 프로퍼티 ────────────────────────────
    @property
    def chunks(self) -> List[Document]:
        """최종 청크 리스트 (가장 자주 쓰는 결과물)"""
        return self.split_result.documents

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    def summary(self) -> str:
        return (
            f"📄 {Path(self.pdf_path).name} | "
            f"페이지 {self.pdf_analysis.page_count}p | "
            f"로더 [{self.load_result.loader_used}] | "
            f"분할 전략 [{self.split_result.strategy_used}] | "
            f"청크 {self.chunk_count}개 | "
            f"총 {self.total_elapsed_sec:.2f}s"
        )

    def print_report(self) -> None:
        """전체 처리 결과를 보기 좋게 출력"""
        sep = "─" * 55
        print(f"\n{sep}")
        print(f"  PDF 처리 리포트")
        print(sep)

        print(f"\n[파일]")
        print(f"  경로          : {self.pdf_path}")
        print(f"  크기          : {self.pdf_metadata.get('file_size_kb', '?')} KB")

        print(f"\n[PDF 분석]")
        a = self.pdf_analysis
        print(f"  페이지 수     : {a.page_count}")
        print(f"  텍스트 밀도   : {a.text_density}")
        print(f"  표 포함       : {a.has_tables}")
        print(f"  이미지 포함   : {a.has_images}")
        print(f"  스캔본 비율   : {a.scanned_pages_ratio:.0%}")
        print(f"  추천 로더     : {a.recommended_loader}")
        print(f"  로더 점수     : {a.loader_scores}")

        print(f"\n[메타데이터]")
        for k, v in self.pdf_metadata.items():
            if v:
                print(f"  {k:<22}: {v}")

        print(f"\n[로드]")
        lr = self.load_result
        print(f"  사용된 로더   : {lr.loader_used}")
        print(f"  폴백 발생     : {lr.fallback_used}")
        print(f"  소요 시간     : {lr.elapsed_sec}s")
        if lr.error_log:
            print(f"  에러 로그     : {lr.error_log}")

        print(f"\n[분할]")
        sr = self.split_result
        print(f"  전략          : {sr.strategy_used}")
        print(f"  원본 문서 수  : {sr.original_count}")
        print(f"  청크 수       : {sr.result_count}")
        print(f"  평균 청크 크기: {sr.avg_chunk_size:.0f}자")
        print(f"  최소 / 최대   : {sr.min_chunk_size} / {sr.max_chunk_size}자")
        if sr.warnings:
            print(f"  경고          : {sr.warnings}")

        print(f"\n[총 소요 시간] {self.total_elapsed_sec:.3f}s")
        if self.warnings:
            print(f"\n[경고]\n  " + "\n  ".join(self.warnings))
        print(sep)


# ──────────────────────────────────────────────
# 메인 파이프라인
# ──────────────────────────────────────────────
class PDFIngestionPipeline:
    """
    PDF 로드 → 구조 분석 → 텍스트 분할 → 메타데이터 삽입을
    단일 run() 호출로 완료하는 통합 파이프라인.

    사용 예시:
        pipeline = PDFIngestionPipeline()
        result   = pipeline.run("report.pdf")
        chunks   = result.chunks   # 바로 벡터 DB에 넣을 수 있는 상태
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        max_workers: int = 4,
        min_section_length: int = 100,
    ) -> None:
        self._processor = AdvancedPDFProcessor(max_workers=max_workers)
        self._splitter  = IntelligentTextSplitter(
            default_chunk_size=chunk_size,
            default_chunk_overlap=chunk_overlap,
            min_section_length=min_section_length,
        )
        self._max_workers = max_workers
        logger.info(
            "PDFIngestionPipeline 초기화 — chunk_size=%d, overlap=%d",
            chunk_size, chunk_overlap,
        )

    # ── 컨텍스트 매니저 ──────────────────────────
    def __enter__(self) -> "PDFIngestionPipeline":
        return self

    def __exit__(self, *_) -> None:
        logger.info("PDFIngestionPipeline 세션 종료")

    # ══════════════════════════════════════════
    # 단일 PDF 처리
    # ══════════════════════════════════════════
    def run(
        self,
        pdf_path: str,
        split_steps: Optional[List[str]] = None,
        chunk_size: Optional[int] = None,
    ) -> PipelineResult:
        """
        PDF 한 파일을 완전 처리하여 PipelineResult 반환.

        Args:
            pdf_path   : 처리할 PDF 경로
            split_steps: 분할 파이프라인 단계 명시 (None이면 자동 선택)
                         예) ["section_aware", "semantic"]
            chunk_size : 이 호출에만 적용할 청크 크기 (None이면 기본값)

        Returns:
            PipelineResult — chunks 프로퍼티로 바로 청크 접근 가능
        """
        total_start = time.perf_counter()
        warnings: List[str] = []

        # ── Step 1: PDF 분석 ──────────────────────
        logger.info("[1/4] PDF 분석: %s", pdf_path)
        pdf_analysis = self._processor.analyze(pdf_path)

        # ── Step 2: PDF 로드 ──────────────────────
        logger.info("[2/4] PDF 로드: %s", pdf_path)
        load_result = self._processor.load(pdf_path)
        if load_result.fallback_used:
            warnings.append(f"폴백 로더 사용됨: {load_result.loader_used}")

        # ── Step 3: 메타데이터 추출 ───────────────
        logger.info("[3/4] 메타데이터 추출")
        pdf_metadata = self._processor.extract_metadata(pdf_path)

        # ── Step 4: 텍스트 분할 ──────────────────
        logger.info("[4/4] 텍스트 분할")
        documents = load_result.documents

        # PDF 분석 결과를 Splitter 전략 선택에 직접 주입
        # → Splitter가 자체 분석을 다시 하지 않아도 됨
        if split_steps:
            split_result = self._splitter.pipeline_split(
                documents, steps=split_steps, chunk_size=chunk_size
            )
        else:
            split_result = self._run_strategy_from_analysis(
                documents, pdf_analysis, chunk_size
            )

        # ── Step 5: 청크에 PDF 메타데이터 삽입 ───
        self._inject_pdf_metadata(split_result.documents, pdf_metadata, pdf_path)

        total_elapsed = round(time.perf_counter() - total_start, 3)
        logger.info("처리 완료: %s → 청크 %d개 (%.2fs)", pdf_path, len(split_result.documents), total_elapsed)

        result = PipelineResult(
            pdf_path=pdf_path,
            pdf_analysis=pdf_analysis,
            pdf_metadata=pdf_metadata,
            load_result=load_result,
            split_result=split_result,
            total_elapsed_sec=total_elapsed,
            warnings=warnings,
        )
        return result

    def _run_strategy_from_analysis(
        self,
        documents: List[Document],
        analysis: PDFAnalysis,
        chunk_size: Optional[int],
    ) -> SplitResult:
        """
        PDFAnalysis 결과를 직접 활용해 분할 전략 결정.
        IntelligentTextSplitter._analyze()를 다시 호출하지 않음.
        """
        # 분석 결과로 전략 직접 결정
        if analysis.has_tables and analysis.has_images:
            # 표 + 이미지 복합: 표 분리 후 의미 단위 분할
            steps = ["table_aware", "semantic"]
            logger.info("전략 결정 (PDF 분석 기반): table_aware → semantic")
            return self._splitter.pipeline_split(documents, steps=steps, chunk_size=chunk_size)

        elif analysis.has_tables:
            logger.info("전략 결정 (PDF 분석 기반): table_aware")
            docs = self._splitter.table_aware_split(documents)
            return self._splitter._build_result(docs, "table_aware", len(documents))

        elif analysis.scanned_pages_ratio > 0.5:
            # 스캔본 비율이 높으면 semantic만 (OCR 결과는 구조 파악 어려움)
            logger.info("전략 결정 (PDF 분석 기반): semantic (스캔본)")
            docs = self._splitter.semantic_split(documents, chunk_size=chunk_size)
            return self._splitter._build_result(docs, "semantic(scanned)", len(documents))

        elif analysis.has_images:
            # 이미지 위주 → 섹션 + 의미 단위
            steps = ["section_aware", "semantic"]
            logger.info("전략 결정 (PDF 분석 기반): section_aware → semantic")
            return self._splitter.pipeline_split(documents, steps=steps, chunk_size=chunk_size)

        elif analysis.text_density > 2.0:
            # 고밀도 텍스트 → 섹션 인식 후 의미 단위
            steps = ["section_aware", "semantic"]
            logger.info("전략 결정 (PDF 분석 기반): section_aware → semantic (고밀도)")
            return self._splitter.pipeline_split(documents, steps=steps, chunk_size=chunk_size)

        else:
            # 일반 텍스트
            logger.info("전략 결정 (PDF 분석 기반): auto_split")
            return self._splitter.auto_split(documents, chunk_size=chunk_size)

    @staticmethod
    def _inject_pdf_metadata(
        chunks: List[Document],
        pdf_metadata: Dict[str, Any],
        pdf_path: str,
    ) -> None:
        """
        모든 청크의 metadata에 PDF 정보를 삽입.
        기존 청크 메타데이터(chunk_index 등)는 유지.
        """
        inject = {
            "source":        pdf_path,
            "pdf_title":     pdf_metadata.get("title", ""),
            "pdf_author":    pdf_metadata.get("author", ""),
            "pdf_pages":     pdf_metadata.get("page_count", 0),
            "pdf_file_size": pdf_metadata.get("file_size_kb", 0),
        }
        for chunk in chunks:
            # setdefault: 청크 자체가 이미 source를 가지면 덮어쓰지 않음
            for k, v in inject.items():
                chunk.metadata.setdefault(k, v)

    # ══════════════════════════════════════════
    # 배치 처리
    # ══════════════════════════════════════════
    def run_batch(
        self,
        pdf_paths: List[str],
        split_steps: Optional[List[str]] = None,
        chunk_size: Optional[int] = None,
        parallel: bool = True,
    ) -> List[Tuple[str, Optional[PipelineResult], Optional[str]]]:
        """
        여러 PDF를 한 번에 처리.

        Returns:
            [(경로, PipelineResult or None, 에러 메시지 or None), ...]
        """
        results: List[Tuple[str, Optional[PipelineResult], Optional[str]]] = []

        def _process(path: str):
            return self.run(path, split_steps=split_steps, chunk_size=chunk_size)

        if parallel:
            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                future_map = {executor.submit(_process, p): p for p in pdf_paths}
                for future in as_completed(future_map):
                    path = future_map[future]
                    try:
                        results.append((path, future.result(), None))
                    except Exception as exc:
                        logger.error("배치 처리 실패 [%s]: %s", path, exc)
                        results.append((path, None, str(exc)))
        else:
            for path in pdf_paths:
                try:
                    results.append((path, _process(path), None))
                except Exception as exc:
                    logger.error("처리 실패 [%s]: %s", path, exc)
                    results.append((path, None, str(exc)))

        success = sum(1 for _, r, _ in results if r is not None)
        logger.info("배치 완료: %d/%d 성공", success, len(pdf_paths))
        return results

    # ══════════════════════════════════════════
    # 모든 청크를 하나의 리스트로 수집 (편의 메서드)
    # ══════════════════════════════════════════
    def collect_chunks(
        self,
        pdf_paths: List[str],
        **kwargs,
    ) -> List[Document]:
        """
        여러 PDF를 처리한 뒤 모든 청크를 단일 리스트로 반환.
        벡터 DB에 한꺼번에 삽입할 때 유용.

        사용 예:
            chunks = pipeline.collect_chunks(["a.pdf", "b.pdf"])
            vectorstore.add_documents(chunks)
        """
        batch = self.run_batch(pdf_paths, **kwargs)
        all_chunks: List[Document] = []
        for path, result, err in batch:
            if result:
                all_chunks.extend(result.chunks)
            else:
                logger.warning("청크 수집 실패, 건너뜀: %s — %s", path, err)
        logger.info("전체 수집 청크: %d개 (%d개 파일)", len(all_chunks), len(pdf_paths))
        return all_chunks


# ──────────────────────────────────────────────
# 사용 예시
# ──────────────────────────────────────────────
if __name__ == "__main__":

    # ── 단일 파일 처리 ───────────────────────────
    pipeline = PDFIngestionPipeline(chunk_size=1000, chunk_overlap=200)

    result = pipeline.run("example.pdf")
    result.print_report()

    # 청크 미리보기
    print("\n▶ 청크 미리보기 (최대 3개)")
    for chunk in result.chunks[:3]:
        m = chunk.metadata
        print(f"  [{m.get('chunk_index','?')}/{m.get('chunk_total','?')}] "
              f"전략={m.get('split_strategy','-')} | "
              f"출처={m.get('source','-')} | "
              f"{chunk.page_content[:80].replace(chr(10),' ')}…")

    # ── 전략 직접 지정 ───────────────────────────
    # result2 = pipeline.run(
    #     "report.pdf",
    #     split_steps=["section_aware", "semantic"],
    #     chunk_size=500,
    # )

    # ── 배치 처리 ────────────────────────────────
    # batch = pipeline.run_batch(["a.pdf", "b.pdf", "c.pdf"])
    # for path, res, err in batch:
    #     if res:
    #         print(f"✅ {path}: {res.chunk_count}청크")
    #     else:
    #         print(f"❌ {path}: {err}")

    # ── 벡터 DB 삽입용 청크 일괄 수집 ────────────
    # all_chunks = pipeline.collect_chunks(["a.pdf", "b.pdf"])
    # vectorstore.add_documents(all_chunks)