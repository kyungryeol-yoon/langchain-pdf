"""
AsyncPipelineRunner
───────────────────
asyncio.gather() 기반 진짜 비동기 병렬 처리 +
psutil 실시간 메모리 모니터링을 기존 파이프라인에 통합한 모듈.

기존 run_batch()와의 차이:
  ┌─────────────────────┬──────────────────────┬──────────────────────┐
  │                     │ 기존 run_batch()      │ AsyncPipelineRunner  │
  ├─────────────────────┼──────────────────────┼──────────────────────┤
  │ 병렬 방식           │ ThreadPoolExecutor   │ asyncio.gather()     │
  │ I/O 블로킹          │ 스레드 점유           │ 이벤트 루프 양보     │
  │ 메모리 제어         │ 없음                 │ psutil 실시간 감시   │
  │ 메모리 임계 초과 시 │ OOM 크래시 가능      │ 자동 GC + 대기       │
  │ 진행 상황           │ 로그만               │ 실시간 콜백          │
  │ 처리량 제어         │ max_workers 고정     │ Semaphore 동적 조절  │
  └─────────────────────┴──────────────────────┴──────────────────────┘

psutil 활용 포인트:
  1. MemoryMonitor  — 백그라운드 태스크로 주기적 샘플링
  2. 임계 초과 감지 — 처리 전 메모리 확인 → GC → 재확인 → 대기
  3. 파일별 메모리 델타 기록 — 어떤 PDF가 메모리를 많이 쓰는지 추적
  4. 전체 세션 메모리 리포트 — peak / avg / delta 통계

asyncio 활용 포인트:
  1. asyncio.gather()    — 여러 PDF를 진짜 동시에 처리
  2. asyncio.Semaphore() — 동시 처리 수 제한 (메모리 연동)
  3. run_in_executor()   — 동기 PDF 로더를 이벤트 루프에 비블로킹 삽입
  4. asyncio.wait_for()  — 파일별 타임아웃 (무한 대기 방지)
  5. 백그라운드 태스크   — 메모리 모니터를 처리와 병렬로 실행
"""

from __future__ import annotations

import asyncio
import gc
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import psutil

from pdf_ingestion_pipeline import PDFIngestionPipeline, PipelineResult

# ──────────────────────────────────────────────
# 로깅
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("AsyncPipelineRunner")


# ──────────────────────────────────────────────
# 데이터 클래스
# ──────────────────────────────────────────────
@dataclass
class MemorySnapshot:
    """단일 시점 메모리 상태"""
    timestamp: float
    rss_mb: float          # 실제 물리 메모리 사용량 (MB)
    vms_mb: float          # 가상 메모리 크기 (MB)
    percent: float         # 시스템 전체 메모리 사용률 (%)
    available_mb: float    # 사용 가능한 시스템 메모리 (MB)

    @classmethod
    def capture(cls) -> "MemorySnapshot":
        """현재 프로세스 + 시스템 메모리 스냅샷"""
        proc   = psutil.Process()
        mem    = proc.memory_info()
        sys_vm = psutil.virtual_memory()
        return cls(
            timestamp    = time.perf_counter(),
            rss_mb       = mem.rss   / 1024 / 1024,
            vms_mb       = mem.vms   / 1024 / 1024,
            percent      = sys_vm.percent,
            available_mb = sys_vm.available / 1024 / 1024,
        )

    def __str__(self) -> str:
        return (
            f"RSS {self.rss_mb:.1f}MB | "
            f"시스템 {self.percent:.1f}% | "
            f"여유 {self.available_mb:.0f}MB"
        )


@dataclass
class FileProcessResult:
    """단일 PDF 비동기 처리 결과"""
    path: str
    status: str                          # "success" | "failed" | "timeout" | "skipped"
    pipeline_result: Optional[PipelineResult] = None
    error: Optional[str]                 = None
    elapsed_sec: float                   = 0.0
    mem_before: Optional[MemorySnapshot] = None
    mem_after:  Optional[MemorySnapshot] = None

    @property
    def mem_delta_mb(self) -> float:
        """이 파일 처리로 인한 메모리 증가량 (MB)"""
        if self.mem_before and self.mem_after:
            return round(self.mem_after.rss_mb - self.mem_before.rss_mb, 2)
        return 0.0

    @property
    def chunk_count(self) -> int:
        return self.pipeline_result.chunk_count if self.pipeline_result else 0


@dataclass
class BatchReport:
    """배치 전체 결과 + 메모리·성능 통계"""
    results: List[FileProcessResult]
    total_elapsed_sec: float
    memory_snapshots: List[MemorySnapshot] = field(default_factory=list)

    # ── 편의 프로퍼티 ──────────────────────────
    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.status == "success")

    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.results if r.status != "success")

    @property
    def total_chunks(self) -> int:
        return sum(r.chunk_count for r in self.results)

    @property
    def peak_memory_mb(self) -> float:
        if not self.memory_snapshots:
            return 0.0
        return max(s.rss_mb for s in self.memory_snapshots)

    @property
    def avg_memory_mb(self) -> float:
        if not self.memory_snapshots:
            return 0.0
        return round(sum(s.rss_mb for s in self.memory_snapshots) / len(self.memory_snapshots), 1)

    def print_report(self) -> None:
        sep = "─" * 60
        print(f"\n{sep}")
        print("  비동기 배치 처리 리포트")
        print(sep)

        print(f"\n[처리 결과]")
        print(f"  성공  : {self.success_count}/{len(self.results)}개")
        print(f"  실패  : {self.fail_count}개")
        print(f"  청크  : {self.total_chunks:,}개")
        print(f"  총 시간: {self.total_elapsed_sec:.2f}s")

        print(f"\n[메모리 통계]")
        print(f"  피크   : {self.peak_memory_mb:.1f} MB")
        print(f"  평균   : {self.avg_memory_mb:.1f} MB")
        if self.memory_snapshots:
            first = self.memory_snapshots[0].rss_mb
            last  = self.memory_snapshots[-1].rss_mb
            print(f"  전체 델타: {last - first:+.1f} MB")

        print(f"\n[파일별 결과]")
        for r in self.results:
            status_icon = "✓" if r.status == "success" else "✗"
            print(
                f"  {status_icon} {Path(r.path).name:<30} "
                f"청크={r.chunk_count:>4} | "
                f"메모리 델타={r.mem_delta_mb:>+6.1f}MB | "
                f"{r.elapsed_sec:.2f}s"
                + (f" | 오류: {r.error}" if r.error else "")
            )

        # 메모리를 많이 쓴 파일 경고
        heavy = [r for r in self.results if r.mem_delta_mb > 100]
        if heavy:
            print(f"\n[메모리 주의 파일]")
            for r in heavy:
                print(f"  {Path(r.path).name}: +{r.mem_delta_mb:.1f}MB")
        print(sep)


# ──────────────────────────────────────────────
# 메모리 모니터 (백그라운드 태스크)
# ──────────────────────────────────────────────
class MemoryMonitor:
    """
    asyncio 백그라운드 태스크로 주기적으로 메모리를 샘플링.
    처리 루프와 완전히 병렬로 실행되어 처리 성능에 영향 없음.
    """

    def __init__(self, interval_sec: float = 1.0) -> None:
        self.interval_sec = interval_sec
        self.snapshots: List[MemorySnapshot] = []
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        """백그라운드 모니터링 시작"""
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run())
        logger.debug("MemoryMonitor 시작 (간격: %.1fs)", self.interval_sec)

    async def stop(self) -> None:
        """백그라운드 모니터링 중지"""
        self._stop_event.set()
        if self._task:
            await self._task
        logger.debug("MemoryMonitor 중지 — 샘플 %d개", len(self.snapshots))

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            snap = MemorySnapshot.capture()
            self.snapshots.append(snap)
            logger.debug("메모리 샘플: %s", snap)
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._stop_event.wait()),
                    timeout=self.interval_sec,
                )
            except asyncio.TimeoutError:
                pass  # 정상적인 인터벌 종료

    def current(self) -> MemorySnapshot:
        return self.snapshots[-1] if self.snapshots else MemorySnapshot.capture()


# ──────────────────────────────────────────────
# 메인 클래스
# ──────────────────────────────────────────────
class AsyncPipelineRunner:
    """
    asyncio.gather() + psutil 메모리 모니터를 결합한
    고성능 비동기 PDF 일괄 처리기.

    사용 예시:
        runner = AsyncPipelineRunner(
            max_concurrent=4,
            max_memory_mb=2048,
        )
        report = asyncio.run(runner.run_async(["a.pdf", "b.pdf", "c.pdf"]))
        report.print_report()
    """

    def __init__(
        self,
        pipeline: Optional[PDFIngestionPipeline] = None,
        max_concurrent: int = 4,         # 동시 처리 PDF 수
        max_memory_mb: float = 2048.0,   # 메모리 임계값 (MB)
        memory_wait_sec: float = 2.0,    # 임계 초과 시 대기 시간 (초)
        file_timeout_sec: float = 120.0, # 파일 1개당 타임아웃 (초)
        monitor_interval: float = 1.0,   # 메모리 샘플링 간격 (초)
        on_progress: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self._pipeline        = pipeline or PDFIngestionPipeline()
        self._semaphore: Optional[asyncio.Semaphore] = None  # 이벤트 루프 생성 후 초기화
        self._max_concurrent  = max_concurrent
        self._max_memory_mb   = max_memory_mb
        self._memory_wait_sec = memory_wait_sec
        self._file_timeout    = file_timeout_sec
        self._monitor_interval = monitor_interval
        self._on_progress     = on_progress or self._default_progress
        logger.info(
            "AsyncPipelineRunner 초기화 — 동시처리: %d, 메모리 한계: %.0fMB",
            max_concurrent, max_memory_mb,
        )

    # ── 진행 콜백 기본값 ─────────────────────────
    @staticmethod
    def _default_progress(path: str, status: str) -> None:
        icon = {"start": "▶", "success": "✓", "failed": "✗",
                "timeout": "⏱", "memory_wait": "⏸"}.get(status, "·")
        logger.info("%s [%s] %s", icon, status, Path(path).name)

    # ══════════════════════════════════════════
    # 메모리 제어
    # ══════════════════════════════════════════
    async def _wait_for_memory(self) -> None:
        """
        현재 메모리 사용량이 임계값을 초과하면 GC 후 대기.
        asyncio.sleep()을 사용해 이벤트 루프를 블로킹하지 않음.
        """
        snap = MemorySnapshot.capture()
        if snap.rss_mb < self._max_memory_mb:
            return

        logger.warning(
            "메모리 임계 초과: %.1fMB / %.0fMB — GC 실행 후 대기",
            snap.rss_mb, self._max_memory_mb,
        )
        gc.collect()
        await asyncio.sleep(self._memory_wait_sec)

        # GC 후에도 초과이면 추가 대기
        snap2 = MemorySnapshot.capture()
        if snap2.rss_mb >= self._max_memory_mb:
            logger.warning("GC 후에도 초과 (%.1fMB) — %.1fs 추가 대기",
                           snap2.rss_mb, self._memory_wait_sec * 2)
            await asyncio.sleep(self._memory_wait_sec * 2)

    # ══════════════════════════════════════════
    # 단일 PDF 비동기 처리
    # ══════════════════════════════════════════
    async def _process_one(
        self,
        path: str,
        loop: asyncio.AbstractEventLoop,
        split_steps: Optional[List[str]],
        chunk_size: Optional[int],
    ) -> FileProcessResult:
        """
        Semaphore로 동시 처리 수를 제한하면서 단일 PDF 처리.
        동기 PDF 로더를 run_in_executor()로 감싸 이벤트 루프 비블로킹 실행.
        """
        assert self._semaphore is not None

        async with self._semaphore:
            # ── 메모리 체크 ──────────────────────
            await self._wait_for_memory()
            self._on_progress(path, "start")

            mem_before = MemorySnapshot.capture()
            start      = time.perf_counter()

            try:
                # ── run_in_executor: 동기 파이프라인을 스레드 풀에서 실행 ──
                # asyncio 이벤트 루프를 블로킹하지 않으면서
                # 기존 PDFIngestionPipeline.run()을 그대로 재사용
                pipeline_result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,  # 기본 ThreadPoolExecutor 사용
                        lambda: self._pipeline.run(
                            path,
                            split_steps=split_steps,
                            chunk_size=chunk_size,
                        ),
                    ),
                    timeout=self._file_timeout,
                )

                mem_after = MemorySnapshot.capture()
                elapsed   = round(time.perf_counter() - start, 3)
                self._on_progress(path, "success")

                return FileProcessResult(
                    path=path,
                    status="success",
                    pipeline_result=pipeline_result,
                    elapsed_sec=elapsed,
                    mem_before=mem_before,
                    mem_after=mem_after,
                )

            except asyncio.TimeoutError:
                elapsed = round(time.perf_counter() - start, 3)
                logger.error("타임아웃 [%.0fs 초과]: %s", self._file_timeout, path)
                self._on_progress(path, "timeout")
                return FileProcessResult(
                    path=path, status="timeout",
                    error=f"타임아웃 ({self._file_timeout}s 초과)",
                    elapsed_sec=elapsed,
                    mem_before=mem_before,
                    mem_after=MemorySnapshot.capture(),
                )

            except Exception as exc:
                elapsed = round(time.perf_counter() - start, 3)
                logger.error("처리 실패 [%s]: %s", Path(path).name, exc)
                self._on_progress(path, "failed")
                return FileProcessResult(
                    path=path, status="failed",
                    error=str(exc), elapsed_sec=elapsed,
                    mem_before=mem_before,
                    mem_after=MemorySnapshot.capture(),
                )

    # ══════════════════════════════════════════
    # 핵심: asyncio.gather() 병렬 처리
    # ══════════════════════════════════════════
    async def run_async(
        self,
        pdf_paths: List[str],
        split_steps: Optional[List[str]] = None,
        chunk_size: Optional[int] = None,
    ) -> BatchReport:
        """
        asyncio.gather()로 모든 PDF를 동시에 처리.
        Semaphore가 실제 동시 처리 수를 max_concurrent로 제한.

        Args:
            pdf_paths  : 처리할 PDF 경로 리스트
            split_steps: 분할 전략 명시 (None이면 자동)
            chunk_size : 청크 크기

        Returns:
            BatchReport — 전체 결과 + 메모리 통계
        """
        # Semaphore를 현재 이벤트 루프에서 생성
        self._semaphore = asyncio.Semaphore(self._max_concurrent)
        loop            = asyncio.get_event_loop()
        total_start     = time.perf_counter()

        logger.info(
            "비동기 배치 시작 — %d개 파일, 동시처리: %d, 메모리 한계: %.0fMB",
            len(pdf_paths), self._max_concurrent, self._max_memory_mb,
        )
        logger.info("초기 메모리: %s", MemorySnapshot.capture())

        # ── 백그라운드 메모리 모니터 시작 ────────
        monitor = MemoryMonitor(interval_sec=self._monitor_interval)
        await monitor.start()

        # ── asyncio.gather(): 모든 파일 동시 처리 ─
        # Semaphore가 max_concurrent 개수만큼만 실제 실행되도록 제어
        tasks = [
            self._process_one(path, loop, split_steps, chunk_size)
            for path in pdf_paths
        ]
        results: List[FileProcessResult] = list(
            await asyncio.gather(*tasks, return_exceptions=False)
        )

        # ── 백그라운드 모니터 종료 ────────────────
        await monitor.stop()

        total_elapsed = round(time.perf_counter() - total_start, 3)
        success = sum(1 for r in results if r.status == "success")
        logger.info(
            "배치 완료 — %d/%d 성공, 총 청크 %d개, %.2fs | 최종 메모리: %s",
            success, len(results),
            sum(r.chunk_count for r in results),
            total_elapsed,
            MemorySnapshot.capture(),
        )

        return BatchReport(
            results=results,
            total_elapsed_sec=total_elapsed,
            memory_snapshots=monitor.snapshots,
        )

    # ══════════════════════════════════════════
    # 동기 진입점 (asyncio.run() 래퍼)
    # ══════════════════════════════════════════
    def run(
        self,
        pdf_paths: List[str],
        split_steps: Optional[List[str]] = None,
        chunk_size: Optional[int] = None,
    ) -> BatchReport:
        """
        동기 컨텍스트에서 호출하는 편의 메서드.
        내부적으로 asyncio.run()으로 이벤트 루프를 생성·실행·종료.

        이미 이벤트 루프 안에 있다면 run_async()를 직접 await하세요.
        """
        return asyncio.run(
            self.run_async(pdf_paths, split_steps=split_steps, chunk_size=chunk_size)
        )

    # ══════════════════════════════════════════
    # 메모리 사용량 프로파일 (단독 실행)
    # ══════════════════════════════════════════
    def profile_memory(self, pdf_path: str) -> Dict[str, Any]:
        """
        단일 PDF의 처리 단계별 메모리 변화를 프로파일링.
        어떤 단계에서 메모리가 가장 많이 증가하는지 파악할 때 사용.
        """
        from advanced_pdf_processor import AdvancedPDFProcessor
        stages: Dict[str, float] = {}

        def snap(label: str) -> None:
            stages[label] = MemorySnapshot.capture().rss_mb

        snap("시작")

        proc = AdvancedPDFProcessor()
        snap("프로세서 초기화")

        analysis = proc.analyze(pdf_path)
        snap("PDF 분석 후")

        load_result = proc.load(pdf_path)
        snap("PDF 로드 후")

        meta = proc.extract_metadata(pdf_path)
        snap("메타데이터 추출 후")

        from intelligent_text_splitter import IntelligentTextSplitter
        splitter = IntelligentTextSplitter()
        split_result = splitter.auto_split(load_result.documents)
        snap("텍스트 분할 후")

        del load_result, split_result
        gc.collect()
        snap("GC 후")

        # 단계별 증감
        keys   = list(stages.keys())
        deltas = {
            f"{keys[i]}→{keys[i+1]}": round(stages[keys[i+1]] - stages[keys[i]], 2)
            for i in range(len(keys) - 1)
        }

        report = {
            "file":        pdf_path,
            "page_count":  analysis.page_count,
            "snapshots_mb": stages,
            "deltas_mb":   deltas,
            "peak_mb":     max(stages.values()),
            "net_delta_mb": round(stages["GC 후"] - stages["시작"], 2),
        }

        print(f"\n▶ 메모리 프로파일: {Path(pdf_path).name}")
        for label, mb in stages.items():
            print(f"  {label:<20}: {mb:.1f} MB")
        print(f"\n  단계별 증감:")
        for label, delta in deltas.items():
            bar = "▲" if delta > 0 else "▼" if delta < 0 else "─"
            print(f"  {bar} {label:<30}: {delta:>+7.1f} MB")
        print(f"\n  피크: {report['peak_mb']:.1f}MB | 순 증가: {report['net_delta_mb']:+.1f}MB")

        return report


# ──────────────────────────────────────────────
# 사용 예시
# ──────────────────────────────────────────────
if __name__ == "__main__":
    from pdf_ingestion_pipeline import PDFIngestionPipeline

    pipeline = PDFIngestionPipeline(chunk_size=1000, chunk_overlap=200)

    runner = AsyncPipelineRunner(
        pipeline=pipeline,
        max_concurrent=4,       # 동시에 최대 4개 파일 처리
        max_memory_mb=2048.0,   # 2GB 초과 시 GC + 대기
        file_timeout_sec=120.0, # 파일당 최대 2분
        monitor_interval=1.0,   # 1초마다 메모리 샘플링
    )

    # ── 1. 비동기 배치 처리 ──────────────────────
    pdf_files = ["doc1.pdf", "doc2.pdf", "doc3.pdf", "doc4.pdf", "doc5.pdf"]
    report    = runner.run(pdf_files)
    report.print_report()

    # ── 2. 성공한 결과만 모아 벡터 저장소로 ──────
    from advanced_vector_store import AdvancedVectorStore

    store = AdvancedVectorStore(embedding_type="openai", persist_dir="./vectorstore")
    for file_result in report.results:
        if file_result.status == "success" and file_result.pipeline_result:
            store.ingest_from_pipeline(file_result.pipeline_result, domain="docs")

    # ── 3. 메모리 프로파일 (단일 파일 분석) ──────
    runner.profile_memory("doc1.pdf")

    # ── 4. 이미 이벤트 루프 안에 있을 때 (예: Jupyter) ──
    # report = await runner.run_async(pdf_files)

    # ── 5. 진행 콜백 커스터마이징 ─────────────────
    # def my_progress(path, status):
    #     print(f"[{status.upper()}] {path}")
    #
    # runner2 = AsyncPipelineRunner(
    #     pipeline=pipeline,
    #     on_progress=my_progress,
    # )