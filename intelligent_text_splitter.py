"""
IntelligentTextSplitter v2
──────────────────────────
문서 유형을 자동 감지하여 최적 분할 전략을 선택하는 고급 텍스트 분할 모듈.

버그 수정:
  1. chunk_operlap  → chunk_overlap (오타)
  2. table_aware_split 에서 빈 part일 때 미정의 split_doc 참조 → 제거
  3. List 타입 힌트 누락 → from typing import List 추가

개선 사항:
  1. 자동 분할 전략 선택    — 문서 내용 분석 후 최적 메서드 자동 결정
  2. 파이프라인 처리        — 여러 분할 단계를 체이닝
  3. 코드 블록 인식 분할    — 마크다운 코드 펜스 보존
  4. 청크 메타데이터 강화   — 인덱스·전체 수·출처 자동 부여
  5. 언어 감지             — 한국어 여부에 따라 구분자 조정
  6. 로깅 시스템           — print() → logging 모듈
  7. 통계 리포트           — 분할 전후 상세 통계 반환
  8. 데이터클래스 결과      — 딕셔너리 대신 SplitResult dataclass
  9. 청크 크기 유효성 검사  — 비정상 설정 사전 차단
 10. 컨텍스트 매니저       — with 문 지원
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import (
    NLTKTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

# ──────────────────────────────────────────────
# 로깅 설정
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("IntelligentTextSplitter")


# ──────────────────────────────────────────────
# 데이터 클래스
# ──────────────────────────────────────────────
@dataclass
class SplitResult:
    """분할 결과 + 통계"""
    documents: List[Document]
    strategy_used: str
    original_count: int
    result_count: int
    avg_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"[{self.strategy_used}] "
            f"원본 {self.original_count}개 → {self.result_count}개 | "
            f"평균 {self.avg_chunk_size:.0f}자 "
            f"(최소 {self.min_chunk_size} / 최대 {self.max_chunk_size})"
        )


# ──────────────────────────────────────────────
# 메인 클래스
# ──────────────────────────────────────────────
class IntelligentTextSplitter:
    """
    문서 구조를 분석해 최적 분할 전략을 자동 선택하는 텍스트 분할기.

    사용 예시:
        with IntelligentTextSplitter() as splitter:
            result = splitter.auto_split(documents)
            print(result.summary())
    """

    # 섹션 헤더 패턴
    SECTION_PATTERNS: List[str] = [
        r"^#{1,6}\s+.+$",       # 마크다운 헤더 (# ~ ######)
        r"^\d+\.\s+.+$",        # 번호 목록  (1. 제목)
        r"^[A-Z][A-Z\s]{2,}:$", # 대문자 라벨 (CHAPTER:)
        r"^\s*\d+\.\d+[\s.]",   # 계층 번호  (1.1 / 1.2.3)
    ]

    # 코드 블록 패턴
    CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```", re.MULTILINE)

    # 표 패턴 (헤더 구분선 포함)
    TABLE_PATTERN = re.compile(
        r"(\|.+\|\n)([\|\-\s:]+\|\n)(\|.+\|\n)+",
        re.MULTILINE,
    )

    def __init__(
        self,
        default_chunk_size: int = 1000,
        default_chunk_overlap: int = 200,
        min_section_length: int = 100,
    ) -> None:
        self._validate_chunk_config(default_chunk_size, default_chunk_overlap)

        self.default_chunk_size    = default_chunk_size
        self.default_chunk_overlap = default_chunk_overlap
        self.min_section_length    = min_section_length

        logger.info(
            "IntelligentTextSplitter 초기화 — chunk_size=%d, overlap=%d",
            default_chunk_size, default_chunk_overlap,
        )

    # ── 컨텍스트 매니저 ──────────────────────────
    def __enter__(self) -> "IntelligentTextSplitter":
        return self

    def __exit__(self, *_) -> None:
        logger.info("IntelligentTextSplitter 세션 종료")

    # ── 유효성 검사 ──────────────────────────────
    @staticmethod
    def _validate_chunk_config(chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size는 양수여야 합니다: {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap은 0 이상이어야 합니다: {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap({chunk_overlap})은 chunk_size({chunk_size})보다 작아야 합니다."
            )

    # ── 언어 감지 ────────────────────────────────
    @staticmethod
    def _is_korean(text: str) -> bool:
        """텍스트에 한글이 일정 비율 이상 포함되어 있으면 True"""
        korean_chars = sum(1 for c in text if "\uAC00" <= c <= "\uD7A3")
        return korean_chars / max(len(text), 1) > 0.1

    # ── 구분자 선택 ──────────────────────────────
    def _get_separators(self, text: str) -> List[str]:
        """언어·포맷에 맞는 구분자 목록 반환"""
        base = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
        if self._is_korean(text):
            # 한국어 문장 종결 어미 우선
            return ["\n\n", "\n", ". ", "다. ", "요. ", "죠. ", ", ", " ", ""]
        return base

    # ── 문서 구조 분석 ───────────────────────────
    def _analyze(self, documents: List[Document]) -> Dict[str, Any]:
        """
        문서 전체를 샘플링해 구조 특성 파악.
        반환값은 auto_split()의 전략 선택에 사용.
        """
        sample_text = "\n".join(
            doc.page_content for doc in documents[:5]
        )
        has_tables   = bool(self.TABLE_PATTERN.search(sample_text))
        has_code     = bool(self.CODE_BLOCK_PATTERN.search(sample_text))
        has_sections = any(
            re.search(pat, sample_text, re.MULTILINE)
            for pat in self.SECTION_PATTERNS
        )
        avg_len      = (
            sum(len(d.page_content) for d in documents) / max(len(documents), 1)
        )
        return {
            "has_tables":   has_tables,
            "has_code":     has_code,
            "has_sections": has_sections,
            "avg_length":   avg_len,
            "is_korean":    self._is_korean(sample_text),
        }

    # ══════════════════════════════════════════
    # 1. 자동 전략 선택
    # ══════════════════════════════════════════
    def auto_split(
        self,
        documents: List[Document],
        chunk_size: Optional[int] = None,
    ) -> SplitResult:
        """
        문서를 분석해 최적 분할 전략을 자동 선택한 후 실행.

        선택 우선순위:
          1. 표 포함  → table_aware_split
          2. 코드 포함 → code_aware_split
          3. 섹션 구조 → section_aware_split
          4. 기본     → semantic_split
        """
        info = self._analyze(documents)
        logger.info("문서 분석 결과: %s", info)

        if info["has_tables"]:
            strategy, method = "table_aware", self.table_aware_split
        elif info["has_code"]:
            strategy, method = "code_aware", self.code_aware_split
        elif info["has_sections"]:
            strategy, method = "section_aware", self.section_aware_split
        else:
            strategy, method = "semantic", lambda d: self.semantic_split(
                d, chunk_size=chunk_size or self.default_chunk_size
            )

        logger.info("선택된 전략: %s", strategy)
        result_docs = method(documents)
        return self._build_result(result_docs, strategy, len(documents))

    # ══════════════════════════════════════════
    # 2. 의미 단위 분할
    # ══════════════════════════════════════════
    def semantic_split(
        self,
        documents: List[Document],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> List[Document]:
        """RecursiveCharacterTextSplitter 기반 의미 단위 분할"""
        size    = chunk_size    or self.default_chunk_size
        overlap = chunk_overlap or self.default_chunk_overlap
        self._validate_chunk_config(size, overlap)

        sample  = documents[0].page_content if documents else ""
        seps    = self._get_separators(sample)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,   # ✅ Fix: chunk_operlap → chunk_overlap
            separators=seps,
            length_function=len,
        )
        docs = splitter.split_documents(documents)
        return self._enrich_metadata(docs, "semantic")

    # ══════════════════════════════════════════
    # 3. 섹션 인식 분할
    # ══════════════════════════════════════════
    def section_aware_split(self, documents: List[Document]) -> List[Document]:
        """헤더·번호 목록 등 섹션 경계를 인식해 분할"""
        split_docs: List[Document] = []
        compiled   = [re.compile(p, re.MULTILINE) for p in self.SECTION_PATTERNS]

        for doc in documents:
            sections        = []
            current_lines: List[str] = []
            current_header  = ""

            for line in doc.page_content.split("\n"):
                is_header = any(pat.match(line.strip()) for pat in compiled)

                if is_header and current_lines:
                    # 현재 섹션 저장
                    sections.append((current_header, "\n".join(current_lines).strip()))
                    current_lines  = [line]
                    current_header = line.strip()
                else:
                    current_lines.append(line)

            # 마지막 섹션 처리
            if current_lines:
                sections.append((current_header, "\n".join(current_lines).strip()))

            for i, (header, body) in enumerate(sections):
                if len(body) < self.min_section_length:
                    logger.debug("섹션 스킵 (너무 짧음, %d자): %s…", len(body), body[:40])
                    continue

                split_docs.append(Document(
                    page_content=body,
                    metadata={
                        **doc.metadata,
                        "section_index":  i,
                        "section_header": header,
                        "section_type":   "content",
                        "split_strategy": "section_aware",
                    },
                ))

        return self._enrich_metadata(split_docs, "section_aware")

    # ══════════════════════════════════════════
    # 4. 표 인식 분할
    # ══════════════════════════════════════════
    def table_aware_split(self, documents: List[Document]) -> List[Document]:
        """표와 일반 텍스트를 분리해 각각 최적 처리"""
        split_docs: List[Document] = []

        for doc in documents:
            content = doc.page_content
            parts: List[Dict[str, str]] = []
            last_end = 0

            # 표 위치를 순서대로 추출
            for match in self.TABLE_PATTERN.finditer(content):
                start, end = match.start(), match.end()

                # 표 앞 텍스트
                pre_text = content[last_end:start].strip()
                if pre_text:
                    parts.append({"type": "text", "content": pre_text})

                # 표 자체
                parts.append({"type": "table", "content": match.group().strip()})
                last_end = end

            # 표 뒤 잔여 텍스트
            tail = content[last_end:].strip()
            if tail:
                parts.append({"type": "text", "content": tail})

            # 표가 없으면 원본 그대로
            if not parts:
                split_docs.append(doc)
                continue

            for i, part in enumerate(parts):
                if not part["content"]:
                    continue  # ✅ Fix: 빈 파트에서 미정의 split_doc 참조 제거

                split_docs.append(Document(
                    page_content=part["content"],
                    metadata={
                        **doc.metadata,
                        "content_type":   part["type"],
                        "part_index":     i,
                        "split_strategy": "table_aware",
                    },
                ))

        return self._enrich_metadata(split_docs, "table_aware")

    # ══════════════════════════════════════════
    # 5. 코드 블록 인식 분할 (신규)
    # ══════════════════════════════════════════
    def code_aware_split(self, documents: List[Document]) -> List[Document]:
        """
        마크다운 코드 블록(``` ```)을 보존하면서 나머지 텍스트만 분할.
        코드 블록은 절대 중간에 잘리지 않음.
        """
        split_docs: List[Document] = []

        for doc in documents:
            content = doc.page_content
            parts: List[Dict[str, str]] = []
            last_end = 0

            for match in self.CODE_BLOCK_PATTERN.finditer(content):
                start, end = match.start(), match.end()

                # 코드 블록 앞 텍스트 → semantic 분할
                pre = content[last_end:start].strip()
                if pre:
                    parts.append({"type": "text", "content": pre})

                # 코드 블록은 통째로 보존
                parts.append({"type": "code", "content": match.group().strip()})
                last_end = end

            tail = content[last_end:].strip()
            if tail:
                parts.append({"type": "text", "content": tail})

            if not parts:
                split_docs.append(doc)
                continue

            for i, part in enumerate(parts):
                if not part["content"]:
                    continue

                if part["type"] == "text":
                    # 텍스트 부분만 semantic 분할 적용
                    sub_docs = self.semantic_split(
                        [Document(page_content=part["content"], metadata=doc.metadata)]
                    )
                    for sub in sub_docs:
                        sub.metadata.update({
                            "content_type":   "text",
                            "part_index":     i,
                            "split_strategy": "code_aware",
                        })
                    split_docs.extend(sub_docs)
                else:
                    split_docs.append(Document(
                        page_content=part["content"],
                        metadata={
                            **doc.metadata,
                            "content_type":   "code",
                            "part_index":     i,
                            "split_strategy": "code_aware",
                        },
                    ))

        return self._enrich_metadata(split_docs, "code_aware")

    # ══════════════════════════════════════════
    # 6. 파이프라인 (신규)
    # ══════════════════════════════════════════
    def pipeline_split(
        self,
        documents: List[Document],
        steps: List[str],
        chunk_size: Optional[int] = None,
    ) -> SplitResult:
        """
        여러 분할 단계를 순서대로 적용.

        사용 예:
            result = splitter.pipeline_split(
                docs, steps=["section_aware", "semantic"]
            )
        """
        STEP_MAP = {
            "semantic":      lambda d: self.semantic_split(d, chunk_size=chunk_size),
            "section_aware": self.section_aware_split,
            "table_aware":   self.table_aware_split,
            "code_aware":    self.code_aware_split,
        }
        for step in steps:
            if step not in STEP_MAP:
                raise ValueError(
                    f"알 수 없는 분할 단계: '{step}'. "
                    f"사용 가능: {list(STEP_MAP.keys())}"
                )

        current = documents
        for step in steps:
            logger.info("파이프라인 단계 실행: %s (%d개)", step, len(current))
            current = STEP_MAP[step](current)

        strategy = " → ".join(steps)
        return self._build_result(current, strategy, len(documents))

    # ── 내부 유틸 ────────────────────────────────
    @staticmethod
    def _enrich_metadata(
        docs: List[Document], strategy: str
    ) -> List[Document]:
        """청크 인덱스·전체 수·전략명을 메타데이터에 자동 추가"""
        total = len(docs)
        for i, doc in enumerate(docs):
            doc.metadata.setdefault("split_strategy", strategy)
            doc.metadata["chunk_index"] = i
            doc.metadata["chunk_total"] = total
        return docs

    @staticmethod
    def _build_result(
        docs: List[Document], strategy: str, original_count: int
    ) -> SplitResult:
        """SplitResult 생성 헬퍼"""
        sizes = [len(d.page_content) for d in docs] if docs else [0]
        warnings: List[str] = []

        # 비정상적으로 큰 청크 경고
        oversized = [s for s in sizes if s > 4000]
        if oversized:
            warnings.append(f"4000자 초과 청크 {len(oversized)}개 존재")

        return SplitResult(
            documents=docs,
            strategy_used=strategy,
            original_count=original_count,
            result_count=len(docs),
            avg_chunk_size=round(sum(sizes) / max(len(sizes), 1), 1),
            min_chunk_size=min(sizes),
            max_chunk_size=max(sizes),
            warnings=warnings,
        )


# ──────────────────────────────────────────────
# 사용 예시
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # 샘플 문서
    sample_documents = [
        Document(
            page_content=(
                "# 1장. 개요\n\n"
                "이 문서는 텍스트 분할 예시입니다.\n\n"
                "## 1.1 배경\n\n"
                "자연어 처리에서 텍스트 분할은 중요한 전처리 단계입니다.\n\n"
                "| 방법       | 속도  | 정확도 |\n"
                "|------------|-------|--------|\n"
                "| Recursive  | 빠름  | 보통   |\n"
                "| Semantic   | 보통  | 높음   |\n\n"
                "```python\n"
                "splitter = RecursiveCharacterTextSplitter(chunk_size=1000)\n"
                "docs = splitter.split_documents(documents)\n"
                "```\n"
            ),
            metadata={"source": "sample.md"},
        )
    ]

    with IntelligentTextSplitter(default_chunk_size=500, default_chunk_overlap=50) as splitter:

        # 1. 자동 전략 선택
        auto_result = splitter.auto_split(sample_documents)
        print("\n▶ 자동 분할")
        print(" ", auto_result.summary())
        if auto_result.warnings:
            print("  경고:", auto_result.warnings)

        # 2. 개별 전략 직접 호출
        semantic_docs  = splitter.semantic_split(sample_documents)
        section_docs   = splitter.section_aware_split(sample_documents)
        table_docs     = splitter.table_aware_split(sample_documents)
        code_docs      = splitter.code_aware_split(sample_documents)

        print("\n▶ 전략별 결과")
        print(f"  semantic      : {len(semantic_docs)}개")
        print(f"  section_aware : {len(section_docs)}개")
        print(f"  table_aware   : {len(table_docs)}개")
        print(f"  code_aware    : {len(code_docs)}개")

        # 3. 파이프라인 (섹션 분할 → 의미 단위 분할)
        pipe_result = splitter.pipeline_split(
            sample_documents,
            steps=["section_aware", "semantic"],
            chunk_size=300,
        )
        print("\n▶ 파이프라인 분할")
        print(" ", pipe_result.summary())

        # 4. 청크 내용 미리보기
        print("\n▶ 자동 분할 청크 미리보기")
        for doc in auto_result.documents:
            meta    = doc.metadata
            preview = doc.page_content[:60].replace("\n", " ")
            print(
                f"  [{meta.get('chunk_index', '?')}/{meta.get('chunk_total', '?')}] "
                f"({meta.get('content_type', meta.get('split_strategy', '-'))}) "
                f"{preview}…"
            )