from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    SpacyTextSplitter,
    NLTKTextSplitter,
    TokenTextSplitter
)
from langchain_core.documents import Document
import re

class IntelligentTextSplitter:
    def __init__(self):
        self.splitters = {
            "recursive": RecursiveCharacterTextSplitter,
            "spacy": SpacyTextSplitter,
            "nltk": NLTKTextSplitter,
            "token": TokenTextSplitter
        }

    def semantic_split(self, documents: List[Document], chunk_size: int = 1000) -> List[Document]:
        # 의미 단위 기반 텍스트 분할

        # 문서 유형별 구분자 설정
        separators = [
            "\n\n", # 단락 구분
            "\n", # 줄 구분
            ". ", # 문장 구분
            "? ", # 질문 구분
            "! ", # 감탄 구분
            "; ", # 세미콜론 구분
            ", ", # 쉼표 구분
            " ", # 공백 구분
            "" # 문자 구분
        ]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_operlap=200,
            separators=separators,
            length_function=len
        )

        return splitter.split_documents(documents)
    
    def section_aware_split(self, documents: List[Document]) -> List[Document]:
        # 섹션 인식 분할
        split_docs = []

        for doc in documents:
            content = doc.page_content

            # 섹션 헤더 패턴 감지
            section_patterns = [
                r'^#{1,6}\s+(.+)$', # 마크다운 헤더
                r'^\d+\.\s+(.+)$', # 번호 목록
                r'^[A-Z\s]+:', # 대문자 라벨
                r'^\s*\d+\.\d+', # 계층 번호
            ]

            sections = []
            current_section = ""

            for line in content.split('\n'):
                is_header = any(re.match(pattern, line, re.MULTILINE) for pattern in section_patterns)
                if is_header and current_section:
                    sections.append(current_section.strip())
                    current_section = line + '\n'
                else:
                    current_section += line + '\n'

            if current_section:
                sections.append(current_section.strip())

            # 각 섹션을 별도 문서로 생성
            for i, section in enumerate(sections):
                if len(section) > 100:
                    section_doc = Document(
                        page_content=section,
                        metadata={
                            **doc.metadata,
                            "section_index": i,
                            "section_type": "content"
                        }
                    )
                    split_docs.append(section_doc)

        return split_docs
    
    def table_aware_split(self, documents: List[Document]) -> List[Document]:
        # 표 인식 분할
        split_docs = []

        for doc in documents:
            content = doc.page_content

            # 표 패턴 감지
            table_pattern = r'(\|[^|\n]*\|[^|\n]*\|.*?\n)+'
            tables = re.findall(table_pattern, content, re.MULTILINE)

            if tables:
                # 표와 일반 텍스트 분리
                text_parts = re.split(table_pattern, content)

                for i, part in enumerate(text_parts):
                    if part.strip():
                        doc_type = "table" if part in tables else "text"

                        split_doc = Document(
                            page_content=part.strip(),
                            metadata={
                                **doc.metadata,
                                "content_type": doc_type,
                                "part_index": i
                            }
                        )
                        split_docs.append(split_doc)

                    else:
                        split_docs.append(split_doc)
            else:
                split_docs.append(doc)

        return split_docs
    
# 사용 예시
intelligent_splitter = IntelligentTextSplitter()

# 의미 단위 분할
semantic_docs = intelligent_splitter.semantic_split(documents)

# 섹션 인식 분할
section_docs = intelligent_splitter.section_aware_split(documents)

# 표 인식 분할
table_docs = intelligent_splitter.table_aware_split(documents)

print(f"원본 문서 수: {len(documents)}")
print(f"의미 단위 분할 후: {len(semantic_docs)}")
print(f"섹션 인식 분할 후: {len(section_docs)}")
print(f"표 인식 분할 후: {len(table_docs)}")