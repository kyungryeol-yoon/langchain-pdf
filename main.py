from langchain_community.document_loaders import (
    PyPDFLoader, UnstructuredPDFLoader, PDFPlumberLoader,
    PyMuPDFLoader, PDFMinerLoader
)
import os
from typing import List, Dict, Any


def main():
    print("Hello from app!")


def first_check():
    loader = PyPDFLoader('hyundai_direct_2022.pdf')

    documents = loader.load()
    print(documents[4].page_content[:300])

class AdvancedPDFProcessor:
    def __init__(self):
        self.loaders = {
            "pypdf": PyPDFLoader,
            "unstructured": UnstructuredPDFLoader,
            "plumber": PDFPlumberLoader,
            "pymupdf": PyMuPDFLoader,
            "pdfminer": PDFMinerLoader,
        }

    def analyze_pdf_structure(self, pdf_path: str) -> Dict[str, Any]:
        # PDF 구조 분석 및 최적 로더 추천
        import PyPDF2

        with open(pdf_path, 'rb') as file:
            # 1. PdfReader 객체 생성
            # 'file'이라는 PDF 파일 객체(binary mode로 열린)를 받아 
            # PyPDF2를 통해 읽을 수 있는 Reader 객체로 변환합니다.
            pdf_reader = PyPDF2.PdfReader(file)

            # 2. 분석 결과 저장용 딕셔너리 초기화
            analysis = {
                # pdf_reader.pages는 페이지 객체들의 리스트입니다. 
                # len()을 사용하여 전체 페이지 수를 구합니다.
                "page_count": len(pdf_reader.pages),

                # 기본값 설정 (현재 이 코드만으로는 이미지/표 존재 여부 자동 판별 불가)
                # 실제로 이미지나 표 존재 여부를 확인하려면 pdf_reader.pages[i].images를 탐색하거나, 표 레이아웃을 분석하는 별도 라이브러리(PDFPlumber 등)를 사용해야 합니다.
                "has_images": False,
                "has_tables": False,

                # 텍스트 밀도 (현재 0으로 초기화)
                "text_density": 0,
                "recommended_loader": "pypdf"
            }

            # 첫 페이지 분석
            first_page = pdf_reader.pages[0]
            text = first_page.extract_text()

            # text가 None일 경우 대비
            if text is None:
                text = ""

            # 텍스트 밀도
            analysis["text_density"] = len(text) / 1000

            # 표 존재 여부 추정 (간단한 휴리스틱)
            if "table" in text.lower() or text.count("|") > 10:
                analysis["has_tables"] = True
                analysis["recommended_loader"] = "plumber"

            # 복잡한 레이아웃 감지
            if analysis["text_density"] < 0.1:
                analysis["recommended_loader"] = "unstructured"
            elif analysis["text_density"] > 2.0:
                analysis["recommended_loader"] = "pymupdf"
 
        return analysis
    
    def load_pdf_with_optimal_loader(self, pdf_path: str):
        """
        ❌ 오류 수정: 원본 코드에서 누락된 메서드
        사용 예시에서 호출되지만 정의가 없었음 → 추가
        """
        analysis = self.analyze_pdf_structure(pdf_path)
        loader_key = analysis["recommended_loader"]
        loader_class = self.loaders[loader_key]
 
        loader = loader_class(pdf_path)
        documents = loader.load()
 
        return documents, loader_key
    
    def extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        import PyPDF2

        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            metadata = pdf_reader.metadata
 
            # ❌ 오류 수정: metadata가 None일 수 있으므로 방어 처리
            if metadata is None:
                metadata = {}
 
            return {
                "title": metadata.get("/Title", ""),
                "author": metadata.get("/Author", ""),
                "subject": metadata.get("/Subject", ""),
                "creator": metadata.get("/Creator", ""),
                "creation_date": metadata.get("/CreationDate", ""),
                "modification_date": metadata.get("/ModDate", ""),
                "page_count": len(pdf_reader.pages)
            }

if __name__ == "__main__":

    first_check()
    processor = AdvancedPDFProcessor()

    pdf_path = "example.pdf"
    documents, used_loader = processor.load_pdf_with_optimal_loader(pdf_path)
    metadata = processor.extract_metadata(pdf_path)

    print(f"사용된 로더: {used_loader}")
    print(f"메타데이터: {metadata}")
