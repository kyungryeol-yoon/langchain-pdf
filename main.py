from langchain_community.document_loaders import (
    PyPDFLoader, UnstructuredPDFLoader, PDFPlumberLoader,
    PyMuPDFLoader, PDFMinerLoader
)
from langchain_community.vectorstores import FAISS
import os
from pdf_ingestion_pipeline import PDFIngestionPipeline
from typing import List, Dict, Any


def main():
    print("Hello from app!")



if __name__ == "__main__":
    pipeline = PDFIngestionPipeline(chunk_size=1000, chunk_overlap=200)

    # 단일 파일
    result = pipeline.run("report.pdf")
    vectorstore.add_documents(result.chunks)

    # 여러 파일 한 번에
    chunks = pipeline.collect_chunks(["a.pdf", "b.pdf", "c.pdf"])
    vectorstore.add_documents(chunks)