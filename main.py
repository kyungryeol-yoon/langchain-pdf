from langchain_community.document_loaders import (
    PyPDFLoader, UnstructuredPDFLoader, PDFPlumberLoader,
    PyMuPDFLoader, PDFMinerLoader
)
from langchain_community.vectorstores import FAISS
import os
from pdf_ingestion_pipeline import PDFIngestionPipeline
import langchain
from langchain_classic.cache import SQLiteCache
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor


def langchain_cache():
    langchain.llm_cache = SQLiteCache(database_path="./cache.db")

def llm_compressor():
    compressor = LLMChainExtractor.from_llm(ChatOpenAI())
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )

def process_documents(docs):
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda doc: retriever.retrieve(doc), docs))

    return results

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