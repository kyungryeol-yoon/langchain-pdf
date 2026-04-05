from langchain_community.vectorstores import FAISS, Chroma, Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
from typing import List, Dict, Optional

class AdvancedVectorStore:
    def __init__(self, embedding_type: str = "openai"):
        self.embedding_type = embedding_type
        self.embeddings = self._initialize_embeddings()
        self.vectorstores = {}

    def _initialize_embeddings(self):
        # 임베딩 모델 초기화
        if self.embedding_type == "openai":
            return OpenAIEmbeddings(model="text-embedding-3-large")
        elif self.embedding_type == "huggingface":
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all_MiniLM-L6-v2"
            )
        else:
            raise ValueError(f"지원하지 않는 임베딩 타입: {self.embedding_type}")
        
    def create_hierarchical_vectorstore(self, documents: List[Document], domain: str):
        # 계층적 벡터 저장소 생성

        # 문서 유형별 분류
        doc_types = {}
        for doc in documents:
            doc_type = doc.metadata.get("content_type", "general")
            if doc_type not in doc_types:
                doc_types[doc_type] = []
            doc_types[doc_type].append(doc)

        # 각 유형별로 별도 벡터 저장소 생성
        for doc_type, type_docs in doc_types.items():
            if type_docs:
                vectorstore = FAISS.from_documents(type_docs, self.embeddings)
                self.vectorstores[f"{domain}_{doc_type}"] = vectorstore

        # 통합 벡터 저장소도 생성
        all_vectorstore = FAISS.from_documents(documents, self.embeddings)
        self.vectorstores[f"{domain}_all"] = all_vectorstore

        return self.vectorstores
    
    def create_multi_embedding_store(self, documents: List[Document], domain: str):
        # 다중 임베딩 벡터 저장소

        # 여러 임베딩 모델 사용
        embedding_models = {
            "openai": OpenAIEmbeddings(model="text-embedding-3-large"),
            "openai_small": OpenAIEmbeddings(model="text-embedding-3-small"),
            "huggingface": HuggingFaceEmbeddings(model_name="sentence-transformers/all_MiniLM-L6-v2")
        }

        multi_stores = {}
        for model_name, embedding_model in embedding_models.items():
            try:
                vectorstore = FAISS.from_documents(documents, embedding_model)
                multi_stores[f"{domain}_{model_name}"] = vectorstore
            except Exception as e:
                print(f"{model_name} 임베딩 실패: {e}")

        return multi_stores
    
    def create_filteredretriever(self, vectorstore, filters: Dict[str, Any]):
        # 필터링된 검색기 생성

        def filter_function(metadata):
            for key, value in filters.items():
                if key in metadata:
                    if isinstance(value, list):
                        if metadata[key] not in value:
                            return False
                    else:
                        if metadata[key] != value:
                            return False
            return True
        
        # 커스텀 검색기 클래스
        class FilteredRetriever:
            def __init__(self, vectorstore, filter_func):
                self.vectorstore = vectorstore
                self.filter_func = filter_func

            def get_relevant_documents(self, query: str, k: int = 5):

                # 더 많은 문서를 검색한 후 필터링
                candidates = self.vectorstore.similarity_search(query, k=k*3)
                filtered = [doc for doc in candidates if self.filter_func(doc.metadata)]

                return filtered[:k]
            
        return FilteredRetriever(vectorstore, filter_function)
    
    def hybrid_search(self, query: str, vectorstore_keys: List[str], weights: List[float] = None):
        # 하이브리드 검색 (여러 벡터 저장소 결합)

        if weights is None:
            weights = [1.0] * len(vectorstore_keys)

        all_results = []

        for i, key in enumerate(vectorstore_keys):
            if key in self.vectorstores:
                results = self.vectorstores[key].similarity_search_with_scroe(query, k=5)

                # 가중치 적용
                weighted_results = [
                    (doc, score * weights[i], key) for doc, score in results
                ]
                all_results.extend(weighted_results)

        # 점수순 정렬
        all_results.sort(key=lambda x: x[1], reverse=True)

        # 중복 제거 및 상위 결과 반환
        seen_content = set()
        unique_results = []

        for doc, score, source in all_results:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append((doc, score, source))

                if len(unique_results) >= 5:
                    break

        return unique_results
    
# 예시
advanced_store = AdvancedVectorStore(embedding_type="openai")

# 계층적 벡터 저장소 생성
hierarchical_stores = advanced_store.create_hierarchical_vectorstore(docs, "legal_docs")

# 다중 임베딩 저장소 생성
multi_stores = advanced_store.create_multi_embedding_store(docs, "legal_docs")

# 필터링된 검색
filters = {"content_type": "text", "section_index": [0, 1, 2]}
filtered_retriever = advanced_store.create_filteredretriever(hierarchical_stores["legal_docs_all"], filters)

# 하이브리드 검색
hybrid_results = advanced_store.hybrid_search(
    "계약 조건",
    ["legal_docs_text", "legal_docs_all"],
    weights=[0.7, 0.3]
)

print(f"하이브리드 검색 결과: {len(hybrid_results)}개")