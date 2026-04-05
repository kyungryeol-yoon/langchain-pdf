from langchain_core.runnables import RunnableBranch, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class AdvancedRAGPipeline:
    def __init__(self, vectorsotre, llm_model='gpt-4'):
        self.vectorstore = vectorsotre
        self.llm = ChatOpenAI(model=llm_model, temperature=0.1)
        self.retriever = vectorsotre.as_retriever()

    def create_contextual_retriever(self):
        # 컨텍스트 인식 검색기
        contextualize_q_system_prompt = """이전 대화 기록과 최신 사용자 질문이 주어졌을 때,
        이전 대화 맥락을 참조하는 질문을 독립적으로 이해할 수 있는 질문으로 재구성하세요.
        질문에 답하지 말고, 필요하다면 재구성만 하고, 그렇지 않으면 그대로 반환하세요."""

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )

        return history_aware_retriever
    
    def create_multi_query_retriever(self):
        # 다중 쿼리 검색기
        multi_query_prompt = ChatPromptTemplate.from_template("""
        다음 질문에 대해 다른 관점에서 3개의 유사한 질문을 생성하세요.
        각 질문은 한 줄에 하나씩 작성하세요.

        원본 질문: {question}
                                                              
        대안 질문들:
        """)

        def genreate_queries(question):
            queries_text = (multi_query_prompt | self.llm | StrOutputParser()).invoke(
                "question": question
            )
            queries = [q.strip() for q in queries_text.split('\n') if q.strip()]

            return [question] + queries # 원본 질문도 포함
        
        def multi_query_search(question):
            queries = genreate_queries(question)
            all_docs = []

            for query in queries:
                docs = self.retriever.get_relevant_documents(query)
                all_docs.extend(docs)

            # 중복 제거
            unique_docs = []
            seen_content = set()

            for doc in all_docs:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_docs.append(doc)

            return unique_docs[:5] # 상위 5개 반환
        
        return multi_query_search
    
    def create_adaptive_rag_chain(self):
        # 적응형 RAG 제안

        # 질문 유형 분류
        def classify_question(question):
            classification_prompt = ChatPromptTemplate.from_template("""
                다음 질문을 분류하세요:
                1. factual: 사실적 정보 요청
                2. analytical: 분석적 사고 필요
                3. comparative: 비교 분석 필요
                4. summariztion: 요약 요청

                질문: {question}

                분류 결과 (한 단어로만 답하세요):
            """)

            result = (classification_prompt | self.llm | StrOutputParser()).invoke(
                {"question": question}
            )

            return result.strip().lower()
        
        # 질문 유형별 프롬프트
        prompts = {
            "factual": ChatPromptTemplate.from_template("""
            다음 문서에서 정확한 사실 정보를 찾아 답변하세요:
                                                    
            문서: {context}

            질문: {question}

            답변 (사실만 간단명료하게):
            """),

            "analytical": ChatPromptTemplate.from_template("""
            다음 문서를 분석하여 깊이 있는 답변을 제공하세요:
                                                           
            문서: {context}

            질문: {question}
                                                           
            분석적 답변:
            """),

            "comparative": ChatPromptTemplate.from_template("""
            다음 문서들을 비교 분석하여 답변하세요:
                                                            
            문서: {context}

            질문: {question}
                                                           
            비교 분석 답변:
            """),

            "summarization": ChatPromptTemplate.from_template("""
            다음 문서의 내용을 요약하여 답변하세요:
                                                            
            문서: {context}

            질문: {question}
                                                           
            요약 답변:
            """),
        }

        # 분기 체인 생성
        def route_question(inputs):
            question = inputs["question"]
            question_type = classify_question(question)

            # 기본값 설정
            if question_type not in prompts:
                question_type = "factual"

            prompt = prompts[question_type]
            context = self.retriever.get_relevant_documents(question)

            return {
                "context": "\n\n".join([doc.page_content for doc in context]),
                "question": question,
                "question_type": question_type
            }
        
        adaptive_chain = (
            RunnablePassthrough()
            | route_question
            | RunnableBranch(
                (lambda x: x["question_type"] == "factual", prompts["factual"] | self.llm),
                (lambda x: x["question_type"] == "analytical", prompts["analytical"] | self.llm),
                (lambda x: x["question_type"] == "comparative", prompts["comparative"] | self.llm),
                (lambda x: x["question_type"] == "summarization", prompts["summarization"] | self.llm),
                prompts["factual"] | self.llm # 기본값
            )
            | StrOutputParser() 
        )

        return adaptive_chain
    
# 예시
rag_pipeline = AdvancedRAGPipeline(vectorsotre)

# 켄텍스트 인식 검색기
contextual_retriever = rag_pipeline.create_contextual_retriever()

# 다중 쿼리 검색기
multi_query_retriever = rag_pipeline.create_multi_query_retriever()

# 적응형 RAG 체인
adaptive_chain = rag_pipeline.create_adaptive_rag_chain()

# 테스트
question = "이 계약서의 주요 조건들을 비교 분석해주세요."
response = adaptive_chain.invoke({"question": question})
print(f"답변: {response}")