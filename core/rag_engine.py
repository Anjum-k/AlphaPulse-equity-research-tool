import logging
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from core.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

EQUITY_RESEARCH_SYSTEM_PROMPT = """You are an elite Equity Research Analyst and Financial Intelligence Assistant.
Your mission is to provide rigorous, accurate, and insightful analysis based STRICTLY on the provided news article excerpts.

CRITICAL INSTRUCTIONS:
1. Ground your answer completely in the provided context excerpts. Do NOT hallucinate financial figures, ticker symbols, or dates.
2. If the context does not contain enough information to fully answer the query, clearly state what is missing.
3. Structure your response professionally using the following format:
   - **Executive Summary**: A concise 2-3 sentence overview answering the core question.
   - **Key Financial & Market Data**: Bullet points highlighting specific numbers, revenue, EPS, margins, guidance, or corporate metrics mentioned.
   - **Strategic Insights & Catalyst Analysis**: Implications for the company, competitors, industry trends, or potential headwinds/risks.
   - **Source Attribution**: Mention which source(s) contributed to each key finding.
4. Keep the tone objective, analytical, and tailored for portfolio managers, buy-side, and sell-side analysts.

Context:
{context}
"""

EQUITY_RESEARCH_USER_PROMPT = """Question: {question}

Provide your detailed equity research assessment:"""


class EquityRAGEngine:
    """
    RAG Engine for Equity Research question-answering with citation and metadata tracking.
    """

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        api_key: Optional[str] = None,
    ):
        self.vector_store_manager = vector_store_manager
        self.llm_model = llm_model
        self.temperature = temperature
        self.api_key = api_key or vector_store_manager.api_key

    def _format_context(self, docs: List[Document]) -> str:
        """Format retrieved documents with source headers for the prompt."""
        formatted_chunks = []
        for i, doc in enumerate(docs, 1):
            title = doc.metadata.get("title", "News Article")
            source = doc.metadata.get("source", "Unknown Source")
            formatted_chunks.append(
                f"[Source {i}: {title} | URL: {source}]\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(formatted_chunks)

    def answer_question(
        self, question: str, top_k: int = 4
    ) -> Dict[str, Any]:
        """
        Retrieve relevant chunks and generate equity research answer.
        Returns a dictionary with:
        - 'answer': str
        - 'sources': List[Dict] with title, url, domain, excerpt, score
        - 'raw_docs': List[Document]
        """
        if not question or not question.strip():
            raise ValueError("Research question cannot be empty.")

        # Perform similarity search with score
        docs_and_scores = self.vector_store_manager.similarity_search_with_score(
            query=question, k=top_k
        )

        if not docs_and_scores:
            return {
                "answer": "No relevant information found in the indexed articles for this query.",
                "sources": [],
                "raw_docs": [],
            }

        docs = [item[0] for item in docs_and_scores]
        scores = [item[1] for item in docs_and_scores]

        context_text = self._format_context(docs)

        # Set up LangChain ChatOpenAI and Prompt
        llm = ChatOpenAI(
            model=self.llm_model,
            temperature=self.temperature,
            api_key=self.api_key,
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", EQUITY_RESEARCH_SYSTEM_PROMPT),
            ("user", EQUITY_RESEARCH_USER_PROMPT),
        ])

        chain = prompt | llm | StrOutputParser()

        try:
            answer_text = chain.invoke({
                "context": context_text,
                "question": question,
            })
        except Exception as e:
            logger.error("Error generating answer from LLM: %s", str(e))
            raise RuntimeError(f"Failed to generate answer from OpenAI LLM: {str(e)}")

        # Prepare source attribution records
        sources_info = []
        for doc, score in zip(docs, scores):
            meta = doc.metadata
            sources_info.append({
                "title": meta.get("title", "Financial Article"),
                "url": meta.get("source", "#"),
                "domain": meta.get("domain", "web"),
                "published_time": meta.get("published_time", None),
                "excerpt": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                "relevance_score": round(float(score), 4) if score is not None else None,
            })

        return {
            "answer": answer_text,
            "sources": sources_info,
            "raw_docs": docs,
        }
