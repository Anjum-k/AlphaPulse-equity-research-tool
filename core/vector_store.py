import os
import shutil
import logging
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings

from core.config import FAISS_INDEX_DIR, CHROMA_DB_DIR, DEFAULT_TOP_K

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages vector databases (FAISS and ChromaDB) for the Equity Research Tool.
    Supports index creation, persistence to disk, loading, resetting, and similarity search.
    """

    def __init__(
        self,
        store_type: str = "FAISS",
        embedding_model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ):
        self.store_type = store_type.upper()
        self.embedding_model = embedding_model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._embeddings = None
        self._vector_store: Optional[VectorStore] = None

    @property
    def embeddings(self) -> OpenAIEmbeddings:
        """Lazy initialization of OpenAI embeddings."""
        if self._embeddings is None:
            if not self.api_key:
                raise ValueError("OpenAI API Key is required to initialize embeddings.")
            self._embeddings = OpenAIEmbeddings(
                model=self.embedding_model,
                api_key=self.api_key,
            )
        return self._embeddings

    def has_existing_index(self) -> bool:
        """Check if an existing vector index exists on disk."""
        if self.store_type == "FAISS":
            index_file = FAISS_INDEX_DIR / "index.faiss"
            pkl_file = FAISS_INDEX_DIR / "index.pkl"
            return index_file.exists() and pkl_file.exists()
        elif self.store_type == "CHROMADB":
            # Chroma creates sqlite3 or parquet files inside directory
            return CHROMA_DB_DIR.exists() and any(CHROMA_DB_DIR.iterdir())
        return False

    def build_from_documents(self, documents: List[Document]) -> VectorStore:
        """
        Build a vector store from Document chunks and persist it to disk.
        """
        if not documents:
            raise ValueError("No documents provided to build vector index.")

        if self.store_type == "FAISS":
            from langchain_community.vectorstores import FAISS

            logger.info("Building FAISS index with %d chunks...", len(documents))
            self._vector_store = FAISS.from_documents(documents, self.embeddings)
            FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
            self._vector_store.save_local(str(FAISS_INDEX_DIR))
            logger.info("FAISS index saved to %s", FAISS_INDEX_DIR)

        elif self.store_type == "CHROMADB":
            from langchain_community.vectorstores import Chroma

            logger.info("Building ChromaDB collection with %d chunks...", len(documents))
            CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
            self._vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(CHROMA_DB_DIR),
            )
            logger.info("ChromaDB saved to %s", CHROMA_DB_DIR)

        else:
            raise ValueError(f"Unsupported vector store type: {self.store_type}")

        return self._vector_store

    def load_existing_index(self) -> Optional[VectorStore]:
        """
        Load an existing vector index from disk if present.
        """
        if not self.has_existing_index():
            return None

        try:
            if self.store_type == "FAISS":
                from langchain_community.vectorstores import FAISS

                logger.info("Loading existing FAISS index from %s", FAISS_INDEX_DIR)
                self._vector_store = FAISS.load_local(
                    str(FAISS_INDEX_DIR),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                return self._vector_store

            elif self.store_type == "CHROMADB":
                from langchain_community.vectorstores import Chroma

                logger.info("Loading existing ChromaDB from %s", CHROMA_DB_DIR)
                self._vector_store = Chroma(
                    persist_directory=str(CHROMA_DB_DIR),
                    embedding_function=self.embeddings,
                )
                return self._vector_store
        except Exception as e:
            logger.error("Error loading existing index: %s", str(e))
            return None

        return None

    def get_or_load_store(self) -> Optional[VectorStore]:
        """Return the current vector store instance, or load from disk if available."""
        if self._vector_store is not None:
            return self._vector_store
        return self.load_existing_index()

    def similarity_search_with_score(
        self, query: str, k: int = DEFAULT_TOP_K
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with query and return (document, score) tuples.
        """
        store = self.get_or_load_store()
        if store is None:
            raise ValueError("Vector store is not initialized or indexed yet.")

        return store.similarity_search_with_score(query, k=k)

    def as_retriever(self, k: int = DEFAULT_TOP_K):
        """Get standard LangChain retriever instance."""
        store = self.get_or_load_store()
        if store is None:
            raise ValueError("Vector store is not initialized or indexed yet.")
        return store.as_retriever(search_kwargs={"k": k})

    def reset_index(self) -> None:
        """Clear local vector store files and reset in-memory state."""
        self._vector_store = None
        if self.store_type == "FAISS" and FAISS_INDEX_DIR.exists():
            shutil.rmtree(FAISS_INDEX_DIR, ignore_errors=True)
            logger.info("FAISS index deleted.")
        elif self.store_type == "CHROMADB" and CHROMA_DB_DIR.exists():
            shutil.rmtree(CHROMA_DB_DIR, ignore_errors=True)
            logger.info("ChromaDB directory deleted.")
