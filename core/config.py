import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

FAISS_INDEX_DIR = DATA_DIR / "faiss_index"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"

# Available Models
AVAILABLE_LLMS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-3.5-turbo",
]

AVAILABLE_EMBEDDINGS = [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
]

# Supported Vector Stores
VECTOR_STORES = ["FAISS", "ChromaDB"]

# Default Hyperparameters
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 4

# Target news scraper user-agent
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)
