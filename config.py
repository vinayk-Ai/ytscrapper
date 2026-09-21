import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


class Settings:
    APP_NAME = os.getenv("APP_NAME", "DSA Lecture RAG")
    APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "dsa_lectures")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


settings = Settings()
