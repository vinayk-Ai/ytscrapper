import json
import os
import re
from pathlib import Path
import requests
from typing import Any
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sentence_transformers import SentenceTransformer

load_dotenv(Path(__file__).resolve().parent / ".env")

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "dsa_lectures")
GROQ_FALLBACK_MODELS = [
    os.getenv("GROQ_MODEL"),
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]
_client = None
class QueryInput(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)


_client = None
def get_client():
    global _client
    if _client is None:
        hf_token = os.getenv("HF_API_KEY") or os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_API_KEY is missing in .env")
        _client = InferenceClient(token=hf_token)
    return _client

def embed_query(text: str, model_name: str = MODEL_NAME):
    client = get_client()
    embedding = client.feature_extraction(text, model=model_name)
    
    # normalize
    import numpy as np
    vec = np.array(embedding)
    norm = vec / np.linalg.norm(vec)
    return norm.tolist()


def get_qdrant_client(url: str | None = None, api_key: str | None = None):
    qdrant_url = url or QDRANT_URL
    qdrant_api_key = api_key or QDRANT_API_KEY
    if not qdrant_url or not qdrant_api_key:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in .env")
    return QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=120)


def validate_dsa_query(question: str) -> bool:
    cleaned = (question or "").strip()
    if not cleaned:
        return False

    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY is missing in .env")

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    prompt = (
        "You are a strict DSA relevance classifier. "
        "Return exactly one word: YES or NO.\n\n"
        "YES only if the message is clearly about DSA, data structures, algorithms, coding interview problems, "
        "time complexity, pointers, trees, graphs, arrays, recursion, DP, hashing, patterns, or other technical programming topics.\n"
        "NO for general life, geography, food, travel, business, personal conversation, or anything unrelated to programming or DSA.\n\n"
        f"Query: {cleaned}"
    )

    candidates = []
    for candidate in GROQ_FALLBACK_MODELS:
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    last_error = None
    for candidate in candidates:
        try:
            completion = client.chat.completions.create(
                model=candidate,
                messages=[
                    {"role": "system", "content": "Return only YES or NO."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=10,
            )
            result = (completion.choices[0].message.content or "").strip().upper()
            return result == "YES"
        except Exception as exc:  # pragma: no cover - guards against stale model names
            last_error = exc

    if last_error is not None:
        raise last_error
    return False


def query_qdrant(question: str, limit: int = 5, video_number: int | None = None, collection_name: str = QDRANT_COLLECTION):
    client = get_qdrant_client()
    query_vector = embed_query(question)

    filters = []
    if video_number is not None:
        filters.append(FieldCondition(key="video_number", match=MatchValue(value=video_number)))

    search_filter = Filter(must=filters) if filters else None
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
        query_filter=search_filter,
        with_payload=True,
        with_vectors=False,
    ).points

    records = []
    for hit in results:
        score = float(hit.score) if hit.score is not None else None
        records.append({
            "id": hit.id,
            "score": score,
            "cosine_similarity": score,
            "video_number": hit.payload.get("video_number"),
            "video_title": hit.payload.get("video_title"),
            "start": hit.payload.get("start"),
            "end": hit.payload.get("end"),
            "text": hit.payload.get("text"),
            "text_with_timestamp": hit.payload.get("text_with_timestamp"),
        })
    return records


def retrieve_top_results(question: str, limit: int = 5, video_number: int | None = None) -> list[dict[str, Any]]:
    cleaned_question = str(question or "").strip()
    if not cleaned_question:
        raise ValueError("Question cannot be empty.")

    return query_qdrant(question=cleaned_question, limit=limit, video_number=video_number)

def ask_llm(question: str, context: str, model_name: str | None = None) -> str:
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY is missing in .env")

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    candidates = []
    if model_name:
        candidates.append(model_name)
    for candidate in GROQ_FALLBACK_MODELS:
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    last_error = None
    for candidate in candidates:
        try:
            completion = client.chat.completions.create(
                model=candidate,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert DSA tutor. Use only the transcript context to answer accurately.",
                    },
                    {
                        "role": "user",
                        "content": f"Question: {question}\n\nContext:\n{context}\n\nAnswer using only this context.",
                    },
                ],
                temperature=0.2,
                max_tokens=800,
            )
            return completion.choices[0].message.content.strip()
        except Exception as exc:  # pragma: no cover - guards against stale model names
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError("No Groq model candidates were available for generation.")


# def answer_question(question: str, limit: int = 5, video_number: int | None = None):
#     cleaned_question = str(question or "").strip()
#     if not cleaned_question:
#         raise ValueError("Question cannot be empty.")

#     results = retrieve_top_results(cleaned_question, limit=limit, video_number=video_number)

#     if not results:
#         answer = "I could not find relevant transcript context for this question."
#         context = "No relevant transcript context found."
#     else:
#         context = "\n\n---\n\n".join(
#             f"Video: {item.get('video_title') or 'Unknown video'} | Time: {item.get('start')}s - {item.get('end')}s\n{item.get('text') or ''}"
#             for item in results
#         )
#         answer = ask_llm(cleaned_question, context)

#     return {
#         "question": cleaned_question,
#         "answer": answer,
#         "relevant_vectors": [
#             {
#                 "id": item.get("id"),
#                 "score": item.get("score"),
#                 "cosine_similarity": item.get("cosine_similarity"),
#                 "video_number": item.get("video_number"),
#                 "video_title": item.get("video_title"),
#                 "start": item.get("start"),
#                 "end": item.get("end"),
#                 "text": item.get("text"),
#             }
#             for item in results
#         ],
#         "model": os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
#         "used_video_filter": video_number is not None,
#     }
