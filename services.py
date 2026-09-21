import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import Groq
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

_EMBEDDING_MODEL_CACHE: dict[str, Any] = {}
LOCAL_VECTOR_DB = Path(__file__).resolve().parent / "vector_db" / "transcripts_vector_db.json"


def get_embedding_model(model_name: str = MODEL_NAME):
    if model_name not in _EMBEDDING_MODEL_CACHE:
        _EMBEDDING_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _EMBEDDING_MODEL_CACHE[model_name]


def embed_query(text: str, model_name: str = MODEL_NAME):
    model = get_embedding_model(model_name)
    return model.encode(text, normalize_embeddings=True).tolist()


def get_qdrant_client(url: str | None = None, api_key: str | None = None):
    qdrant_url = url or QDRANT_URL
    qdrant_api_key = api_key or QDRANT_API_KEY
    if not qdrant_url or not qdrant_api_key:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in .env")
    return QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=120)


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

    return [
        {
            "id": hit.id,
            "score": hit.score,
            "video_number": hit.payload.get("video_number"),
            "video_title": hit.payload.get("video_title"),
            "start": hit.payload.get("start"),
            "end": hit.payload.get("end"),
            "text": hit.payload.get("text"),
            "text_with_timestamp": hit.payload.get("text_with_timestamp"),
        }
        for hit in results
    ]


def local_fallback_search(question: str, limit: int = 5, video_number: int | None = None):
    if not LOCAL_VECTOR_DB.exists():
        return []

    with LOCAL_VECTOR_DB.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    tokens = [token.lower() for token in re.findall(r"[A-Za-z0-9\u0900-\u097F]+", question)]
    if not tokens:
        return []

    scored = []
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        if video_number is not None and record.get("video_number") != video_number:
            continue

        text_blob = " ".join([
            str(record.get("text") or ""),
            str(record.get("text_with_timestamp") or ""),
        ]).lower()
        score = sum(1 for token in tokens if token in text_blob)
        if score > 0:
            scored.append({
                "id": idx,
                "score": float(score),
                "video_number": record.get("video_number"),
                "video_title": record.get("video_title"),
                "start": record.get("start"),
                "end": record.get("end"),
                "text": record.get("text"),
                "text_with_timestamp": record.get("text_with_timestamp"),
            })

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:limit]


def build_context(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No relevant transcript context found."

    parts = []
    for item in results:
        title = item.get("video_title") or "Unknown video"
        start = item.get("start")
        end = item.get("end")
        text = item.get("text") or ""
        parts.append(f"Video: {title} | Time: {start}s - {end}s\n{text}")

    return "\n\n---\n\n".join(parts)


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


def answer_question(question: str, limit: int = 5, video_number: int | None = None):
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    try:
        results = query_qdrant(question=question, limit=limit, video_number=video_number)
    except Exception:
        results = local_fallback_search(question=question, limit=limit, video_number=video_number)

    context = build_context(results)
    if not results:
        answer = "I could not find relevant transcript context for this question."
    else:
        answer = ask_llm(question, context)

    return {
        "question": question,
        "answer": answer,
        "context": context,
        "sources": [
            {
                "id": item.get("id"),
                "score": item.get("score"),
                "video_number": item.get("video_number"),
                "video_title": item.get("video_title"),
                "start": item.get("start"),
                "end": item.get("end"),
                "text": item.get("text"),
            }
            for item in results
        ],
        "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "used_video_filter": video_number is not None,
    }
