import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

load_dotenv(Path(__file__).resolve().parent / ".env")

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "dsa_transcripts")

_EMBEDDING_MODEL_CACHE: dict[str, Any] = {}


def get_embedding_model(model_name: str = MODEL_NAME):
    if model_name not in _EMBEDDING_MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        _EMBEDDING_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _EMBEDDING_MODEL_CACHE[model_name]


def embed_query(query_text: str, model_name: str = MODEL_NAME):
    model = get_embedding_model(model_name)
    return model.encode(query_text, normalize_embeddings=True).tolist()


def get_qdrant_client(url: str | None = None, api_key: str | None = None):
    qdrant_url = url or QDRANT_URL
    qdrant_api_key = api_key or QDRANT_API_KEY
    if not qdrant_url or not qdrant_api_key:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in the .env file.")
    return QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=120)


def query_qdrant(question: str, limit: int = 5, collection_name: str = QDRANT_COLLECTION, score_threshold: float | None = None, video_number: int | None = None):
    client = get_qdrant_client()
    query_vector = embed_query(question)

    filter_conditions = []
    if video_number is not None:
        filter_conditions.append(FieldCondition(key="video_number", match=MatchValue(value=video_number)))

    search_filter = Filter(must=filter_conditions) if filter_conditions else None

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
        query_filter=search_filter,
        score_threshold=score_threshold,
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


def build_context_block(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No relevant transcript context found."

    context_parts = []
    for item in results:
        title = item.get("video_title") or "Unknown video"
        start = item.get("start")
        end = item.get("end")
        text = item.get("text") or ""
        context_parts.append(f"Video: {title} | Time: {start}s - {end}s\n{text}\n")

    return "\n---\n".join(context_parts)


def build_rag_prompt(question: str, results: list[dict[str, Any]]) -> str:
    context = build_context_block(results)
    return f"""You are a helpful assistant answering from the transcript context below.

Question:
{question}

Context:
{context}

Answer using only the context above. If the answer is not present, say that clearly.
"""


def run_query(question: str, limit: int = 5, video_number: int | None = None):
    results = query_qdrant(question, limit=limit, video_number=video_number)
    return {
        "question": question,
        "results": results,
        "context": build_context_block(results),
        "prompt": build_rag_prompt(question, results),
    }


if __name__ == "__main__":
    sample_question = "wo question chahiye jisme slow and fast pointer padhaya gaya hai"
    response = run_query(sample_question, limit=3)
    print(response["prompt"])
