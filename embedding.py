import json
import os
import re
from pathlib import Path
from typing import Any, Iterable
from huggingface_hub import InferenceClient
from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parent / ".env")

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "dsa_lectures")

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


def get_transcript_dir() -> Path:
    transcript_dir = Path(__file__).resolve().parent / "transcript (1)" / "transcripts_json_kaggle"
    if not transcript_dir.exists():
        raise FileNotFoundError(f"Transcript folder not found: {transcript_dir}")
    return transcript_dir


def _clean_segment_text(text: str | None) -> str:
    """Remove blank, time-gap, and noisy filler text that adds useless vectors."""
    if text is None:
        return ""

    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    if not cleaned:
        return ""

    cleaned = re.sub(r"[\u200b\ufeff]", "", cleaned)
    if re.fullmatch(r"[\W_]+", cleaned):
        return ""

    lowered = cleaned.lower()
    if lowered in {"loop", "gap", "word loop", "word gap", "loop gap", "gap loop"}:
        return ""

    words = re.findall(r"[A-Za-z\u0900-\u097F]+", lowered)
    if not words:
        return ""

    if len(words) == 1:
        word = words[0]
        if word in {"ok", "okay", "hmm", "um", "uh", "ah", "yeah", "yes", "no", "right", "well", "so", "just", "like", "basically", "actually", "loop", "gap"}:
            return ""
        if len(word) >= 2 and re.fullmatch(r"(.)\1+", word):
            return ""

    if len(words) > 1:
        counts = {}
        for word in words:
            counts[word] = counts.get(word, 0) + 1
        top_count = max(counts.values())
        if top_count / len(words) >= 0.7:
            return ""

    return cleaned


def _is_repeated_single_word_noise(text: str, repeated_word_counts: dict[str, int]) -> bool:
    words = re.findall(r"[A-Za-z\u0900-\u097F]+", text.lower())
    if not words:
        return True

    if len(words) == 1:
        word = words[0]
        if word in {"ok", "okay", "hmm", "um", "uh", "ah", "yeah", "yes", "no", "right", "well", "so", "just", "like", "basically", "actually", "loop", "gap"}:
            return True
        return repeated_word_counts.get(word, 0) > 3

    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    top_count = max(counts.values())
    if top_count / len(words) >= 0.7:
        return True

    return False


def load_transcript_records(start: int | None = None, end: int | None = None):
    """Load transcript records with video_number, video_title, and cleaned segments."""
    records = []
    for file_path in sorted(get_transcript_dir().glob("*.json")):
        match = re.match(r"^(\d+)", file_path.stem)
        if not match:
            continue

        video_number = int(match.group(1))
        in_range = True
        if start is not None and video_number < start:
            in_range = False
        if end is not None and video_number > end:
            in_range = False

        if not in_range:
            continue

        with file_path.open("r", encoding="utf-8") as f:
            segments = json.load(f)

        repeated_word_counts: dict[str, int] = {}
        for seg in segments or []:
            if not isinstance(seg, dict):
                continue
            text = _clean_segment_text(seg.get("text"))
            if not text:
                continue
            for word in re.findall(r"[A-Za-z\u0900-\u097F]+", text.lower()):
                if len(word) <= 2:
                    repeated_word_counts[word] = repeated_word_counts.get(word, 0) + 1

        cleaned_segments = []
        for seg in segments or []:
            if not isinstance(seg, dict):
                continue

            text = _clean_segment_text(seg.get("text"))
            if not text:
                continue
            if _is_repeated_single_word_noise(text, repeated_word_counts):
                continue

            seg = {**seg, "text": text}
            cleaned_segments.append(seg)

        title = re.sub(r"^\d+\s*-\s*", "", file_path.stem).strip()
        records.append({
            "video_number": video_number,
            "video_title": title,
            "segments": cleaned_segments,
        })

    return records


def chunk_transcript_segments(transcript_data, window_seconds: float = 75.0, overlap_seconds: float = 3.0):
    if overlap_seconds >= window_seconds:
        raise ValueError("overlap_seconds must be smaller than window_seconds")

    segments = []
    for item in transcript_data or []:
        if isinstance(item, list):
            segments.extend(item)
        elif isinstance(item, dict):
            segments.append(item)

    if not segments:
        return []

    sorted_segments = sorted(
        segments,
        key=lambda seg: (float(seg.get("start", 0.0)), float(seg.get("end", seg.get("start", 0.0))))
    )

    start_time = float(sorted_segments[0].get("start", 0.0))
    end_time = max(float(seg.get("end", seg.get("start", 0.0))) for seg in sorted_segments)
    step = window_seconds - overlap_seconds

    chunks = []
    current_start = start_time

    while current_start < end_time:
        current_end = current_start + window_seconds
        matched_segments = []
        text_parts = []

        for seg in sorted_segments:
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", seg_start))
            if seg_end <= current_start:
                continue
            if seg_start >= current_end:
                continue

            matched_segments.append(seg)
            seg_text = str(seg.get("text", "")).strip()
            if seg_text:
                text_parts.append(seg_text)

        if text_parts:
            chunks.append({
                "start": round(current_start, 2),
                "end": round(min(current_end, end_time), 2),
                "text": " ".join(text_parts),
                "segments": matched_segments,
            })

        current_start += step

    return chunks


def build_vector_text_with_timestamps(segments: list[dict[str, Any]]) -> str:
    lines = []
    for seg in segments:
        seg_text = str(seg.get("text", "")).strip()
        if not seg_text:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", seg.get("start", 0.0)))
        lines.append(f"[{start:.1f}s - {end:.1f}s] {seg_text}")
    return "\n".join(lines)


def create_embedding(text: str, model_name: str = MODEL_NAME):
    return embed_query(text, model_name=model_name)


def build_vector_records(transcript_records: Iterable[dict[str, Any]], window_seconds: float = 75.0, overlap_seconds: float = 3.0):
    vector_records = []

    for record in transcript_records:
        video_number = record.get("video_number")
        video_title = record.get("video_title")
        chunks = chunk_transcript_segments(
            record.get("segments", []),
            window_seconds=window_seconds,
            overlap_seconds=overlap_seconds,
        )

        for chunk in chunks:
            text = (chunk.get("text") or "").strip()
            if not text:
                continue

            vector_records.append({
                "video_number": video_number,
                "video_title": video_title,
                "start": chunk.get("start"),
                "end": chunk.get("end"),
                "text": text,
                "text_with_timestamp": build_vector_text_with_timestamps(chunk.get("segments", [])),
                "vector": create_embedding(text, model_name=MODEL_NAME),
            })

    return vector_records


def save_json_vector_db(vector_records, output_path: str | None = None):
    if output_path is None:
        output_path = Path(__file__).resolve().parent / "vector_db" / "transcripts_vector_db.json"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(vector_records, f, ensure_ascii=False, indent=2)

    return str(output_path)


def save_to_qdrant(vector_records, collection_name: str = QDRANT_COLLECTION, qdrant_url: str | None = None, qdrant_api_key: str | None = None, batch_size: int = 100):
    if not vector_records:
        return {"status": "skipped", "message": "No vector records to insert"}

    qdrant_url = qdrant_url or QDRANT_URL
    qdrant_api_key = qdrant_api_key or QDRANT_API_KEY

    if not qdrant_url or not qdrant_api_key:
        return {"status": "skipped", "message": "QDRANT_URL or QDRANT_API_KEY missing in .env"}

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams

        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=120)
        embedding_size = len(vector_records[0]["vector"])
        collections = client.get_collections().collections
        existing = [item.name for item in collections]

        if collection_name not in existing:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
            )

        total_uploaded = 0
        for start_index in range(0, len(vector_records), batch_size):
            batch = vector_records[start_index:start_index + batch_size]
            points = []

            for index, item in enumerate(batch):
                global_index = start_index + index
                payload = {
                    "video_number": item["video_number"],
                    "video_title": item["video_title"],
                    "start": item["start"],
                    "end": item["end"],
                    "text": item["text"],
                    "text_with_timestamp": item["text_with_timestamp"],
                }
                points.append(PointStruct(id=global_index, vector=item["vector"], payload=payload))

            client.upsert(collection_name=collection_name, points=points)
            total_uploaded += len(points)

        return {"status": "success", "collection": collection_name, "uploaded": total_uploaded}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def run_pipeline(start: int | None = None, end: int | None = None, save_json: bool = True, save_qdrant: bool = True):
    records = load_transcript_records(start=start, end=end)
    vector_records = build_vector_records(records, window_seconds=75.0, overlap_seconds=3.0)

    json_path = None
    if save_json:
        json_path = save_json_vector_db(vector_records)

    qdrant_result = None
    if save_qdrant:
        qdrant_result = save_to_qdrant(vector_records)

    return {
        "records_loaded": len(records),
        "vector_count": len(vector_records),
        "json_path": json_path,
        "qdrant": qdrant_result,
        "records": vector_records,
    }


if __name__ == "__main__":
    result = run_pipeline()
    print(f"records_loaded={result['records_loaded']}")
    print(f"vector_count={result['vector_count']}")
    print(f"json_path={result['json_path']}")
    print(f"qdrant={result['qdrant']}")
