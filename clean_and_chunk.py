from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List, Optional

CHUNK_SECONDS = 75.0
OVERLAP_SECONDS = 15.0


def _normalize_segment(segment: dict[str, Any]) -> dict[str, Any]:
    """Normalize a transcript segment to a consistent shape."""
    start = float(segment.get("start", 0.0))
    end = float(segment.get("end", start))
    text = str(segment.get("text", "")).strip()
    return {
        "start": start,
        "end": end,
        "text": text,
        "words": segment.get("words", []),
    }


def chunk_by_duration(
    segments: Iterable[dict[str, Any]],
    chunk_seconds: float = CHUNK_SECONDS,
    overlap_seconds: float = OVERLAP_SECONDS,
) -> List[dict[str, Any]]:
    """Split transcript segments into fixed-size time windows with overlap.

    Example: 75s chunk size and 15s overlap yields windows [0,75], [60,135], [120,195],...
    """
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be greater than zero")
    if overlap_seconds < 0:
        raise ValueError("overlap_seconds cannot be negative")
    if overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be smaller than chunk_seconds")

    normalized = [_normalize_segment(segment) for segment in segments]
    if not normalized:
        return []

    step = chunk_seconds - overlap_seconds
    chunks: List[dict[str, Any]] = []

    # Walk the timeline in fixed-size chunks. The overlap is intentional.
    current_start = 0.0
    max_end = max(segment["end"] for segment in normalized)

    while current_start < max_end:
        current_end = current_start + chunk_seconds

        selected = [
            seg for seg in normalized
            if seg["end"] > current_start and seg["start"] < current_end
        ]

        if not selected:
            current_start += step
            continue

        chunk_text = " ".join(seg["text"] for seg in selected if seg.get("text")).strip()
        chunk = {
            "start": round(current_start, 2),
            "end": round(current_end, 2),
            "text": chunk_text,
            "segments": [
                {
                    "start": round(seg["start"], 2),
                    "end": round(seg["end"], 2),
                    "text": seg["text"],
                }
                for seg in selected
            ],
        }
        chunks.append(chunk)
        current_start += step

    return chunks


def load_transcript(path: str | Path) -> list[dict[str, Any]]:
    """Load transcript JSON/Dict from disk."""
    raw = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        segments = data.get("segments") or data.get("transcript") or []
        return list(segments)
    raise TypeError(f"Unsupported transcript format in {path}")


def chunk_transcript_file(
    transcript_path: str | Path,
    *,
    chunk_seconds: float = CHUNK_SECONDS,
    overlap_seconds: float = OVERLAP_SECONDS,
    video_id: Optional[str] = None,
    video_title: Optional[str] = None,
) -> List[dict[str, Any]]:
    """Read a transcript file and return time-windowed chunk metadata for embedding."""
    segments = load_transcript(transcript_path)
    chunks = chunk_by_duration(segments, chunk_seconds=chunk_seconds, overlap_seconds=overlap_seconds)

    for chunk in chunks:
        chunk["video_id"] = video_id
        chunk["video_title"] = video_title
        chunk["time_start"] = chunk["start"]
        chunk["time_end"] = chunk["end"]

    return chunks


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Chunk transcript JSON files into overlapping time windows.")
    parser.add_argument("transcript", nargs="?", help="Path to a transcript JSON file")
    parser.add_argument("--chunk-seconds", type=float, default=CHUNK_SECONDS)
    parser.add_argument("--overlap-seconds", type=float, default=OVERLAP_SECONDS)
    parser.add_argument("--output", help="Optional path to write chunk JSON output")
    args = parser.parse_args()

    if not args.transcript:
        parser.error("Please provide a transcript JSON file path.")

    chunks = chunk_transcript_file(
        args.transcript,
        chunk_seconds=args.chunk_seconds,
        overlap_seconds=args.overlap_seconds,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved {len(chunks)} chunks to {output_path}")
    else:
        print(json.dumps(chunks[:3], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
