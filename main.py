from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from services import ask_llm, retrieve_top_results, validate_dsa_query

app = FastAPI(title="DSA Lecture RAG", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "dsa-rag-agent"}


@app.get("/")
def root():
    return {"message": "DSA Lecture RAG API is running", "docs": "/docs"}


@app.post("/ask")
async def ask_question(request: Request):
    try:
        raw_body = await request.body()
        if not raw_body:
            raise ValueError("Question cannot be empty.")

        try:
            payload = await request.json()
        except Exception:
            payload = None

        if isinstance(payload, dict):
            question_value = payload.get("question")
        elif isinstance(payload, str):
            question_value = payload
        else:
            question_value = None

        if question_value is None:
            try:
                question_value = raw_body.decode("utf-8").strip()
            except Exception:
                question_value = None

        cleaned = (question_value or "").strip()
        if not cleaned:
            raise ValueError("Question cannot be empty.")

        is_dsa = validate_dsa_query(cleaned)
        if not is_dsa:
            return {
                "question": cleaned,
                "answer": "This system only supports DSA and algorithm-related questions. Please ask about data structures, algorithms, complexity, patterns, or coding interview problems.",
                "relevant_vectors": [],
                "allowed": False,
                "model": "qwen/qwen3.8-27b",
            }

        results = retrieve_top_results(cleaned, limit=5)
        if not results:
            return {
                "question": cleaned,
                "answer": "I could not find relevant transcript context for this question.",
                "relevant_vectors": [],
                "allowed": True,
                "model": "qwen/qwen3.8-27b",
            }

        context = "\n\n---\n\n".join(
            f"Video: {item.get('video_title') or 'Unknown video'} | Time: {item.get('start')}s - {item.get('end')}s\n{item.get('text') or ''}"
            for item in results
        )
        answer = ask_llm(cleaned, context)

        return {
            "question": cleaned,
            "answer": answer,
            "relevant_vectors": [
                {
                    "id": item.get("id"),
                    "score": item.get("score"),
                    "cosine_similarity": item.get("cosine_similarity"),
                    "video_number": item.get("video_number"),
                    "video_title": item.get("video_title"),
                    "start": item.get("start"),
                    "end": item.get("end"),
                    "text": item.get("text"),
                }
                for item in results
            ],
            "allowed": True,
            "model": "qwen/qwen3.8-27b",
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
