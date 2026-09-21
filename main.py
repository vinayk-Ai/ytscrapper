from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from services import answer_question


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    limit: int = Field(default=5, ge=1, le=10)
    video_number: int | None = None


class SourceItem(BaseModel):
    id: int | str | None = None
    score: float | None = None
    video_number: int | None = None
    video_title: str | None = None
    start: float | None = None
    end: float | None = None
    text: str | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    context: str
    sources: list[SourceItem] = []
    model: str
    used_video_filter: bool = False


app = FastAPI(title="DSA Lecture RAG", version="1.0.0")


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "dsa-rag-agent"}


@app.get("/")
def root():
    return {"message": "DSA Lecture RAG API is running", "docs": "/docs"}


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest):
    try:
        result = answer_question(
            question=payload.question,
            limit=payload.limit,
            video_number=payload.video_number,
        )
        return AskResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
