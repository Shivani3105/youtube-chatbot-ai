import os
from typing import Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Import our modular AI & RAG backend module
import backend

load_dotenv()

if not os.environ.get("GEMINI_API_KEY"):
    env_key = os.getenv("GEMINI_API_KEY")
    if env_key:
        os.environ["GEMINI_API_KEY"] = env_key

app = FastAPI(title="YouTube Chatbot API")

# Dynamically get current directory path (works on Windows & Linux Cloud servers like Render)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Global in-memory cache for current active video state
current_video_state: Dict[str, Any] = {
    "video_id": None,
    "main_chain": None,
    "chunk_count": 0,
    "language": "unknown"
}


class LoadVideoRequest(BaseModel):
    video_url: str

class QueryRequest(BaseModel):
    question: str


@app.post("/api/load-video")
async def load_video(req: LoadVideoRequest):
    """Delegate video transcript loading & RAG chain creation directly to backend.py."""
    try:
        # Call backend.py to build RAG pipeline
        rag_data = backend.build_rag_chain(req.video_url)

        # Update server state
        current_video_state["video_id"] = rag_data["video_id"]
        current_video_state["main_chain"] = rag_data["main_chain"]
        current_video_state["chunk_count"] = rag_data["chunk_count"]
        current_video_state["language"] = rag_data["language"]

        return {
            "status": "success",
            "video_id": rag_data["video_id"],
            "chunk_count": rag_data["chunk_count"],
            "language": rag_data["language"],
            "transcript_snippet": rag_data["transcript_snippet"]
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backend processing error: {str(e)}")


@app.post("/api/query")
async def query_video(req: QueryRequest):
    """Delegate user question answering directly to backend.py."""
    if not current_video_state.get("main_chain"):
        raise HTTPException(status_code=400, detail="No video loaded yet. Please load a YouTube video first.")
    
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        # Call backend.py to run query
        answer = backend.query_rag_chain(current_video_state["main_chain"], question)
        return {
            "status": "success",
            "question": question,
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")


@app.get("/api/status")
async def get_status():
    return {
        "status": "online",
        "video_loaded": current_video_state["video_id"] is not None,
        "video_id": current_video_state["video_id"],
        "chunk_count": current_video_state["chunk_count"],
        "language": current_video_state["language"]
    }

# Mount static files directory using dynamic BASE_DIR
app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")

@app.get("/")
async def root():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
