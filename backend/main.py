"""
main.py
-------
FastAPI application entry point.

Responsibilities:
  - Create the FastAPI app instance with metadata
  - Configure CORS (allows Streamlit frontend to call the API)
  - Initialise the MySQL schema on startup via lifespan event
  - Register all API routes from routes.py

Run with:
    uvicorn backend.main:app --reload --port 8000
or (from the project root):
    uvicorn main:app --reload --port 8000  (if CWD is backend/)
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import init_db
from routes import router


# ─── Lifespan: runs init_db once at startup ───────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise DB schema before the first request is served."""
    print("[Startup] Initialising database schema …")
    init_db()
    print("[Startup] Database ready.")
    yield
    # (optional teardown goes here)


# ─── App factory ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Model Evaluation Platform",
    description=(
        "Upload datasets, run LLM evaluations, compute BLEU/ROUGE/accuracy "
        "metrics, and compare model performance."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow all origins so the Streamlit dashboard (any port) can reach the API.
# In production, restrict `allow_origins` to your actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all route handlers
app.include_router(router)


# ─── Dev entry-point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
