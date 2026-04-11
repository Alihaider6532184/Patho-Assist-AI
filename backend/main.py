"""
=============================================================================
 Patho-Assist AI — FastAPI Backend (main.py)
=============================================================================
 Central entrypoint for the backend API.

 Responsibilities:
   1. Application lifecycle (startup / shutdown hooks)
   2. CORS configuration for the Next.js frontend
   3. Route registration for PDF ingestion, vision analysis, and chat
   4. Session-level state management (image descriptions per session)
   5. Ollama model-swap orchestration (16 GB RAM constraint)

 Architecture Notes:
   • Every AI model call goes through a load → infer → unload cycle.
   • Only ONE model may reside in RAM at any given time.
   • ChromaDB runs in-process (no external server).
   • Vision descriptions are stored in the session alongside the
     per-session ChromaDB collection for true cross-modal context.
=============================================================================
"""

import os
import uuid
import shutil
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Internal modules
# ---------------------------------------------------------------------------
from rag_engine import RAGEngine
from vision_engine import VisionEngine
from chat_engine import ChatEngine

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()  # Reads backend/.env

OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
VISION_MODEL      = os.getenv("VISION_MODEL", "paligemma")
TEXT_MODEL         = os.getenv("TEXT_MODEL", "gemma2:2b")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
MAX_PDF_SIZE_MB   = int(os.getenv("MAX_PDF_SIZE_MB", "50"))
MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "20"))
CHUNK_SIZE        = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP     = int(os.getenv("CHUNK_OVERLAP", "200"))

# Upload directory for temporary file storage
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Logging — structured, professional output
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("patho-assist")

# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------
# In-memory store mapping session_id → latest image description + metadata.
# In production you'd use Redis or a DB; for a local tool this is sufficient.
session_store: Dict[str, dict] = {}


def get_or_create_session(session_id: Optional[str] = None) -> str:
    """Return an existing session ID or mint a new one."""
    if session_id and session_id in session_store:
        return session_id
    new_id = session_id or str(uuid.uuid4())
    session_store[new_id] = {
        "image_description": None,   # Populated by /analyze-image
        "image_filename": None,      # Original filename for reference
        "chat_history": [],          # Conversation turns
    }
    logger.info("Session created: %s", new_id)
    return new_id


# ---------------------------------------------------------------------------
# Engine Singletons
# ---------------------------------------------------------------------------
rag_engine: Optional[RAGEngine] = None
vision_engine: Optional[VisionEngine] = None
chat_engine: Optional[ChatEngine] = None


# ---------------------------------------------------------------------------
# Application Lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup / Shutdown hooks.
    - On startup:  initialise all engines.
    - On shutdown: persist ChromaDB and clean up temp files.
    """
    global rag_engine, vision_engine, chat_engine

    logger.info("=" * 60)
    logger.info("  Patho-Assist AI  —  Starting Up")
    logger.info("=" * 60)
    logger.info("Ollama URL     : %s", OLLAMA_BASE_URL)
    logger.info("Vision model   : %s", VISION_MODEL)
    logger.info("Text model     : %s", TEXT_MODEL)
    logger.info("ChromaDB dir   : %s", CHROMA_PERSIST_DIR)
    logger.info("Embedding model: %s", EMBEDDING_MODEL)

    # Initialise the RAG engine (loads embedding model + ChromaDB)
    rag_engine = RAGEngine(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_model_name=EMBEDDING_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    logger.info("RAG engine initialised ✓")

    # Initialise the Vision engine (no model loaded yet — lazy)
    vision_engine = VisionEngine(
        ollama_base_url=OLLAMA_BASE_URL,
        model_name=VISION_MODEL,
    )
    logger.info("Vision engine initialised ✓")

    # Initialise the Chat engine (no model loaded yet — lazy)
    chat_engine = ChatEngine(
        ollama_base_url=OLLAMA_BASE_URL,
        model_name=TEXT_MODEL,
    )
    logger.info("Chat engine initialised ✓")

    logger.info("=" * 60)
    logger.info("  All systems ready — accepting requests")
    logger.info("=" * 60)

    yield  # ← Application runs here

    # --- Shutdown ---
    logger.info("Shutting down — cleaning temp uploads …")
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    logger.info("Patho-Assist AI stopped.")


# ---------------------------------------------------------------------------
# FastAPI App Instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Patho-Assist AI",
    description=(
        "A 100 % offline, multimodal AI assistant for medical researchers. "
        "Analyses histopathology images and queries patient-history PDFs "
        "using local Retrieval-Augmented Generation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — Allow the Next.js dev server and production build
# ---------------------------------------------------------------------------
ALLOWED_ORIGINS = [
    "http://localhost:3000",        # Next.js dev server
    "http://127.0.0.1:3000",
    "http://localhost:3001",        # Alternate port
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],            # GET, POST, PUT, DELETE, OPTIONS
    allow_headers=["*"],            # Content-Type, Authorization, etc.
)


# ===========================================================================
# Pydantic Models (Request / Response Schemas)
# ===========================================================================

class HealthResponse(BaseModel):
    """GET /health response."""
    status: str = "ok"
    version: str = "1.0.0"
    ollama_url: str
    models: dict
    models_available: dict = {}


class IngestResponse(BaseModel):
    """POST /ingest-pdf response."""
    session_id: str
    filename: str
    num_chunks: int
    message: str


class AnalyzeImageResponse(BaseModel):
    """POST /analyze-image response."""
    session_id: str
    filename: str
    description: str
    message: str


class ChatRequest(BaseModel):
    """POST /chat request body."""
    session_id: str
    question: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    """POST /chat response."""
    session_id: str
    answer: str
    sources: list  # ChromaDB document snippets used
    used_image_context: bool


# ===========================================================================
# Routes
# ===========================================================================

# ---- Health Check ---------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Verify the API is running and report configured models.

    Also checks whether each model is actually pulled in Ollama,
    so the frontend can show a clear "model missing" warning.
    """
    # Check model availability (non-blocking, fast checks)
    vision_available = await vision_engine.is_model_available() if vision_engine else False
    chat_available = await chat_engine.is_model_available() if chat_engine else False

    return HealthResponse(
        status="ok",
        version="1.0.0",
        ollama_url=OLLAMA_BASE_URL,
        models={
            "vision": VISION_MODEL,
            "text": TEXT_MODEL,
        },
        models_available={
            "vision": vision_available,
            "text": chat_available,
        },
    )


# ---- PDF Ingestion --------------------------------------------------------
@app.post("/ingest-pdf", response_model=IngestResponse, tags=["RAG"])
async def ingest_pdf(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
):
    """
    Upload a patient-history PDF.

    Pipeline:
      1. Validate file type and size.
      2. Save to disk temporarily.
      3. Extract text → chunk with RecursiveCharacterTextSplitter.
      4. Embed chunks and upsert into ChromaDB.
      5. Return chunk count to the frontend.
    """
    # --- Validation ---
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_PDF_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Limit is {MAX_PDF_SIZE_MB} MB.",
        )

    # --- Persist to disk ---
    sid = get_or_create_session(session_id)
    save_path = UPLOAD_DIR / f"{sid}_{file.filename}"
    save_path.write_bytes(contents)
    logger.info("PDF saved: %s (%.2f MB)", save_path.name, size_mb)

    # --- Ingest via RAG engine ---
    try:
        num_chunks = rag_engine.ingest_pdf(
            pdf_path=str(save_path),
            session_id=sid,
        )
    except Exception as exc:
        logger.exception("PDF ingestion failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        # Remove the temp file — data lives in ChromaDB now
        save_path.unlink(missing_ok=True)

    logger.info("Ingested %d chunks for session %s", num_chunks, sid)

    return IngestResponse(
        session_id=sid,
        filename=file.filename,
        num_chunks=num_chunks,
        message=f"Successfully ingested {num_chunks} chunks from '{file.filename}'.",
    )


# ---- Vision Analysis (FULL IMPLEMENTATION) --------------------------------
@app.post("/analyze-image", response_model=AnalyzeImageResponse, tags=["Vision"])
async def analyze_image(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
):
    """
    Upload a histopathology image for AI analysis via paligemma.

    Memory-Safe Pipeline:
      1. Validate image type and size.
      2. Save image to disk temporarily.
      3. VisionEngine: evict all models → load paligemma → infer → evict.
      4. Store the text description in session state.
      5. Clean up the temp image file.

    After this endpoint returns, paligemma is FULLY EVICTED from RAM.
    The text description remains in the session for /chat to use.
    """
    # --- Validation ---
    allowed_types = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type '{ext}'. Allowed: {allowed_types}",
        )

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large ({size_mb:.1f} MB). Limit is {MAX_IMAGE_SIZE_MB} MB.",
        )

    # --- Save image temporarily ---
    sid = get_or_create_session(session_id)
    save_path = UPLOAD_DIR / f"{sid}_image{ext}"
    save_path.write_bytes(contents)
    logger.info("Image saved: %s (%.2f MB)", save_path.name, size_mb)

    # --- Vision analysis via paligemma ---
    # This call is fully async — it won't block the FastAPI event loop.
    # Internally: evict all → load paligemma → infer → evict paligemma.
    try:
        description = await vision_engine.analyze_image(str(save_path))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Vision analysis failed for session %s", sid)
        raise HTTPException(
            status_code=500,
            detail=(
                f"Vision analysis failed: {str(exc)}. "
                "Ensure Ollama is running and paligemma is pulled "
                "(`ollama pull paligemma`)."
            ),
        )
    finally:
        # Clean up temp image — description is stored in session
        save_path.unlink(missing_ok=True)

    # --- Store vision description in session state ---
    # This is the cross-modal bridge: /chat reads this same field
    # to combine image context with PDF context.
    session_store[sid]["image_description"] = description
    session_store[sid]["image_filename"] = file.filename

    logger.info(
        "✓ Image '%s' analysed for session %s — description stored (%d chars)",
        file.filename, sid, len(description),
    )

    return AnalyzeImageResponse(
        session_id=sid,
        filename=file.filename,
        description=description,
        message="Image analysis complete. Description stored in session for cross-modal chat.",
    )


# ---- Cross-Modal Chat (FULL IMPLEMENTATION) -------------------------------
@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Answer a user question using RAG + optional image context via gemma2:2b.

    Memory-Safe Pipeline:
      1. Retrieve relevant PDF chunks from session's ChromaDB collection.
      2. Fetch the paligemma image description from session state (if any).
      3. ChatEngine: evict all models → load gemma2:2b → generate → evict.
      4. Record the Q&A turn in session history.

    After this endpoint returns, gemma2:2b is FULLY EVICTED from RAM.
    
    Context Fusion:
      The prompt sent to gemma2:2b includes:
        - System instructions (medical AI persona)
        - Retrieved PDF excerpts with page numbers
        - paligemma's image description (if image was analysed)
        - Recent chat history (last 3 turns)
        - The current question
    """
    sid = request.session_id
    if sid not in session_store:
        raise HTTPException(status_code=404, detail=f"Session '{sid}' not found.")

    # --- Step 1: Retrieve relevant PDF context from ChromaDB ---
    try:
        retrieval_results = rag_engine.query(
            question=request.question,
            session_id=sid,
            top_k=5,
        )
    except Exception as exc:
        logger.exception("RAG retrieval failed")
        raise HTTPException(status_code=500, detail=str(exc))

    # --- Step 2: Fetch image description from session state ---
    image_desc = session_store[sid].get("image_description")
    used_image_context = image_desc is not None
    chat_history = session_store[sid].get("chat_history", [])

    logger.info(
        "Chat context: %d RAG chunks, image=%s, history=%d turns",
        len(retrieval_results),
        "yes" if used_image_context else "no",
        len(chat_history),
    )

    # --- Step 3: Generate answer via gemma2:2b ---
    # Internally: evict all → load gemma2:2b → generate → evict.
    try:
        answer = await chat_engine.generate_answer(
            question=request.question,
            rag_context=retrieval_results,
            image_description=image_desc,
            chat_history=chat_history,
        )
    except Exception as exc:
        logger.exception("Chat generation failed for session %s", sid)
        raise HTTPException(
            status_code=500,
            detail=(
                f"Chat generation failed: {str(exc)}. "
                "Ensure Ollama is running and the text model is pulled "
                f"(`ollama pull {TEXT_MODEL}`)."
            ),
        )

    # --- Step 4: Record in chat history ---
    session_store[sid]["chat_history"].append({
        "question": request.question,
        "answer": answer,
    })

    logger.info("✓ Chat response generated for session %s (%d chars)", sid, len(answer))

    return ChatResponse(
        session_id=sid,
        answer=answer,
        sources=retrieval_results,
        used_image_context=used_image_context,
    )


# ---- Session Management ---------------------------------------------------
@app.post("/session", tags=["Session"])
async def create_session():
    """Create a new session and return its ID."""
    sid = get_or_create_session()
    return {"session_id": sid}


@app.get("/session/{session_id}", tags=["Session"])
async def get_session(session_id: str):
    """Retrieve session metadata (image status, chat history length)."""
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found.")
    data = session_store[session_id]
    return {
        "session_id": session_id,
        "has_image": data["image_description"] is not None,
        "image_filename": data["image_filename"],
        "chat_turns": len(data["chat_history"]),
    }


@app.delete("/session/{session_id}", tags=["Session"])
async def delete_session(session_id: str):
    """
    Delete a session and its associated ChromaDB data.
    
    Useful for clearing patient data after analysis is complete.
    """
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found.")

    # Remove ChromaDB collection for this session
    rag_engine.delete_session_data(session_id)

    # Remove session state
    del session_store[session_id]

    logger.info("Session %s deleted (session state + ChromaDB)", session_id)
    return {"message": f"Session '{session_id}' and all associated data deleted."}


# ===========================================================================
# Entrypoint
# ===========================================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,        # Hot-reload during development
        log_level="info",
    )
