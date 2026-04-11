"""
=============================================================================
 Patho-Assist AI — RAG Engine (rag_engine.py)
=============================================================================
 Handles the complete Retrieval-Augmented Generation pipeline:

   1. PDF text extraction        (pypdf)
   2. Text chunking              (LangChain RecursiveCharacterTextSplitter)
   3. Embedding generation       (sentence-transformers, runs locally)
   4. Vector storage & retrieval (ChromaDB, on-disk persistence)

 Design Decisions:
   • Embeddings are produced by `all-MiniLM-L6-v2` — a lightweight model
     (~80 MB) that runs comfortably alongside Ollama on 16 GB RAM.
   • ChromaDB runs in embedded mode (no Docker, no server).
   • Each session gets its own ChromaDB collection so that different
     patients' data never bleeds across sessions.
   • Metadata (page number, chunk index) is stored alongside each vector
     for transparent source attribution.

 Public API:
   RAGEngine.ingest_pdf(pdf_path, session_id) → int   (num chunks stored)
   RAGEngine.query(question, session_id, top_k)  → list[dict]  (ranked results)
=============================================================================
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import chromadb
from chromadb.config import Settings as ChromaSettings

from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("patho-assist.rag")


# ===========================================================================
# RAG Engine
# ===========================================================================
class RAGEngine:
    """
    Encapsulates PDF ingestion + semantic retrieval.

    Usage:
        engine = RAGEngine(persist_directory="./chroma_db")
        num_chunks = engine.ingest_pdf("patient_report.pdf", session_id="abc123")
        results = engine.query("What is the tumour grade?", session_id="abc123")
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialise the RAG engine.

        Args:
            persist_directory:    Where ChromaDB stores its on-disk data.
            embedding_model_name: HuggingFace model ID for sentence embeddings.
            chunk_size:           Max characters per text chunk.
            chunk_overlap:        Overlap between consecutive chunks (for context continuity).
        """
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # --- Text Splitter (LangChain) ------------------------------------
        # RecursiveCharacterTextSplitter tries to split on paragraph → sentence
        # → word boundaries before resorting to hard character cuts.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],  # Priority order
            is_separator_regex=False,
        )

        # --- Embedding Model (local, ~80 MB in RAM) ----------------------
        logger.info("Loading embedding model: %s …", embedding_model_name)
        self.embedding_model = SentenceTransformer(embedding_model_name)
        logger.info("Embedding model loaded ✓ (dim=%d)", self.embedding_model.get_sentence_embedding_dimension())

        # --- ChromaDB Client (persistent, in-process) ---------------------
        os.makedirs(persist_directory, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,  # No data leaves the machine
            ),
        )
        logger.info("ChromaDB initialised at: %s ✓", persist_directory)

    # -----------------------------------------------------------------------
    # Collection Helpers
    # -----------------------------------------------------------------------
    def _get_collection_name(self, session_id: str) -> str:
        """
        Derive a ChromaDB collection name from the session ID.

        ChromaDB collection names must:
          - Be 3–63 characters
          - Start and end with an alphanumeric character
          - Contain only alphanumerics, underscores, or hyphens
        """
        # Prefix ensures min length; truncate to respect max length
        name = f"session_{session_id.replace('-', '_')}"
        return name[:63]

    def _get_or_create_collection(self, session_id: str) -> chromadb.Collection:
        """Get (or create) the ChromaDB collection for a given session."""
        name = self._get_collection_name(session_id)
        collection = self.chroma_client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},  # Cosine similarity
        )
        logger.info(
            "Collection '%s' ready (existing docs: %d)",
            name,
            collection.count(),
        )
        return collection

    # -----------------------------------------------------------------------
    # PDF Text Extraction
    # -----------------------------------------------------------------------
    @staticmethod
    def _extract_text_from_pdf(pdf_path: str) -> List[Dict]:
        """
        Extract text from every page of a PDF.

        Returns:
            A list of dicts: [{"page": 1, "text": "..."}, ...]
        """
        reader = PdfReader(pdf_path)
        pages = []

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                pages.append({
                    "page": page_num,
                    "text": text.strip(),
                })
                logger.debug("Page %d: %d chars extracted", page_num, len(text))
            else:
                logger.warning("Page %d: no extractable text (scanned image?)", page_num)

        if not pages:
            raise ValueError(
                f"No text could be extracted from '{pdf_path}'. "
                "The PDF may contain only scanned images. "
                "Please use an OCR-processed PDF."
            )

        logger.info(
            "Extracted text from %d / %d pages in '%s'",
            len(pages),
            len(reader.pages),
            Path(pdf_path).name,
        )
        return pages

    # -----------------------------------------------------------------------
    # Chunking
    # -----------------------------------------------------------------------
    def _chunk_pages(self, pages: List[Dict]) -> List[Dict]:
        """
        Split page-level text into smaller, overlapping chunks.

        Each chunk preserves its source page number for attribution.

        Returns:
            [{"text": "...", "page": 1, "chunk_index": 0}, ...]
        """
        all_chunks = []
        global_index = 0

        for page_data in pages:
            page_num = page_data["page"]
            raw_text = page_data["text"]

            # LangChain's splitter returns a list of strings
            chunks = self.text_splitter.split_text(raw_text)

            for chunk_text in chunks:
                all_chunks.append({
                    "text": chunk_text,
                    "page": page_num,
                    "chunk_index": global_index,
                })
                global_index += 1

        logger.info(
            "Chunking complete: %d chunks (size=%d, overlap=%d)",
            len(all_chunks),
            self.chunk_size,
            self.chunk_overlap,
        )
        return all_chunks

    # -----------------------------------------------------------------------
    # Embedding
    # -----------------------------------------------------------------------
    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate dense vector embeddings for a batch of text strings.

        Uses sentence-transformers (runs 100% locally on CPU).
        Returns a list of float vectors.
        """
        # .encode() returns numpy arrays; ChromaDB expects plain lists
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True,  # Unit vectors for cosine similarity
        )
        return embeddings.tolist()

    # -----------------------------------------------------------------------
    # Public: Ingest PDF
    # -----------------------------------------------------------------------
    def ingest_pdf(self, pdf_path: str, session_id: str) -> int:
        """
        Full ingestion pipeline:  PDF → text → chunks → embeddings → ChromaDB.

        Args:
            pdf_path:   Absolute or relative path to the PDF file.
            session_id: Unique session identifier (determines ChromaDB collection).

        Returns:
            Number of chunks stored in ChromaDB.
        """
        logger.info("▶ Ingesting PDF: %s (session: %s)", pdf_path, session_id)

        # 1. Extract text page-by-page
        pages = self._extract_text_from_pdf(pdf_path)

        # 2. Chunk the pages
        chunks = self._chunk_pages(pages)

        if not chunks:
            logger.warning("No chunks produced — PDF may be empty.")
            return 0

        # 3. Generate embeddings for all chunks
        texts = [c["text"] for c in chunks]
        logger.info("Generating embeddings for %d chunks …", len(texts))
        embeddings = self._embed_texts(texts)

        # 4. Prepare ChromaDB upsert payload
        ids = [f"{session_id}_chunk_{c['chunk_index']}" for c in chunks]
        metadatas = [
            {
                "session_id": session_id,
                "page": c["page"],
                "chunk_index": c["chunk_index"],
                "char_count": len(c["text"]),
            }
            for c in chunks
        ]

        # 5. Upsert into ChromaDB (idempotent — safe to re-ingest same PDF)
        collection = self._get_or_create_collection(session_id)
        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(
            "✓ Ingestion complete: %d chunks stored in collection '%s'",
            len(chunks),
            self._get_collection_name(session_id),
        )
        return len(chunks)

    # -----------------------------------------------------------------------
    # Public: Semantic Query
    # -----------------------------------------------------------------------
    def query(
        self,
        question: str,
        session_id: str,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Retrieve the most relevant document chunks for a given question.

        Args:
            question:   The user's natural-language question.
            session_id: Determines which ChromaDB collection to search.
            top_k:      Number of top results to return.

        Returns:
            A list of dicts, each containing:
              - text:       The chunk content
              - page:       Source page number
              - distance:   Cosine distance (lower = more relevant)
              - chunk_index: Position in the original document
        """
        collection = self._get_or_create_collection(session_id)

        # If the collection is empty, return gracefully
        if collection.count() == 0:
            logger.info("Collection is empty — no PDF ingested for session %s", session_id)
            return []

        # Embed the question using the same model
        question_embedding = self._embed_texts([question])[0]

        # Query ChromaDB
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        # Flatten ChromaDB's nested response structure
        formatted = []
        if results and results["documents"]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                formatted.append({
                    "text": doc,
                    "page": meta.get("page"),
                    "chunk_index": meta.get("chunk_index"),
                    "distance": round(dist, 4),
                })

        logger.info(
            "Query returned %d results for session %s (top distance: %.4f)",
            len(formatted),
            session_id,
            formatted[0]["distance"] if formatted else -1,
        )
        return formatted

    # -----------------------------------------------------------------------
    # Utility: Delete Session Data
    # -----------------------------------------------------------------------
    def delete_session_data(self, session_id: str) -> bool:
        """
        Remove all vectors for a given session from ChromaDB.

        Returns True if the collection existed and was deleted.
        """
        name = self._get_collection_name(session_id)
        try:
            self.chroma_client.delete_collection(name)
            logger.info("Deleted collection: %s", name)
            return True
        except ValueError:
            logger.warning("Collection '%s' not found — nothing to delete.", name)
            return False

    # -----------------------------------------------------------------------
    # Utility: List All Sessions
    # -----------------------------------------------------------------------
    def list_sessions(self) -> List[str]:
        """Return all session collection names currently in ChromaDB."""
        collections = self.chroma_client.list_collections()
        return [c.name for c in collections]
