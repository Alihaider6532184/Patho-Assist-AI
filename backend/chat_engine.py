"""
=============================================================================
 Patho-Assist AI — Chat Engine (chat_engine.py)
=============================================================================
 Handles cross-modal chat using Ollama's text/reasoning model (gemma2:2b).

 Memory-Safe Execution Flow (16 GB RAM constraint):
   1. Ensure NO other Ollama model is loaded (defensive eviction)
   2. Build a rich prompt combining:
      a) Retrieved PDF chunks from ChromaDB (RAG context)
      b) Vision description from paligemma (if available in session)
      c) The user's natural-language question
      d) Recent chat history for conversational continuity
   3. Call ollama.generate() with the combined prompt
   4. Collect the response (non-streaming for reliability)
   5. Immediately evict gemma2:2b (keep_alive=0) → 0 bytes GPU/RAM

 Uses the official `ollama` Python SDK (AsyncClient) instead of raw
 httpx calls — matching vision_engine.py for full consistency.

 Public API:
   ChatEngine.generate_answer(question, rag_context, image_description, chat_history) → str
============================================================================="""

import asyncio
import logging
from typing import List, Dict, Optional

from ollama import AsyncClient

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("patho-assist.chat")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Timeout for text generation — gemma2:2b is smaller and faster than paligemma,
# but we still allow generous time for complex multi-context queries.
INFERENCE_TIMEOUT = 180.0  # seconds

# ---------------------------------------------------------------------------
# System Prompt — defines the AI's persona and constraints
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are Patho-Assist AI, a highly knowledgeable medical research assistant \
specialising in histopathology. You help medical researchers by:

1. Answering questions about patient history documents (PDFs)
2. Interpreting histopathology image analysis results
3. Correlating findings between documents and images
4. Providing evidence-based clinical insights

IMPORTANT RULES:
- Always base your answers on the provided context (PDF excerpts and/or image analysis).
- If the context does not contain enough information to answer, say so clearly.
- Use proper medical terminology but explain complex terms when helpful.
- Never fabricate findings — only report what is supported by the provided evidence.
- When referencing PDF content, mention the source page number if available.
- When referencing image analysis, explicitly note that the finding comes from the microscopy analysis.
- Structure your responses clearly with relevant headings when appropriate.
"""


class ChatEngine:
    """
    Manages the text/reasoning model lifecycle via the official Ollama SDK.

    The 16 GB RAM constraint means only ONE model can be loaded at a time.
    This engine ensures gemma2:2b is:
      loaded → used → IMMEDIATELY evicted (keep_alive=0)

    Usage:
        engine = ChatEngine(ollama_base_url="http://localhost:11434")
        answer = await engine.generate_answer(
            question="What is the tumour grade?",
            rag_context=[{"text": "...", "page": 3}],
            image_description="Tissue shows ...",
            chat_history=[{"question": "...", "answer": "..."}],
        )
        # At this point, gemma2:2b is already evicted from RAM.
    """

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        model_name: str = "gemma2:2b",
    ):
        """
        Args:
            ollama_base_url: URL of the local Ollama server.
            model_name:      Model tag as shown by `ollama list`.
        """
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.model_name = model_name
        self.client = AsyncClient(host=self.ollama_base_url)
        logger.info("ChatEngine initialised (model: %s, url: %s)", model_name, self.ollama_base_url)

    # -----------------------------------------------------------------------
    # Public: Generate Answer
    # -----------------------------------------------------------------------
    async def generate_answer(
        self,
        question: str,
        rag_context: List[Dict],
        image_description: Optional[str] = None,
        chat_history: Optional[List[Dict]] = None,
    ) -> str:
        """
        Full chat pipeline: build prompt → infer → evict.

        Args:
            question:          The user's natural-language question.
            rag_context:       Retrieved PDF chunks from ChromaDB.
                               Each dict has: text, page, distance, chunk_index.
            image_description: paligemma's analysis of the histopathology image
                               (None if no image was uploaded in this session).
            chat_history:      Previous Q&A turns in this session (for continuity).

        Returns:
            The model's answer as a string.

        Raises:
            httpx.HTTPStatusError: If Ollama returns a non-2xx status.
            RuntimeError: If the model fails to generate a response.
        """
        logger.info("▶ Chat generation starting — question: '%s'", question[:80])

        # Step 1: Defensively evict any lingering model from RAM
        await self._evict_all_models()

        # Step 2: Build the combined prompt
        prompt = self._build_prompt(
            question=question,
            rag_context=rag_context,
            image_description=image_description,
            chat_history=chat_history or [],
        )
        logger.info("Prompt built (%d chars, %d RAG chunks, image=%s)",
                     len(prompt), len(rag_context), image_description is not None)

        # Step 3: Run inference with keep_alive=0 (auto-evict after response)
        answer = await self._run_inference(prompt)

        # Step 4: Belt-and-suspenders — explicitly confirm eviction
        await self._evict_model(self.model_name)

        logger.info("✓ Chat generation complete (%d chars). Model evicted.", len(answer))
        return answer

    # -----------------------------------------------------------------------
    # Private: Prompt Construction
    # -----------------------------------------------------------------------
    def _build_prompt(
        self,
        question: str,
        rag_context: List[Dict],
        image_description: Optional[str],
        chat_history: List[Dict],
    ) -> str:
        """
        Assemble a structured prompt that combines all available context.

        The prompt follows a clear hierarchy:
          1. System instructions (persona + rules)
          2. PDF context (RAG retrieval results)
          3. Image analysis context (if available)
          4. Recent chat history (for conversational continuity)
          5. The user's current question

        This structure ensures the model has maximum relevant context
        while respecting token limits for the smaller gemma2:2b model.
        """
        sections: list[str] = []

        # --- System Instructions ---
        sections.append(SYSTEM_PROMPT)

        # --- PDF / Document Context ---
        if rag_context:
            sections.append("=" * 50)
            sections.append("PATIENT DOCUMENT CONTEXT (Retrieved from PDF)")
            sections.append("=" * 50)

            for i, chunk in enumerate(rag_context, 1):
                page = chunk.get("page", "?")
                distance = chunk.get("distance", None)
                relevance = f" (relevance: {1 - distance:.2f})" if distance is not None else ""
                sections.append(f"\n--- Excerpt {i} (Page {page}{relevance}) ---")
                sections.append(chunk.get("text", ""))

            sections.append("\n[End of document context]\n")
        else:
            sections.append(
                "\n[No PDF documents have been uploaded for this session. "
                "Answers will be based only on image analysis if available.]\n"
            )

        # --- Image Analysis Context ---
        if image_description:
            sections.append("=" * 50)
            sections.append("HISTOPATHOLOGY IMAGE ANALYSIS (from microscopy AI)")
            sections.append("=" * 50)
            sections.append(image_description)
            sections.append("\n[End of image analysis]\n")
        else:
            sections.append(
                "\n[No histopathology image has been analysed for this session.]\n"
            )

        # --- Chat History (last 3 turns for context window efficiency) ---
        if chat_history:
            recent = chat_history[-3:]  # Only last 3 turns to save tokens
            sections.append("-" * 50)
            sections.append("RECENT CONVERSATION HISTORY")
            sections.append("-" * 50)

            for turn in recent:
                sections.append(f"\nResearcher: {turn.get('question', '')}")
                sections.append(f"Patho-Assist AI: {turn.get('answer', '')}")

            sections.append("")

        # --- Current Question ---
        sections.append("=" * 50)
        sections.append("CURRENT QUESTION FROM RESEARCHER")
        sections.append("=" * 50)
        sections.append(f"\n{question}\n")
        sections.append(
            "Please provide a thorough, evidence-based answer. "
            "Reference the document excerpts and/or image analysis above when applicable."
        )

        return "\n".join(sections)

    # -----------------------------------------------------------------------
    # Private: Ollama Inference via Official SDK
    # -----------------------------------------------------------------------
    async def _run_inference(self, prompt: str) -> str:
        """
        Send the combined prompt to Ollama's text model via the official SDK.

        Uses `client.generate()` with non-streaming mode.
        The system prompt is prepended to the user prompt since generate()
        does not natively support a system role — this is the cleanest
        approach for text-only models like gemma2:2b.

        The `keep_alive=0` parameter evicts the model from RAM the
        instant generation completes.
        """
        # Combine system prompt with user prompt for generate() endpoint
        full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"

        logger.info(
            "Sending prompt to Ollama (%s) via SDK generate() — generating answer …",
            self.model_name,
        )

        try:
            response = await self.client.generate(
                model=self.model_name,
                prompt=full_prompt,
                stream=False,
                # ⚠️ CRITICAL: Evict the model from RAM as soon as generation ends.
                # This is the cornerstone of our 16 GB memory-swap strategy.
                keep_alive=0,
                options={
                    "temperature": 0.5,        # Balanced creativity vs. factual accuracy
                    "num_predict": 2048,       # Generous token budget for detailed answers
                    "top_p": 0.9,              # Nucleus sampling for quality
                    "repeat_penalty": 1.1,     # Discourage repetitive output
                },
            )
        except Exception as exc:
            logger.exception("Ollama generate() call failed for model '%s'", self.model_name)
            raise RuntimeError(
                f"Chat inference failed: {exc}. "
                f"Ensure Ollama is running and '{self.model_name}' is pulled "
                f"(`ollama pull {self.model_name}`)."
            ) from exc

        # The SDK returns a GenerateResponse object with attributes
        full_response = (response.response or "").strip()

        # Log generation stats if available
        total_duration = getattr(response, "total_duration", 0) or 0
        eval_count = getattr(response, "eval_count", 0) or 0
        if total_duration:
            logger.info(
                "Generation complete: %d tokens, %.1f s",
                eval_count,
                total_duration / 1e9,
            )

        if not full_response:
            raise RuntimeError(
                f"Ollama returned an empty response for model '{self.model_name}'. "
                "Ensure the model is pulled (`ollama pull gemma2:2b`) and Ollama is running."
            )

        return full_response

    # -----------------------------------------------------------------------
    # Private: Model Memory Management (using SDK)
    # -----------------------------------------------------------------------
    async def _evict_model(self, model_name: str) -> None:
        """
        Force-evict a specific model from Ollama's memory.

        Sends a generate request with keep_alive=0 and an empty prompt
        to trigger immediate unloading without producing output.
        Uses the official SDK for reliability.
        """
        try:
            await self.client.generate(
                model=model_name,
                prompt="",
                keep_alive=0,       # ← Evict NOW
                stream=False,
            )
            logger.info("Model '%s' evicted from RAM ✓", model_name)
        except Exception as exc:
            # Non-fatal — model may not have been loaded
            logger.debug("Eviction request for '%s' failed: %s (non-fatal)", model_name, exc)

    async def _evict_all_models(self) -> None:
        """
        Defensively evict ALL running models before loading a new one.

        Queries Ollama's running model list via the SDK, then evicts
        each one. Guarantees a clean RAM slate.
        """
        try:
            ps_response = await self.client.ps()
            running_models = ps_response.models or []

            if not running_models:
                logger.info("No models currently loaded — RAM is clear")
                return

            for model_info in running_models:
                name = model_info.model or "unknown"
                size_bytes = model_info.size or 0
                size_mb = size_bytes / (1024 * 1024) if size_bytes else 0
                logger.warning(
                    "Found loaded model '%s' (%.0f MB) — evicting …", name, size_mb
                )
                await self._evict_model(name)

            await asyncio.sleep(1.0)
            logger.info("All models evicted — RAM available for next task")

        except Exception as exc:
            logger.warning("Could not query Ollama ps: %s (continuing)", exc)

    # -----------------------------------------------------------------------
    # Utility: Check if Model is Available
    # -----------------------------------------------------------------------
    async def is_model_available(self) -> bool:
        """
        Check if the text model is pulled/available in Ollama.
        Does NOT load the model — just checks the registry.
        Uses the official SDK `list()` method.
        """
        try:
            response = await self.client.list()
            models = response.models or []
            model_names = [m.model for m in models if m.model]
            available = any(self.model_name in name for name in model_names)

            if available:
                logger.info("Text model '%s' is available in Ollama ✓", self.model_name)
            else:
                logger.warning(
                    "Text model '%s' NOT found. Available: %s",
                    self.model_name,
                    model_names,
                )
            return available

        except Exception as exc:
            logger.error("Cannot reach Ollama at %s: %s", self.ollama_base_url, exc)
            return False
