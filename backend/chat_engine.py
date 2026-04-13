"""
=============================================================================
 Patho-Assist AI — Chat Engine (chat_engine.py)
=============================================================================
 Handles cross-modal chat using either:
   • LOCAL mode  → Ollama (gemma2:2b) running on the host machine
   • CLOUD mode  → Meta Llama API (Llama 3.3 70B) via official SDK

 Memory-Safe Execution Flow (LOCAL — 16 GB RAM constraint):
   1. Ensure NO other Ollama model is loaded (defensive eviction)
   2. Build a rich prompt combining:
      a) Retrieved PDF chunks from ChromaDB (RAG context)
      b) Vision description from the vision model (if available)
      c) The user's natural-language question
      d) Recent chat history for conversational continuity
   3. Call ollama.generate() with the combined prompt
   4. Collect the response (non-streaming for reliability)
   5. Immediately evict the model (keep_alive=0) → 0 bytes GPU/RAM

 Cloud Execution Flow (CLOUD — Llama API):
   1. Build a structured messages array (system + user content)
   2. Call AsyncLlamaAPIClient.chat.completions.create() with messages
   3. Return the generated answer (no eviction needed — stateless API)

 Uses the official `ollama` Python SDK (LOCAL) or `llama-api-client` (CLOUD).

 Public API:
   ChatEngine.generate_answer(question, rag_context, image_description, chat_history) → str
============================================================================="""

import asyncio
import logging
from typing import List, Dict, Optional

from ollama import AsyncClient as OllamaAsyncClient

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("patho-assist.chat")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Timeout for text generation (local mode)
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
    Manages the text/reasoning model lifecycle — dual-mode (local Ollama / cloud Llama API).

    LOCAL mode (16 GB RAM constraint):
      The engine ensures the text model is:
        loaded → used → IMMEDIATELY evicted (keep_alive=0)

    CLOUD mode:
      Uses Meta's hosted Llama 3.3 70B Instruct via the official SDK.
      No local GPU/RAM required. Stateless API — no eviction needed.

    Usage:
        engine = ChatEngine(run_mode="cloud", api_key="...", cloud_model_name="...")
        answer = await engine.generate_answer(question="...", rag_context=[...])
    """

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        model_name: str = "gemma2:2b",
        run_mode: str = "local",
        api_key: Optional[str] = None,
        cloud_model_name: Optional[str] = None,
    ):
        """
        Args:
            ollama_base_url:   URL of the local Ollama server (local mode).
            model_name:        Ollama model tag as shown by `ollama list`.
            run_mode:          "local" (Ollama) or "cloud" (Llama API).
            api_key:           Llama API key (cloud mode only).
            cloud_model_name:  Cloud model identifier (e.g. "Llama-3.3-70B-Instruct").
        """
        self.run_mode = run_mode.lower()
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.model_name = model_name
        self.cloud_model_name = cloud_model_name
        self.api_key = api_key

        # Initialise the appropriate client
        if self.run_mode == "cloud":
            if not api_key:
                raise ValueError("LLAMA_API_KEY is required for cloud mode.")
            from llama_api_client import AsyncLlamaAPIClient
            self.llama_client = AsyncLlamaAPIClient(api_key=api_key)
            self.ollama_client = None
            logger.info(
                "ChatEngine initialised (mode: CLOUD, model: %s)",
                cloud_model_name,
            )
        else:
            self.ollama_client = OllamaAsyncClient(host=self.ollama_base_url)
            self.llama_client = None
            logger.info(
                "ChatEngine initialised (mode: LOCAL, model: %s, url: %s)",
                model_name, self.ollama_base_url,
            )

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
        Full chat pipeline — delegates to local or cloud based on run_mode.

        Args:
            question:          The user's natural-language question.
            rag_context:       Retrieved PDF chunks from ChromaDB.
            image_description: Vision model's image analysis (None if no image).
            chat_history:      Previous Q&A turns in this session.

        Returns:
            The model's answer as a string.
        """
        logger.info("▶ Chat generation starting — question: '%s' (mode: %s)",
                     question[:80], self.run_mode.upper())

        if self.run_mode == "cloud":
            return await self._generate_cloud(
                question, rag_context, image_description, chat_history or []
            )
        else:
            return await self._generate_local(
                question, rag_context, image_description, chat_history or []
            )

    # -----------------------------------------------------------------------
    # CLOUD: Llama API Inference
    # -----------------------------------------------------------------------
    async def _generate_cloud(
        self,
        question: str,
        rag_context: List[Dict],
        image_description: Optional[str],
        chat_history: List[Dict],
    ) -> str:
        """
        Generate an answer using the Llama API (cloud mode).

        Builds a structured messages array with system prompt + user context,
        then calls the Llama API chat completions endpoint.
        """
        # Build the user content combining all context
        user_content = self._build_prompt(
            question=question,
            rag_context=rag_context,
            image_description=image_description,
            chat_history=chat_history,
        )

        logger.info(
            "Sending prompt to Llama API (%s) — cloud inference …",
            self.cloud_model_name,
        )

        try:
            response = await self.llama_client.chat.completions.create(
                model=self.cloud_model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.5,
                top_p=0.9,
            )
        except Exception as exc:
            logger.exception("Llama API chat call failed for model '%s'", self.cloud_model_name)
            raise RuntimeError(
                f"Chat inference failed (cloud): {exc}. "
                "Check your LLAMA_API_KEY and network connection."
            ) from exc

        # Extract the response text
        full_response = ""
        if hasattr(response, "completion_message") and response.completion_message:
            content = response.completion_message.content
            if isinstance(content, str):
                full_response = content.strip()
            elif isinstance(content, list):
                full_response = " ".join(
                    block.text for block in content
                    if hasattr(block, "text")
                ).strip()
            elif hasattr(content, "text"):
                full_response = content.text.strip()

        if not full_response:
            raise RuntimeError(
                f"Llama API returned an empty response for model '{self.cloud_model_name}'."
            )

        logger.info("✓ Cloud chat generation complete (%d chars)", len(full_response))
        return full_response

    # -----------------------------------------------------------------------
    # LOCAL: Ollama Inference (unchanged from original)
    # -----------------------------------------------------------------------
    async def _generate_local(
        self,
        question: str,
        rag_context: List[Dict],
        image_description: Optional[str],
        chat_history: List[Dict],
    ) -> str:
        """
        Generate an answer using Ollama (local mode).

        Full pipeline: evict → build prompt → infer → evict.
        """
        # Step 1: Defensively evict any lingering model from RAM
        await self._evict_all_models()

        # Step 2: Build the combined prompt
        prompt = self._build_prompt(
            question=question,
            rag_context=rag_context,
            image_description=image_description,
            chat_history=chat_history,
        )
        logger.info("Prompt built (%d chars, %d RAG chunks, image=%s)",
                     len(prompt), len(rag_context), image_description is not None)

        # Step 3: Run inference with keep_alive=0 (auto-evict after response)
        answer = await self._run_local_inference(prompt)

        # Step 4: Belt-and-suspenders — explicitly confirm eviction
        await self._evict_model(self.model_name)

        logger.info("✓ Local chat generation complete (%d chars). Model evicted.", len(answer))
        return answer

    # -----------------------------------------------------------------------
    # Private: Prompt Construction (shared by both modes)
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
          1. PDF context (RAG retrieval results)
          2. Image analysis context (if available)
          3. Recent chat history (for conversational continuity)
          4. The user's current question
        """
        sections: list[str] = []

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
            recent = chat_history[-3:]
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
    # Private: Ollama Inference (LOCAL mode)
    # -----------------------------------------------------------------------
    async def _run_local_inference(self, prompt: str) -> str:
        """
        Send the combined prompt to Ollama's text model via the official SDK.
        """
        full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"

        logger.info(
            "Sending prompt to Ollama (%s) via SDK generate() — generating answer …",
            self.model_name,
        )

        try:
            response = await self.ollama_client.generate(
                model=self.model_name,
                prompt=full_prompt,
                stream=False,
                keep_alive=0,
                options={
                    "temperature": 0.5,
                    "num_predict": 2048,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                },
            )
        except Exception as exc:
            logger.exception("Ollama generate() call failed for model '%s'", self.model_name)
            raise RuntimeError(
                f"Chat inference failed: {exc}. "
                f"Ensure Ollama is running and '{self.model_name}' is pulled "
                f"(`ollama pull {self.model_name}`)."
            ) from exc

        full_response = (response.response or "").strip()

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
    # Private: Model Memory Management (LOCAL mode only)
    # -----------------------------------------------------------------------
    async def _evict_model(self, model_name: str) -> None:
        """Force-evict a specific model from Ollama's memory."""
        if self.run_mode == "cloud":
            return
        try:
            await self.ollama_client.generate(
                model=model_name,
                prompt="",
                keep_alive=0,
                stream=False,
            )
            logger.info("Model '%s' evicted from RAM ✓", model_name)
        except Exception as exc:
            logger.debug("Eviction request for '%s' failed: %s (non-fatal)", model_name, exc)

    async def _evict_all_models(self) -> None:
        """Defensively evict ALL running models before loading a new one."""
        if self.run_mode == "cloud":
            return
        try:
            ps_response = await self.ollama_client.ps()
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
        Check if the text model is available.
        - LOCAL: checks Ollama's model registry.
        - CLOUD: verifies the Llama API connection.
        """
        if self.run_mode == "cloud":
            if not self.api_key:
                logger.warning("Cloud mode but no LLAMA_API_KEY set")
                return False
            try:
                models_response = await self.llama_client.models.list()
                logger.info("Llama API connected ✓ (text model: %s)", self.cloud_model_name)
                return True
            except Exception as exc:
                logger.error("Cannot reach Llama API: %s", exc)
                return False
        else:
            try:
                response = await self.ollama_client.list()
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
