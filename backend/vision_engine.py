"""
=============================================================================
 Patho-Assist AI — Vision Engine (vision_engine.py)
=============================================================================
 Handles histopathology image analysis via Ollama's paligemma model.

 Memory-Safe Execution Flow (16 GB RAM constraint):
   1. Ensure NO other Ollama model is loaded (defensive eviction)
   2. Read the image file as raw bytes
   3. Call ollama.generate() with the image bytes in the `images` param
   4. Collect the full text description
   5. Immediately evict paligemma (keep_alive=0) → 0 bytes GPU/RAM

 Uses the official `ollama` Python SDK (AsyncClient) instead of raw
 httpx calls. The SDK handles endpoint routing, serialisation, and
 error handling internally — eliminating 404 issues with manual URLs.

 Public API:
   VisionEngine.analyze_image(image_path) → str
=============================================================================
"""

import base64
import asyncio
import logging
from pathlib import Path

from ollama import AsyncClient

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("patho-assist.vision")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# The medical prompt sent alongside the image.
# Carefully crafted to elicit structured, clinically useful output.
VISION_PROMPT = (
    "You are a histopathology expert AI assistant. "
    "Analyze this histopathology slide image in detail. "
    "Describe the following:\n"
    "1. Tissue type and architecture observed\n"
    "2. Cellular morphology and any abnormalities\n"
    "3. Nuclear features (size, shape, chromatin pattern)\n"
    "4. Any signs of dysplasia, neoplasia, or malignancy\n"
    "5. Inflammatory infiltrates or necrosis if present\n"
    "6. Overall assessment and notable findings\n\n"
    "Be specific, thorough, and use proper medical terminology."
)


class VisionEngine:
    """
    Manages the paligemma vision model lifecycle via the official Ollama SDK.

    The 16 GB RAM constraint means only ONE model can be loaded at a time.
    This engine ensures paligemma is:
      loaded → used → IMMEDIATELY evicted (keep_alive=0)

    Usage:
        engine = VisionEngine(ollama_base_url="http://localhost:11434")
        description = await engine.analyze_image("/path/to/slide.png")
        # At this point, paligemma is already evicted from RAM.
    """

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        model_name: str = "paligemma",
    ):
        """
        Args:
            ollama_base_url: URL of the local Ollama server.
            model_name:      Model tag as shown by `ollama list`.
        """
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.model_name = model_name
        # Create the async client pointing at our local Ollama instance
        self.client = AsyncClient(host=self.ollama_base_url)
        logger.info("VisionEngine initialised (model: %s, url: %s)", model_name, self.ollama_base_url)

    # -----------------------------------------------------------------------
    # Public: Analyze Image
    # -----------------------------------------------------------------------
    async def analyze_image(self, image_path: str) -> str:
        """
        Full vision pipeline: encode → infer → evict.

        Args:
            image_path: Absolute path to the histopathology image file.

        Returns:
            A detailed text description of the cellular anomalies.

        Raises:
            FileNotFoundError: If the image file doesn't exist.
            RuntimeError: If the model fails to generate a response.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        logger.info("▶ Vision analysis starting: %s", image_path.name)

        # Step 1: Defensively evict any lingering model from RAM
        await self._evict_all_models()

        # Step 2: Read the image as raw bytes (SDK accepts bytes directly)
        image_bytes = image_path.read_bytes()
        logger.info("Image read (%d bytes)", len(image_bytes))

        # Step 3: Run inference with keep_alive=0 (auto-evict after response)
        description = await self._run_inference(image_bytes)

        # Step 4: Belt-and-suspenders — explicitly confirm eviction
        await self._evict_model(self.model_name)

        logger.info("✓ Vision analysis complete (%d chars). Model evicted.", len(description))
        return description

    # -----------------------------------------------------------------------
    # Private: Ollama Inference via Official SDK
    # -----------------------------------------------------------------------
    async def _run_inference(self, image_bytes: bytes) -> str:
        """
        Send the image + medical prompt to Ollama using the official SDK.

        Uses `client.generate()` (NOT `client.chat()`) because paligemma
        only supports the generate endpoint. The SDK handles the correct
        endpoint routing and payload serialisation internally.

        The `images` parameter accepts a list of raw bytes or base64 strings.
        The `keep_alive` parameter is set to 0 to evict the model immediately
        after generation completes — this is the cornerstone of our 16 GB
        memory-swap strategy.
        """
        logger.info(
            "Sending image to Ollama (%s) via SDK generate() — this may take a while …",
            self.model_name,
        )

        try:
            response = await self.client.generate(
                model=self.model_name,
                prompt=VISION_PROMPT,
                images=[image_bytes],       # SDK accepts raw bytes directly
                stream=False,               # Get the full response at once
                keep_alive=0,               # ⚠️ CRITICAL: Evict immediately after
                options={
                    "temperature": 0.3,     # Low temp for factual medical analysis
                    "num_predict": 1024,    # Max tokens for the description
                },
            )
        except Exception as exc:
            logger.exception("Ollama generate() call failed for model '%s'", self.model_name)
            raise RuntimeError(
                f"Vision inference failed: {exc}. "
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
                "Ensure the model is pulled (`ollama pull paligemma`) and supports vision."
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
        Defensively evict ALL known models before loading a new one.

        Queries Ollama's running model list via the SDK, then evicts
        each one. This guarantees a clean RAM slate.
        """
        try:
            ps_response = await self.client.ps()
            running_models = ps_response.models or []

            if not running_models:
                logger.info("No models currently loaded in Ollama — RAM is clear")
                return

            for model_info in running_models:
                name = model_info.model or "unknown"
                size_bytes = model_info.size or 0
                size_mb = size_bytes / (1024 * 1024) if size_bytes else 0
                logger.warning(
                    "Found loaded model '%s' (%.0f MB) — evicting …", name, size_mb
                )
                await self._evict_model(name)

            # Brief pause to let Ollama fully release memory
            await asyncio.sleep(1.0)
            logger.info("All models evicted — RAM available for next task")

        except Exception as exc:
            logger.warning("Could not query Ollama ps: %s (continuing anyway)", exc)

    # -----------------------------------------------------------------------
    # Utility: Check if Model is Available
    # -----------------------------------------------------------------------
    async def is_model_available(self) -> bool:
        """
        Check if the vision model is pulled/available in Ollama.
        Does NOT load the model — just checks the model registry.
        Uses the official SDK `list()` method.
        """
        try:
            response = await self.client.list()
            models = response.models or []
            model_names = [m.model for m in models if m.model]
            available = any(self.model_name in name for name in model_names)

            if available:
                logger.info("Vision model '%s' is available in Ollama ✓", self.model_name)
            else:
                logger.warning(
                    "Vision model '%s' NOT found. Available: %s",
                    self.model_name,
                    model_names,
                )
            return available

        except Exception as exc:
            logger.error("Cannot reach Ollama at %s: %s", self.ollama_base_url, exc)
            return False
