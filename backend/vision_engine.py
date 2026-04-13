"""
=============================================================================
 Patho-Assist AI — Vision Engine (vision_engine.py)
=============================================================================
 Handles histopathology image analysis via either:
   • LOCAL mode  → Ollama (llava) running on the host machine
   • CLOUD mode  → Meta Llama API (Llama 4 Maverick) via official SDK

 Memory-Safe Execution Flow (LOCAL — 16 GB RAM constraint):
   1. Ensure NO other Ollama model is loaded (defensive eviction)
   2. Read the image file as raw bytes
   3. Call ollama.generate() with the image bytes in the `images` param
   4. Collect the full text description
   5. Immediately evict the model (keep_alive=0) → 0 bytes GPU/RAM

 Cloud Execution Flow (CLOUD — Llama API):
   1. Read the image file and base64-encode it
   2. Call AsyncLlamaAPIClient.chat.completions.create() with inline
      base64 image data URI + medical prompt
   3. Return the generated description (no eviction needed — stateless API)

 Public API:
   VisionEngine.analyze_image(image_path) → str
=============================================================================
"""

import base64
import asyncio
import logging
from pathlib import Path
from typing import Optional

from ollama import AsyncClient as OllamaAsyncClient

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
    Manages the vision model lifecycle — dual-mode (local Ollama / cloud Llama API).

    LOCAL mode (16 GB RAM constraint):
      The engine ensures the vision model is:
        loaded → used → IMMEDIATELY evicted (keep_alive=0)

    CLOUD mode:
      Uses Meta's hosted Llama 4 Maverick (multimodal) via the official SDK.
      No local GPU/RAM required. Stateless API — no eviction needed.

    Usage:
        engine = VisionEngine(run_mode="cloud", api_key="...", cloud_model_name="...")
        description = await engine.analyze_image("/path/to/slide.png")
    """

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        model_name: str = "llava",
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
            cloud_model_name:  Cloud model identifier (e.g. "Llama-4-Maverick-17B-128E-Instruct-FP8").
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
            # Lazy import to avoid dependency issues when running in local-only mode
            from llama_api_client import AsyncLlamaAPIClient
            self.llama_client = AsyncLlamaAPIClient(api_key=api_key)
            self.ollama_client = None
            logger.info(
                "VisionEngine initialised (mode: CLOUD, model: %s)",
                cloud_model_name,
            )
        else:
            self.ollama_client = OllamaAsyncClient(host=self.ollama_base_url)
            self.llama_client = None
            logger.info(
                "VisionEngine initialised (mode: LOCAL, model: %s, url: %s)",
                model_name, self.ollama_base_url,
            )

    # -----------------------------------------------------------------------
    # Public: Analyze Image
    # -----------------------------------------------------------------------
    async def analyze_image(self, image_path: str) -> str:
        """
        Full vision pipeline — delegates to local or cloud based on run_mode.

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

        logger.info("▶ Vision analysis starting: %s (mode: %s)", image_path.name, self.run_mode.upper())

        if self.run_mode == "cloud":
            return await self._analyze_cloud(image_path)
        else:
            return await self._analyze_local(image_path)

    # -----------------------------------------------------------------------
    # CLOUD: Llama API Inference
    # -----------------------------------------------------------------------
    async def _analyze_cloud(self, image_path: Path) -> str:
        """
        Analyze the image using the Llama API (cloud mode).

        Sends the image as a base64 data URI in a multimodal
        chat completions request. Llama 4 Maverick natively supports
        vision input via the standard content blocks format.
        """
        # Read and base64-encode the image
        image_bytes = image_path.read_bytes()
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        # Determine MIME type from extension
        ext = image_path.suffix.lower()
        mime_map = {
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".tiff": "image/tiff", ".bmp": "image/bmp", ".webp": "image/webp",
        }
        mime_type = mime_map.get(ext, "image/jpeg")

        logger.info(
            "Sending image to Llama API (%s) — cloud inference …",
            self.cloud_model_name,
        )

        try:
            response = await self.llama_client.chat.completions.create(
                model=self.cloud_model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": VISION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{b64_image}"
                                },
                            },
                        ],
                    }
                ],
                temperature=0.3,
            )
        except Exception as exc:
            logger.exception("Llama API vision call failed for model '%s'", self.cloud_model_name)
            raise RuntimeError(
                f"Vision inference failed (cloud): {exc}. "
                "Check your LLAMA_API_KEY and network connection."
            ) from exc

        # Extract the response text
        full_response = ""
        if hasattr(response, "completion_message") and response.completion_message:
            content = response.completion_message.content
            if isinstance(content, str):
                full_response = content.strip()
            elif isinstance(content, list):
                # Content may be a list of content blocks
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

        logger.info("✓ Cloud vision analysis complete (%d chars)", len(full_response))
        return full_response

    # -----------------------------------------------------------------------
    # LOCAL: Ollama Inference (unchanged from original)
    # -----------------------------------------------------------------------
    async def _analyze_local(self, image_path: Path) -> str:
        """
        Analyze the image using Ollama (local mode).

        Full pipeline: evict → read image → infer → evict.
        """
        # Step 1: Defensively evict any lingering model from RAM
        await self._evict_all_models()

        # Step 2: Read the image as raw bytes (SDK accepts bytes directly)
        image_bytes = image_path.read_bytes()
        logger.info("Image read (%d bytes)", len(image_bytes))

        # Step 3: Run inference with keep_alive=0 (auto-evict after response)
        description = await self._run_local_inference(image_bytes)

        # Step 4: Belt-and-suspenders — explicitly confirm eviction
        await self._evict_model(self.model_name)

        logger.info("✓ Local vision analysis complete (%d chars). Model evicted.", len(description))
        return description

    async def _run_local_inference(self, image_bytes: bytes) -> str:
        """
        Send the image + medical prompt to Ollama using the official SDK.
        """
        logger.info(
            "Sending image to Ollama (%s) via SDK generate() — this may take a while …",
            self.model_name,
        )

        try:
            response = await self.ollama_client.generate(
                model=self.model_name,
                prompt=VISION_PROMPT,
                images=[image_bytes],
                stream=False,
                keep_alive=0,
                options={
                    "temperature": 0.3,
                    "num_predict": 1024,
                },
            )
        except Exception as exc:
            logger.exception("Ollama generate() call failed for model '%s'", self.model_name)
            raise RuntimeError(
                f"Vision inference failed: {exc}. "
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
                f"Ensure the model is pulled (`ollama pull {self.model_name}`) and supports vision."
            )

        return full_response

    # -----------------------------------------------------------------------
    # Private: Model Memory Management (LOCAL mode only)
    # -----------------------------------------------------------------------
    async def _evict_model(self, model_name: str) -> None:
        """Force-evict a specific model from Ollama's memory."""
        if self.run_mode == "cloud":
            return  # No eviction needed for cloud mode
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
            return  # No eviction needed for cloud mode
        try:
            ps_response = await self.ollama_client.ps()
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

            await asyncio.sleep(1.0)
            logger.info("All models evicted — RAM available for next task")

        except Exception as exc:
            logger.warning("Could not query Ollama ps: %s (continuing anyway)", exc)

    # -----------------------------------------------------------------------
    # Utility: Check if Model is Available
    # -----------------------------------------------------------------------
    async def is_model_available(self) -> bool:
        """
        Check if the vision model is available.
        - LOCAL: checks Ollama's model registry.
        - CLOUD: verifies API key is set and makes a lightweight models.list() call.
        """
        if self.run_mode == "cloud":
            if not self.api_key:
                logger.warning("Cloud mode but no LLAMA_API_KEY set")
                return False
            try:
                # Lightweight check — list available models
                models_response = await self.llama_client.models.list()
                logger.info("Llama API connected ✓ (vision model: %s)", self.cloud_model_name)
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
