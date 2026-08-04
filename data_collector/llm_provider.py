"""
OpenAICompatibleProvider - OpenAI-compatible interface provider.
Used to call locally deployed open-source models (vLLM, Ollama, text-generation-webui, etc.)
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Attempt to import the OpenAI SDK
try:
    from openai import AsyncOpenAI, OpenAI

    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    OpenAI = None
    AsyncOpenAI = None

# Attempt to import the Google GenAI SDK
try:
    from google import genai
    from google.genai import types as genai_types

    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False
    genai = None
    genai_types = None

# ==========================================
# Base Classes (Inlined for standalone usage)
# ==========================================


@dataclass
class AnalysisResult:
    """Wrapper for analysis results."""

    success: bool
    content: str | None = None
    error: str | None = None
    raw_response: Any = None

    @classmethod
    def ok(cls, content: str, raw_response: Any = None) -> AnalysisResult:
        return cls(success=True, content=content, raw_response=raw_response)

    @classmethod
    def fail(cls, error: str) -> AnalysisResult:
        return cls(success=False, error=error)


class ModelProvider:
    """Base class for model providers."""

    name: str = "base"
    supports_json_mode: bool = False

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        **kwargs,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.kwargs = kwargs

    def is_available(self) -> bool:
        return True


# ==========================================
# User Provided Implementation
# ==========================================


class OpenAICompatibleProvider(ModelProvider):
    """
    OpenAI-compatible interface provider.

    Supports any service that offers an OpenAI-compatible API, including:
    - vLLM (local deployment)
    - Ollama
    - text-generation-webui
    - LM Studio
    - the official OpenAI API
    """

    name = "openai_compatible"
    supports_json_mode = True

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        use_base64: bool = True,  # whether to encode images as base64
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        self.use_base64 = use_base64
        self._async_client: AsyncOpenAI | None = None
        self._initialize_client()

    def _initialize_client(self) -> bool:
        """Initialize the OpenAI-compatible client."""
        if not _OPENAI_AVAILABLE:
            logger.error("OpenAI SDK is not installed. Install it with: pip install openai")
            return False

        try:
            # API key (a placeholder works for local services)
            key = self.api_key or os.getenv("OPENAI_API_KEY") or "qwen-local-key"
            url = self.base_url or os.getenv("OPENAI_BASE_URL")

            # Synchronous client
            self._client = OpenAI(
                api_key=key,
                base_url=url,
            )

            # Asynchronous client
            self._async_client = AsyncOpenAI(
                api_key=key,
                base_url=url,
            )

            return True
        except Exception as e:
            logger.exception("Failed to initialize the OpenAI-compatible client: %s", e)
            self._client = None
            self._async_client = None
            return False

    def _encode_image_base64(self, image_path: Path) -> str:
        """Encode an image as base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _get_image_mime_type(self, image_path: Path) -> str:
        """Get the MIME type from the file extension."""
        suffix = image_path.suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        return mime_types.get(suffix, "image/jpeg")

    async def analyze_image_async(
        self,
        image_path: Path,
        prompt: str,
        json_mode: bool = False,
    ) -> AnalysisResult:
        """Analyze an image asynchronously."""
        if not self._async_client:
            return AnalysisResult.fail("OpenAI-compatible client not initialized")

        if not image_path.exists():
            return AnalysisResult.fail(f"Image file does not exist: {image_path}")

        try:
            # Initialize extra kwargs
            extra_kwargs: dict[str, Any] = {}
            if json_mode:
                extra_kwargs["response_format"] = {"type": "json_object"}

            # Detect whether this is a thinking model
            is_thinking_model = "thinking" in self.model_name.lower()

            # Thinking models: force-disable JSON mode to avoid truncating the thinking process
            if is_thinking_model and json_mode:
                json_mode = False
                if "response_format" in extra_kwargs:
                    del extra_kwargs["response_format"]

            # Build the message
            if self.use_base64:
                image_data = self._encode_image_base64(image_path)
                mime_type = self._get_image_mime_type(image_path)
                user_content = [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}},
                ]
            else:
                user_content = [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_path.resolve().as_uri()}},
                ]

            messages = [{"role": "user", "content": user_content}]

            # Anti-repetition parameters
            if "frequency_penalty" not in extra_kwargs:
                extra_kwargs["frequency_penalty"] = 0.1
            if "presence_penalty" not in extra_kwargs:
                extra_kwargs["presence_penalty"] = 0.1

            # Automatic retry logic (a local current_temperature keeps self.temperature unchanged)
            max_runaway_retries = 3
            current_temperature = self.temperature  # local copy; do not mutate the object state

            for attempt in range(max_runaway_retries + 1):
                try:
                    # Call the API asynchronously
                    response = await self._async_client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=current_temperature,
                        **extra_kwargs,
                    )

                    # Check for truncation (runaway check)
                    finish_reason = response.choices[0].finish_reason
                    if finish_reason == "length":
                        if attempt < max_runaway_retries:
                            logger.warning(
                                f"Thinking runaway detected (length truncated), retrying ({attempt + 1}/{max_runaway_retries})..."
                            )
                            continue
                        else:
                            logger.error("Thinking runaway retries exhausted; skipping this sample.")

                    # Extract the response text
                    message = response.choices[0].message
                    text = getattr(message, "content", None)

                    if isinstance(text, str) and text.strip():
                        # Strip <think> blocks
                        cleaned_text = text
                        if is_thinking_model:
                            if "</think>" in text:
                                cleaned_text = text.split("</think>")[-1].strip()
                            elif "<think>" in text:
                                cleaned_text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()

                        # Strip Markdown code fences
                        if "```json" in cleaned_text:
                            cleaned_text = cleaned_text.replace("```json", "").replace("```", "")

                        # Enhanced JSON extraction and validation
                        json_obj = None
                        validation_errors = []

                        try:
                            start_idx = cleaned_text.find("{")
                            end_idx = cleaned_text.rfind("}")

                            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                                potential_json = cleaned_text[start_idx : end_idx + 1]
                                json_obj = json.loads(potential_json)
                                cleaned_text = json.dumps(json_obj, ensure_ascii=False)
                            else:
                                validation_errors.append("No JSON brackets '{}' found")
                        except json.JSONDecodeError as e:
                            validation_errors.append(f"JSON Parse Failed: {str(e)[:50]}...")
                        except Exception as e:
                            validation_errors.append(f"JSON Extraction Error: {str(e)}")

                        if not json_obj:
                            pass
                        else:
                            # Simple validation; adjust as needed
                            pass

                        if validation_errors:
                            if attempt < max_runaway_retries:
                                logger.warning(
                                    f"Output validation failed ({', '.join(validation_errors)}), retrying ({attempt + 1}/{max_runaway_retries})..."
                                )
                                current_temperature = min(current_temperature + 0.1, 1.0)
                                continue
                            else:
                                logger.error(f"Final validation failed: {', '.join(validation_errors)}")
                                return AnalysisResult.fail(f"Validation Failed: {', '.join(validation_errors)}")

                        return AnalysisResult.ok(cleaned_text.strip(), raw_response=response)

                    continue

                except Exception as e:
                    if attempt == max_runaway_retries:
                        raise e
                    logger.warning(f"API Error during attempt {attempt}: {e}")

            return AnalysisResult.fail("Max retries exceeded with invalid output")

        except Exception as e:
            logger.exception("OpenAI-compatible API analysis failed: %s", e)
            return AnalysisResult.fail(str(e))


class GeminiProvider(ModelProvider):
    """
    Google Gemini API provider (uses the google-genai SDK).

    Supports the Gemini model family, including:
    - gemini-3.1-flash-lite-preview
    - gemini-3.1-flash-preview
    - gemini-2.5-flash
    - gemini-2.5-pro

    Usage:
        provider = GeminiProvider(
            model_name="gemini-3.1-flash-lite-preview",
            api_key="your-gemini-api-key",
        )
        result = await provider.analyze_image_async(image_path, prompt, json_mode=True)
    """

    name = "gemini"
    supports_json_mode = True

    def __init__(
        self,
        model_name: str = "gemini-3.1-flash-lite-preview",
        api_key: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.9,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name, api_key=api_key, max_tokens=max_tokens, temperature=temperature, **kwargs
        )
        self._client = None
        self._initialize_client()

    def _initialize_client(self) -> bool:
        """Initialize the Google GenAI client."""
        if not _GENAI_AVAILABLE:
            logger.error("google-genai SDK is not installed. Install it with: pip install google-genai")
            return False

        try:
            key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not key:
                logger.error("Gemini API key is not set. Set GEMINI_API_KEY or pass api_key.")
                return False

            self._client = genai.Client(api_key=key)
            logger.info("Gemini client initialized (model=%s)", self.model_name)
            return True
        except Exception as e:
            logger.exception("Failed to initialize the Gemini client: %s", e)
            self._client = None
            return False

    def is_available(self) -> bool:
        return self._client is not None

    def _get_image_mime_type(self, image_path: Path) -> str:
        """Get the MIME type from the file extension."""
        suffix = image_path.suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        return mime_types.get(suffix, "image/jpeg")

    def _clean_response(self, text: str) -> str:
        """Clean the Gemini response text and extract JSON."""
        if not text:
            return ""

        # Strip Markdown code fences
        cleaned = text.strip()
        if "```json" in cleaned:
            cleaned = cleaned.replace("```json", "").replace("```", "")
        elif "```" in cleaned:
            cleaned = cleaned.replace("```", "")

        # Extract JSON
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            potential_json = cleaned[start_idx : end_idx + 1]
            try:
                json_obj = json.loads(potential_json)
                return json.dumps(json_obj, ensure_ascii=False)
            except json.JSONDecodeError:
                pass

        return cleaned.strip()

    async def analyze_image_async(
        self,
        image_path: Path,
        prompt: str,
        json_mode: bool = False,
    ) -> AnalysisResult:
        """Analyze an image asynchronously with the Gemini API.

        Note: generate_content in the google-genai SDK is synchronous; it is
        wrapped with asyncio.to_thread to support async calls.
        """
        if not self._client:
            return AnalysisResult.fail("Gemini client not initialized")

        if not image_path.exists():
            return AnalysisResult.fail(f"Image file does not exist: {image_path}")

        max_retries = 3

        for attempt in range(max_retries + 1):
            try:
                # Read the image file
                with open(image_path, "rb") as f:
                    image_bytes = f.read()

                mime_type = self._get_image_mime_type(image_path)

                # Build the Gemini content
                image_part = genai_types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type,
                )

                # Build the generation config
                gen_config = genai_types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )

                # Enable JSON mode if requested
                if json_mode:
                    gen_config.response_mime_type = "application/json"

                # Wrap the synchronous call with asyncio.to_thread
                response = await asyncio.to_thread(
                    self._client.models.generate_content,
                    model=self.model_name,
                    contents=[prompt, image_part],
                    config=gen_config,
                )

                # Extract the text
                text = response.text
                if not text or not text.strip():
                    if attempt < max_retries:
                        logger.warning(f"Gemini returned an empty response, retrying ({attempt + 1}/{max_retries})...")
                        await asyncio.sleep(1)
                        continue
                    return AnalysisResult.fail("Gemini returned empty response after retries")

                # Clean the response
                cleaned_text = self._clean_response(text)

                # Validate JSON
                if json_mode:
                    try:
                        json.loads(cleaned_text)
                    except json.JSONDecodeError:
                        if attempt < max_retries:
                            logger.warning(f"Gemini JSON parse failed, retrying ({attempt + 1}/{max_retries})...")
                            await asyncio.sleep(1)
                            continue
                        return AnalysisResult.fail(f"JSON parse failed: {cleaned_text[:100]}")

                return AnalysisResult.ok(cleaned_text, raw_response=response)

            except Exception as e:
                error_str = str(e)
                # Rate limit handling
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"Gemini rate limit hit, waiting {wait}s ({attempt + 1}/{max_retries})...")
                    await asyncio.sleep(wait)
                    continue
                # Server errors
                if "500" in error_str or "503" in error_str:
                    if attempt < max_retries:
                        logger.warning(f"Gemini server error, retrying ({attempt + 1}/{max_retries}): {error_str[:80]}")
                        await asyncio.sleep(2)
                        continue

                if attempt >= max_retries:
                    logger.error(f"Gemini API call failed: {e}")
                    return AnalysisResult.fail(str(e))

        return AnalysisResult.fail("Max retries exceeded")
