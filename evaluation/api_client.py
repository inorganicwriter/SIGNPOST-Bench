"""
evaluation/api_client.py
========================
Unified API client for multimodal geo-localization evaluation.

Supports two provider backends:
  - Google Gemini API via google-genai SDK
  - OpenAI-compatible Sponsor gateway

Usage:
    from evaluation.api_client import build_client

    client = build_client("gpt-5.4")
    result = client.predict_location(base64_image)
"""

import base64
import io
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

# Try importing google-genai SDK (same as generate_attacks.py)
try:
    from google import genai
    from google.genai import types as genai_types

    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False
    genai = None
    genai_types = None

# Fallback to requests for non-genai providers
import requests

# Ensure repo root is on sys.path for evaluation.prompts import.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.prompts import COORDINATE_PROMPT

logger = logging.getLogger(__name__)

# Maximum temperature for retry escalation (prevents runaway temperature)
_MAX_RETRY_TEMP = 1.0


# ===========================================================================
#  Provider Configuration Registry
# ===========================================================================

PROVIDER_CONFIGS = {
    "sponsor": {
        "api_base": os.environ.get("SPONSOR_API_BASE", ""),
        "api_key": os.environ.get("SPONSOR_API_KEY", ""),
    },
    "gemini": {
        "api_base": "",
        "api_key": os.environ.get("GEMINI_API_KEY", ""),
    },
}

# Provider-specific client defaults.
# Note: max_retries means "additional retries after the first request".
DEFAULT_PROVIDER_TIMEOUTS = {
    "sponsor": 120,
    "gemini": 120,
}
DEFAULT_PROVIDER_MAX_RETRIES = {
    "sponsor": 1,  # 2 total attempts
    "gemini": 5,  # 6 total attempts
}

# ===========================================================================
#  Model Registry: short_name -> {model_id, provider, tier}
#  tier: "main" (13 flagships) | "secondary" (7 lightweight/cost-efficient)
# ===========================================================================

MODEL_REGISTRY = {
    # =========================================================
    # Main Tier (13 models) — Flagships & strong reasoners
    # =========================================================
    # --- Anthropic ---
    "claude-opus-4.6": {
        "model": "claude-opus-4-6",
        "provider": "sponsor",
        "thinking": False,
        "tier": "main",
        "freq_penalty": False,
    },
    "claude-sonnet-4.6": {
        "model": "claude-sonnet-4-6",
        "provider": "sponsor",
        "thinking": False,
        "tier": "main",
        "freq_penalty": False,
    },
    # --- OpenAI ---
    "gpt-5.4": {"model": "gpt-5.4", "provider": "sponsor", "thinking": False, "tier": "main", "freq_penalty": False},
    "gpt-5": {"model": "gpt-5", "provider": "sponsor", "thinking": False, "tier": "main", "freq_penalty": False},
    "gpt-4o": {"model": "gpt-4o", "provider": "sponsor", "thinking": False, "tier": "main"},
    # --- Google Gemini ---
    "gemini-3.1-pro": {"model": "gemini-3.1-pro-preview", "provider": "gemini", "thinking": False, "tier": "main"},
    "gemini-2.5-pro": {"model": "gemini-2.5-pro", "provider": "sponsor", "thinking": False, "tier": "main"},
    # --- Moonshot / Kimi ---
    "kimi-k2.5": {"model": "kimi-k2.5", "provider": "sponsor", "thinking": False, "tier": "main"},
    "moonshot-128k-vision": {
        "model": "moonshot-v1-128k-vision-preview",
        "provider": "sponsor",
        "thinking": False,
        "tier": "main",
    },
    # --- Alibaba Qwen (VL = Vision-Language) ---
    "qwen3-vl-plus": {"model": "qwen3-vl-plus", "provider": "sponsor", "thinking": False, "tier": "main"},
    "qwen3-vl-235b": {"model": "qwen3-vl-235b-a22b-instruct", "provider": "sponsor", "thinking": False, "tier": "main"},
    # --- ByteDance / Doubao ---
    "seed-2.0-pro": {"model": "doubao-seed-2-0-pro-260215", "provider": "sponsor", "thinking": False, "tier": "main"},
    # --- xAI ---
    "grok-4": {"model": "grok-4", "provider": "sponsor", "thinking": False, "tier": "main", "freq_penalty": False},
    # =========================================================
    # Secondary Tier (7 models) — Lightweight & cost-efficient
    # =========================================================
    # --- Anthropic ---
    "claude-haiku-4.5": {
        "model": "claude-haiku-4-5",
        "provider": "sponsor",
        "thinking": False,
        "tier": "secondary",
        "freq_penalty": False,
    },
    # --- OpenAI ---
    "gpt-4o-mini": {"model": "gpt-4o-mini", "provider": "sponsor", "thinking": False, "tier": "secondary"},
    # --- Google Gemini ---
    "gemini-3-flash": {"model": "gemini-3-flash", "provider": "sponsor", "thinking": False, "tier": "secondary"},
    "gemini-2.5-flash": {"model": "gemini-2.5-flash", "provider": "sponsor", "thinking": False, "tier": "secondary"},
    # --- Alibaba Qwen (VL) ---
    "qwen3-vl-30b": {
        "model": "qwen3-vl-30b-a3b-instruct",
        "provider": "sponsor",
        "thinking": False,
        "tier": "secondary",
    },
    # --- ByteDance / Doubao ---
    "seed-2.0-lite": {
        "model": "doubao-seed-2-0-lite-260215",
        "provider": "sponsor",
        "thinking": False,
        "tier": "secondary",
    },
    # --- Moonshot ---
    "moonshot-32k-vision": {
        "model": "moonshot-v1-32k-vision-preview",
        "provider": "sponsor",
        "thinking": False,
        "tier": "secondary",
    },
}


# ===========================================================================
#  Unified API Client
# ===========================================================================


class GeoLocalizationClient:
    """
    Unified client for geo-localization inference across multiple API providers.
    """

    PROMPT = COORDINATE_PROMPT

    def __init__(
        self,
        model_name: str,
        api_base: str,
        api_key: str,
        provider: str = "sponsor",
        is_thinking_model: bool = False,
        max_tokens: int = 8192,
        max_retries: int = 5,
        timeout: int = 120,
        supports_frequency_penalty: bool = True,
        custom_prompt: str | None = None,
        skip_image: bool = False,
    ):
        self.model_name = model_name
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.provider = provider
        self.is_thinking_model = is_thinking_model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.timeout = timeout
        self.custom_prompt = custom_prompt
        self.skip_image = skip_image

        # Gemini genai client (lazy init)
        self._genai_client = None

        # Provider defaults
        self.supports_frequency_penalty = supports_frequency_penalty
        self.max_image_size_mb = 20
        self.extra_headers = {}

    def _build_headers(self) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.extra_headers)
        return headers

    def _compress_image_if_needed(self, base64_image: str) -> str:
        """Compress image if it exceeds provider's size limit."""
        if self.max_image_size_mb is None:
            return base64_image
        size_mb = len(base64_image) * 3 / 4 / (1024 * 1024)
        if size_mb <= self.max_image_size_mb:
            return base64_image
        try:
            from PIL import Image

            img_bytes = base64.b64decode(base64_image)
            img = Image.open(io.BytesIO(img_bytes))
            max_pixels = int(self.max_image_size_mb * 1024 * 1024 * 0.75)
            w, h = img.size
            scale = (max_pixels / (w * h * 3)) ** 0.5
            if scale < 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            compressed = base64.b64encode(buf.getvalue()).decode("utf-8")
            return compressed
        except ImportError:
            logger.warning("Pillow not installed, cannot compress image. pip install Pillow")
            return base64_image
        except Exception as e:
            logger.warning(f"Image compression failed: {e}")
            return base64_image

    def _clean_thinking_tags(self, text: str) -> str:
        """Strip <think>...</think> tags from Thinking model output."""
        if not text:
            return ""
        if "</think>" in text:
            text = text.split("</think>")[-1]
        elif "<think>" in text:
            text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
        text = re.sub(r"```[a-z]*\n?", "", text)
        return text.strip()

    # =============================================
    #  Retry & error handling helpers
    # =============================================

    def _should_retry_error(self, error_str: str) -> str:
        """Classify error for retry strategy. Returns: 'rate_limit', 'server_error', or 'fatal'."""
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            return "rate_limit"
        if any(code in error_str for code in ("500", "502", "503")):
            return "server_error"
        return "fatal"

    def _retry_wait(self, error_type: str, attempt: int):
        """Sleep based on error type and attempt number."""
        if error_type == "rate_limit":
            wait = min(2 ** (attempt + 1), 60)
            logger.warning(f"  ⏳ Rate limited, waiting {wait}s (attempt {attempt + 1}/{self.max_retries + 1})...")
            time.sleep(wait)
        elif error_type == "server_error":
            time.sleep(2 * (attempt + 1))
        else:
            time.sleep(1)

    def _validate_and_clean_response(self, text: str, attempt: int) -> str | None:
        """
        Validate response text: clean thinking tags, check for parseable coordinates.
        Returns cleaned content if valid, or None to signal retry needed.
        If on last attempt, returns content even without valid coordinates.
        """
        if not text or not text.strip():
            return None

        content = text.strip()
        if self.is_thinking_model:
            content = self._clean_thinking_tags(content)

        if not content:
            return None

        lat, lon = self.parse_coordinates(content)
        if lat is not None and lon is not None:
            return content

        # No valid coordinates — retry if possible, else return raw content
        if attempt >= self.max_retries:
            return content
        return None  # Signal: retry

    # =============================================
    #  Google AI Studio (Gemini API) — via google-genai SDK
    # =============================================

    def _ensure_genai_client(self):
        """Lazily initialize the genai client for Gemini API access."""
        if not _GENAI_AVAILABLE:
            raise RuntimeError("google-genai SDK not installed. Run: pip install google-genai")
        if self._genai_client is None:
            logger.info("Initializing Gemini client for %s via direct API key", self.model_name)
            self._genai_client = genai.Client(api_key=self.api_key)

    def _detect_mime_from_bytes(self, data: bytes) -> str:
        """Detect image MIME type from file header bytes."""
        if data[:3] == b"\xff\xd8\xff":
            return "image/jpeg"
        if data[:4] == b"\x89PNG":
            return "image/png"
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return "image/webp"
        if data[:3] == b"GIF":
            return "image/gif"
        if data[:2] == b"BM":
            return "image/bmp"
        return "image/jpeg"  # default fallback

    def _detect_mime_from_extension(self, path: Path) -> str:
        """Detect image MIME type from file extension."""
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        return mime_map.get(path.suffix.lower(), "image/jpeg")

    def _call_gemini(
        self,
        image_part,
        prompt: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> str | None:
        """
        Core Gemini API call with retry logic. Shared by base64 and file-path methods.

        Args:
            image_part: A genai_types.Part containing the image data.
            prompt: Custom prompt text. If None, uses self.PROMPT.
            max_output_tokens: Override max tokens. If None, uses self.max_tokens.
            temperature: Generation temperature.
            json_mode: If True, request JSON response MIME type.

        Returns:
            Cleaned response text, or None on failure.
        """
        self._ensure_genai_client()
        actual_prompt = prompt or self.custom_prompt or self.PROMPT
        actual_max_tokens = max_output_tokens or self.max_tokens

        for attempt in range(self.max_retries + 1):
            try:
                gen_config = genai_types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=actual_max_tokens,
                )
                if json_mode:
                    gen_config.response_mime_type = "application/json"

                contents = [actual_prompt] if self.skip_image else [actual_prompt, image_part]
                response = self._genai_client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=gen_config,
                )

                result = self._validate_and_clean_response(response.text, attempt)
                if result is not None:
                    return result

                # No valid result, retry
                if attempt < self.max_retries:
                    time.sleep(1)
                    continue

            except Exception as e:
                error_str = str(e)
                error_type = self._should_retry_error(error_str)

                if error_type == "fatal" and attempt >= self.max_retries:
                    logger.error(f"   Gemini error: {error_str[:150]}")
                    return None
                if error_type == "fatal":
                    # Some fatal errors should not retry (e.g., 400 bad request)
                    if "400" in error_str or "INVALID_ARGUMENT" in error_str:
                        logger.error(f"   Gemini fatal error: {error_str[:150]}")
                        return None

                self._retry_wait(error_type, attempt)

        return None

    def _predict_gemini_from_base64(self, base64_image: str) -> str | None:
        """Gemini API via google-genai SDK from base64."""
        image_bytes = base64.b64decode(base64_image)
        mime_type = self._detect_mime_from_bytes(image_bytes)
        image_part = genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        return self._call_gemini(image_part)

    def _predict_gemini_from_path(self, image_path: str) -> str | None:
        """Gemini API via google-genai SDK from file path."""
        p = Path(image_path)
        if not p.exists():
            logger.error(f"   Image not found: {image_path}")
            return None

        with open(p, "rb") as f:
            image_bytes = f.read()

        mime_type = self._detect_mime_from_extension(p)
        image_part = genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        return self._call_gemini(image_part)

    # =============================================
    #  OpenAI-compatible providers
    # =============================================
    def _build_openai_payload(self, base64_image: str, temperature: float, mime_type: str = "image/jpeg") -> dict:
        """Build OpenAI-compatible chat completion payload."""
        prompt_text = self.custom_prompt or self.PROMPT
        if self.skip_image:
            content = prompt_text
        else:
            content = [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}},
            ]
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "max_completion_tokens": self.max_tokens,
        }
        if self.supports_frequency_penalty:
            payload["frequency_penalty"] = 0.1
        return payload

    def _predict_openai_compatible(self, base64_image: str) -> str | None:
        """OpenAI-compatible chat completion API."""
        url = f"{self.api_base}/chat/completions"
        headers = self._build_headers()
        mime_type = "image/jpeg"
        try:
            mime_type = self._detect_mime_from_bytes(base64.b64decode(base64_image))
        except Exception:
            pass
        current_temp = 0.0
        # Track whether to skip max_completion_tokens (some APIs reject it)
        _skip_max_completion_tokens = False

        for attempt in range(self.max_retries + 1):
            payload = self._build_openai_payload(base64_image, current_temp, mime_type=mime_type)
            if _skip_max_completion_tokens:
                payload.pop("max_completion_tokens", None)
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                response.raise_for_status()
                result = response.json()

                if "choices" not in result or not result["choices"]:
                    if attempt < self.max_retries:
                        logger.warning(
                            "   Empty choices from API for %s, retrying (%d/%d)...",
                            self.model_name,
                            attempt + 1,
                            self.max_retries + 1,
                        )
                    current_temp = min(current_temp + 0.1, _MAX_RETRY_TEMP)
                    continue

                choice = result["choices"][0]
                finish_reason = choice.get("finish_reason", "")

                if finish_reason == "length" and self.is_thinking_model:
                    if attempt < self.max_retries:
                        current_temp = min(current_temp + 0.1, _MAX_RETRY_TEMP)
                        continue
                    return None

                content = choice["message"].get("content", "")

                # Debug: log token usage and finish reason
                usage = result.get("usage", {})
                comp_tokens = usage.get("completion_tokens", 0)
                logger.info(
                    "   %s: prompt=%s, completion=%s, total=%s, finish=%s, max_tokens=%d",
                    self.model_name,
                    usage.get("prompt_tokens"),
                    comp_tokens,
                    usage.get("total_tokens"),
                    finish_reason,
                    self.max_tokens,
                )
                # Warn if output exceeds expected max_tokens
                if comp_tokens > self.max_tokens:
                    logger.warning(
                        "   %s: completion_tokens=%d exceeds max_tokens=%d! "
                        "API may be ignoring max_tokens/max_completion_tokens.",
                        self.model_name,
                        comp_tokens,
                        self.max_tokens,
                    )

                validated = self._validate_and_clean_response(content, attempt)
                if validated is not None:
                    return validated
                # Bump temperature on parse failure
                if attempt < self.max_retries:
                    logger.warning(
                        "   Unparseable response from API for %s, retrying (%d/%d)...",
                        self.model_name,
                        attempt + 1,
                        self.max_retries + 1,
                    )
                current_temp = min(current_temp + 0.15, _MAX_RETRY_TEMP)

            except requests.exceptions.HTTPError as e:
                response_obj = getattr(e, "response", None)
                status = response_obj.status_code if response_obj is not None else None
                error_body = ""
                try:
                    error_body = response_obj.text[:500] if response_obj is not None else ""
                except Exception:
                    pass

                if status == 400 and "max_completion_tokens" in error_body and not _skip_max_completion_tokens:
                    # Some APIs reject max_completion_tokens — retry without it
                    logger.warning(
                        "   %s: API rejected max_completion_tokens, retrying with max_tokens only.",
                        self.model_name,
                    )
                    _skip_max_completion_tokens = True
                    continue

                if status == 429:
                    logger.warning(
                        "   HTTP 429 from API for %s (attempt %d/%d).",
                        self.model_name,
                        attempt + 1,
                        self.max_retries + 1,
                    )
                    self._retry_wait("rate_limit", attempt)
                elif status in (500, 502, 503):
                    logger.warning(
                        "   HTTP %s from API for %s (attempt %d/%d).",
                        status,
                        self.model_name,
                        attempt + 1,
                        self.max_retries + 1,
                    )
                    self._retry_wait("server_error", attempt)
                else:
                    logger.error(f"   HTTP {status}: {e}")
                    return None
            except requests.exceptions.Timeout:
                logger.warning(
                    "   Request timeout after %ss for %s (attempt %d/%d).",
                    self.timeout,
                    self.model_name,
                    attempt + 1,
                    self.max_retries + 1,
                )
                self._retry_wait("server_error", attempt)
            except Exception as e:
                logger.error(f"  API Error (attempt {attempt + 1}): {e}")
                time.sleep(1)

        return None

    # =============================================
    #  Public API
    # =============================================

    def predict_location_from_path(self, image_path: str) -> str | None:
        """Run geo-localization from file path. Preferred for Gemini (uses genai SDK directly)."""
        if self.provider == "gemini":
            return self._predict_gemini_from_path(image_path)

        # Fallback: encode to base64 for other providers
        try:
            with open(image_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")
        except FileNotFoundError:
            logger.error(f"   Image not found: {image_path}")
            return None
        except Exception as e:
            logger.error(f"   Failed to read image: {e}")
            return None
        return self.predict_location(base64_image)

    def predict_location(self, base64_image: str) -> str | None:
        """Run geo-localization inference from base64. Returns raw text response or None."""
        base64_image = self._compress_image_if_needed(base64_image)

        # Dispatch by provider
        if self.provider == "gemini":
            return self._predict_gemini_from_base64(base64_image)

        # OpenAI-compatible providers (sponsor, etc.)
        return self._predict_openai_compatible(base64_image)

    def predict_custom_from_path(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> str | None:
        """
        Run custom prompt inference from file path.
        Used by probing/defense/generalization experiments.

        For Gemini provider: uses native google-genai SDK (no base64 overhead).
        For other providers: falls back to OpenAI-compatible API with base64.

        Args:
            image_path: Path to the image file.
            prompt: Custom prompt text.
            max_tokens: Max output tokens (default: self.max_tokens).
            temperature: Generation temperature.
            json_mode: If True and supported, request JSON response format.

        Returns:
            Model response text, or None on failure.
        """
        max_tokens = max_tokens or self.max_tokens

        if self.provider == "gemini":
            p = Path(image_path)
            if not p.exists():
                logger.error(f"   Image not found: {image_path}")
                return None
            with open(p, "rb") as f:
                image_bytes = f.read()
            mime_type = self._detect_mime_from_extension(p)
            image_part = genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            return self._call_gemini(
                image_part, prompt=prompt, max_output_tokens=max_tokens, temperature=temperature, json_mode=json_mode
            )

        # For non-Gemini providers: use OpenAI-compatible API with base64
        return self.predict_custom_from_base64(
            self._read_and_encode(image_path),
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def predict_custom_from_base64(
        self, base64_image: str, prompt: str, max_tokens: int | None = None, temperature: float = 0.0
    ) -> str | None:
        """
        Run custom prompt inference from base64 image.

        For Gemini: uses native genai SDK.
        For other providers: uses OpenAI-compatible chat completions API.
        """
        if base64_image is None:
            return None
        max_tokens = max_tokens or self.max_tokens
        base64_image = self._compress_image_if_needed(base64_image)

        if self.provider == "gemini":
            image_bytes = base64.b64decode(base64_image)
            mime_type = self._detect_mime_from_bytes(image_bytes)
            image_part = genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            return self._call_gemini(image_part, prompt=prompt, max_output_tokens=max_tokens, temperature=temperature)

        # OpenAI-compatible providers
        url = f"{self.api_base}/chat/completions"
        headers = self._build_headers()

        mime_type = "image/jpeg"
        try:
            mime_type = self._detect_mime_from_bytes(base64.b64decode(base64_image))
        except Exception:
            pass

        for attempt in range(self.max_retries + 1):
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}},
                        ],
                    }
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "max_completion_tokens": max_tokens,
            }
            if self.supports_frequency_penalty:
                payload["frequency_penalty"] = 0.1
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                response.raise_for_status()
                result = response.json()
                if "choices" in result and result["choices"]:
                    content = result["choices"][0]["message"].get("content", "")
                    if self.is_thinking_model:
                        content = self._clean_thinking_tags(content)
                    if content and content.strip():
                        return content.strip()
                if attempt < self.max_retries:
                    time.sleep(1)
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response else None
                if status == 429:
                    self._retry_wait("rate_limit", attempt)
                elif status in (500, 502, 503):
                    self._retry_wait("server_error", attempt)
                else:
                    logger.error(f"   HTTP {status}: {e}")
                    return None
            except requests.exceptions.Timeout:
                self._retry_wait("server_error", attempt)
            except Exception as e:
                logger.error(f"  Custom API Error (attempt {attempt + 1}): {e}")
                time.sleep(1)
        return None

    def _read_and_encode(self, image_path: str) -> str | None:
        """Read image file and return base64 string."""
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except (FileNotFoundError, PermissionError) as e:
            logger.error(f"   Cannot read image {image_path}: {e}")
            return None

    @staticmethod
    def parse_coordinates(text: str) -> tuple[float | None, float | None]:
        """Parse GPS coordinates from model output."""
        if not text:
            return None, None

        # 1. JSON format
        try:
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                if "latitude" in data and "longitude" in data:
                    lat = float(data["latitude"])
                    lon = float(data["longitude"])
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        return lat, lon
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # 2. Tuple format: (lat, lon)
        match = re.search(r"[\(\[]\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*[\)\]]", text)
        if match:
            try:
                lat, lon = float(match.group(1)), float(match.group(2))
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return lat, lon
            except ValueError:
                pass

        # 3. Labeled format
        lat_match = re.search(r"[Ll]at(?:itude)?[:\s]+(-?\d+\.?\d*)", text)
        lon_match = re.search(r"[Ll]on(?:gitude)?[:\s]+(-?\d+\.?\d*)", text)
        if lat_match and lon_match:
            try:
                lat = float(lat_match.group(1))
                lon = float(lon_match.group(1))
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return lat, lon
            except ValueError:
                pass

        # 4. Plain two-number
        nums = re.findall(r"-?\d{1,3}\.\d{2,}", text)
        if len(nums) >= 2:
            try:
                lat_v, lon_v = float(nums[0]), float(nums[1])
                if -90 <= lat_v <= 90 and -180 <= lon_v <= 180:
                    return lat_v, lon_v
            except ValueError:
                pass

        return None, None


# ===========================================================================
#  Factory Function
# ===========================================================================


def build_client(
    model_short_name: str,
    provider: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    **kwargs,
) -> GeoLocalizationClient:
    """Build a GeoLocalizationClient from a model short name."""

    if model_short_name.endswith("-vertex"):
        raise ValueError(
            "Legacy Vertex model aliases are no longer supported. "
            "Use supported short names such as 'gemini-2.5-flash' or 'gemini-3.1-pro'."
        )

    # Look up registry
    if model_short_name in MODEL_REGISTRY:
        entry = MODEL_REGISTRY[model_short_name]
        resolved_model = entry["model"]
        resolved_provider = provider or entry["provider"]
        is_thinking = entry.get("thinking", False)
        freq_penalty = entry.get("freq_penalty", True)
        default_max_tokens = entry.get("max_tokens", 8192)
        registry_prompt = entry.get("prompt", None)
        registry_skip_image = entry.get("skip_image", False)
    else:
        resolved_model = model_short_name
        resolved_provider = provider or "sponsor"
        is_thinking = kwargs.pop("is_thinking_model", False)
        freq_penalty = kwargs.pop("freq_penalty", True)
        default_max_tokens = kwargs.pop("max_tokens", 8192)
        registry_prompt = None
        registry_skip_image = False

    # Get provider config
    if resolved_provider not in PROVIDER_CONFIGS:
        raise ValueError(
            f"Unknown provider '{resolved_provider}'. Available providers: {', '.join(sorted(PROVIDER_CONFIGS))}."
        )
    provider_cfg = PROVIDER_CONFIGS[resolved_provider]
    resolved_api_base = api_base or provider_cfg.get("api_base", "")

    # Resolve API key: explicit arg > provider config > env var
    resolved_api_key = api_key or provider_cfg.get("api_key", "")
    if not resolved_api_key:
        env_var = f"{resolved_provider.upper()}_API_KEY"
        resolved_api_key = os.environ.get(env_var, "")
    if not resolved_api_key:
        raise ValueError(
            f"API key required for provider '{resolved_provider}'. "
            f"Set env var {resolved_provider.upper()}_API_KEY or pass api_key=."
        )

    resolved_timeout = kwargs.pop(
        "timeout",
        DEFAULT_PROVIDER_TIMEOUTS.get(resolved_provider, 120),
    )
    resolved_max_tokens = kwargs.pop("max_tokens", default_max_tokens)
    resolved_max_retries = kwargs.pop(
        "max_retries",
        DEFAULT_PROVIDER_MAX_RETRIES.get(resolved_provider, 5),
    )

    client = GeoLocalizationClient(
        model_name=resolved_model,
        api_base=resolved_api_base,
        api_key=resolved_api_key,
        provider=resolved_provider,
        is_thinking_model=is_thinking,
        max_tokens=resolved_max_tokens,
        timeout=resolved_timeout,
        max_retries=resolved_max_retries,
        supports_frequency_penalty=freq_penalty,
        custom_prompt=registry_prompt,
        skip_image=registry_skip_image,
    )

    return client


def list_models(provider: str | None = None, tier: str | None = None) -> None:
    """Print available models. Filter by provider or tier ('main'/'secondary')."""
    print(f"{'Short Name':<28} {'Tier':<12} {'Model ID':<45} {'Thinking'}")
    print("-" * 95)
    for name, entry in MODEL_REGISTRY.items():
        if provider and entry["provider"] != provider:
            continue
        if tier and entry.get("tier") != tier:
            continue
        thinking = "Y" if entry.get("thinking") else ""
        t = entry.get("tier", "")
        print(f"{name:<28} {t:<12} {entry['model']:<45} {thinking}")


if __name__ == "__main__":
    list_models()
