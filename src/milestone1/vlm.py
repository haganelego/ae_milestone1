"""VLM (Vision-Language Model) client for saturation scoring via ollama /api/chat with images."""

import base64
import io
import json
from typing import Any

import requests
from PIL import Image

from milestone1.llm import _parse_llm_json


VLM_SINGLE_PROMPT = """You are a visual scene analysis expert. You are shown a single image captured at a certain observation quality level (resolution).

Your task: Judge whether the scene has reached "semantic saturation" — i.e., whether getting a sharper or closer view of this scene is unlikely to reveal important new objects or spatial relationships that are not already visible.

Consider:
1. Can you clearly identify the main objects and their arrangement?
2. Are there blurry or indistinct regions that might contain important objects?
3. Is the image detailed enough to understand the key relationships between objects?
4. Would higher resolution likely reveal new semantic content (new object types, new relationships)?

OUTPUT: Respond with ONLY a JSON object (no markdown, no explanation outside JSON):
{
  "saturation_score": <float 0.0-1.0, where 1.0 = fully saturated, no benefit from higher resolution>,
  "stop_decision": <boolean, true if saturation_score >= 0.8>,
  "reason": "<brief explanation of your judgment>"
}
"""

VLM_SEQUENCE_PROMPT = """You are a visual scene analysis expert. You are shown a sequence of images of the SAME scene at gradually improving observation quality (resolution). The images go from lower to higher resolution (oldest to newest).

Your task: Judge whether the scene has reached "semantic saturation" — i.e., whether further improving image quality is unlikely to reveal meaningful new objects or spatial relationships.

Consider:
1. Comparing consecutive images: are new objects or details becoming visible with each quality improvement?
2. Has the rate of new discoveries slowed down or stopped?
3. Are the most recent images showing essentially the same semantic content?
4. Would even higher resolution likely reveal important new content?

OUTPUT: Respond with ONLY a JSON object (no markdown, no explanation outside JSON):
{
  "saturation_score": <float 0.0-1.0, where 1.0 = fully saturated, no benefit from higher resolution>,
  "stop_decision": <boolean, true if saturation_score >= 0.8>,
  "reason": "<brief explanation of your judgment>"
}
"""


def _load_and_downscale_image(
    image_path: str, max_side: int = 512
) -> str:
    """Load an image, downscale so longest side <= max_side, return base64 JPEG."""
    img = Image.open(image_path)
    # Downscale if needed
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    # Convert to RGB if needed (e.g., RGBA, P mode)
    if img.mode != "RGB":
        img = img.convert("RGB")
    # Encode as JPEG base64
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _do_vlm_request(
    payload: dict,
    chat_endpoint: str,
    socket_timeout: int,
) -> str:
    """Make the actual VLM API call (runs in a thread for wall-clock timeout)."""
    resp = requests.post(chat_endpoint, json=payload, timeout=socket_timeout)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def call_vlm(
    prompt: str,
    images_b64: list[str],
    model: str,
    chat_endpoint: str,
    timeout: int = 300,
    num_ctx: int | None = None,
) -> str:
    """Call Ollama chat API with images and return the response text.

    ``num_ctx`` (optional) only sizes the context window; the short image prompts
    never approach ollama's default, so outputs are unaffected. Matching another
    concurrent client's num_ctx avoids model reloads.

    Uses a ThreadPoolExecutor with wall-clock timeout to handle cases
    where ollama keeps the connection alive with slow token generation.
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": images_b64,
            },
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 1024,
            **({"num_ctx": num_ctx} if num_ctx else {}),
        },
    }
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_do_vlm_request, payload, chat_endpoint, timeout)
        try:
            return future.result(timeout=timeout + 30)
        except FuturesTimeout:
            raise TimeoutError(f"VLM call exceeded wall-clock timeout of {timeout + 30}s")


def smoke_test_vlm(
    model: str, chat_endpoint: str, timeout: int = 60
) -> bool:
    """Smoke test: send a tiny 8x8 red image and check we get a response."""
    img = Image.new("RGB", (8, 8), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    try:
        result = call_vlm(
            "What color is this image? Respond with just the color name.",
            [b64],
            model,
            chat_endpoint,
            timeout=timeout,
        )
        return len(result.strip()) > 0
    except Exception as e:
        print(f"VLM smoke test failed for {model}: {e}")
        return False


def score_vlm_single(
    image_path: str,
    model: str,
    chat_endpoint: str,
    max_image_side: int = 512,
    timeout: int = 300,
    num_ctx: int | None = None,
) -> dict[str, Any]:
    """Score saturation using a single image (VLM single mode)."""
    try:
        b64 = _load_and_downscale_image(image_path, max_image_side)
        raw = call_vlm(
            VLM_SINGLE_PROMPT, [b64], model, chat_endpoint, timeout, num_ctx=num_ctx
        )
        result = _parse_llm_json(raw)
        return _normalize_result(result)
    except Exception as e:
        return {
            "saturation_score": 0.0,
            "stop_decision": False,
            "reason": f"error: {str(e)[:200]}",
        }


def score_vlm_sequence(
    image_paths: list[str],
    model: str,
    chat_endpoint: str,
    max_image_side: int = 512,
    timeout: int = 300,
) -> dict[str, Any]:
    """Score saturation using a sequence of images (VLM sequence mode)."""
    try:
        b64_list = [
            _load_and_downscale_image(p, max_image_side) for p in image_paths
        ]
        raw = call_vlm(
            VLM_SEQUENCE_PROMPT, b64_list, model, chat_endpoint, timeout
        )
        result = _parse_llm_json(raw)
        return _normalize_result(result)
    except Exception as e:
        return {
            "saturation_score": 0.0,
            "stop_decision": False,
            "reason": f"error: {str(e)[:200]}",
        }


def _normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    """Ensure required fields exist and values are in range."""
    if "saturation_score" not in result:
        result["saturation_score"] = 0.0
    if "stop_decision" not in result:
        result["stop_decision"] = result["saturation_score"] >= 0.8
    if "reason" not in result:
        result["reason"] = "no reason provided"
    result["saturation_score"] = max(0.0, min(1.0, float(result["saturation_score"])))
    result["stop_decision"] = bool(result["stop_decision"])
    return result
