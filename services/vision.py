"""
Calls llama-server (llama.cpp) running Qwen2.5-VL Q4_K_M on port 5600.
llama-server exposes /v1/chat/completions with vision support.

Start command (on your server):
  llama-server \
    --model qwen2.5-vl-7b-instruct-q4_k_m.gguf \
    --port 5600 \
    --n-gpu-layers 0 \
    --ctx-size 4096 \
    --threads 8
"""

import base64
import json
import re
import httpx

from settings import settings

VISION_PROMPT = """You are a document parser. Analyse this document page image.
Return ONLY a valid JSON object with this structure — no markdown, no explanation:
{
  "title": "section or page title if visible, else null",
  "page_type": "text|table|form|mixed|image",
  "content": {
    "paragraphs": ["..."],
    "tables": [{"headers": ["col1", "col2"], "rows": [["val1", "val2"]]}],
    "key_value_pairs": {"key": "value"},
    "lists": [["item1", "item2"]]
  },
  "language": "en",
  "summary": "one sentence summary of this page"
}"""


def _encode_image(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def _parse_json_response(raw: str) -> dict:
    """Parse JSON from model output, tolerating minor formatting slippage."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # strip markdown fences if model added them
        raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise ValueError(f"Vision model returned unparseable output: {raw[:300]}")


async def parse_page_image(image_bytes: bytes) -> dict:
    b64 = _encode_image(image_bytes)

    payload = {
        "model": "qwen",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {"type": "text", "text": VISION_PROMPT},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
        "response_format": {"type": "json_object"},
    }

    async with httpx.AsyncClient(timeout=120.0) as http:
        resp = await http.post(
            f"{settings.vision_url}/v1/chat/completions",
            json=payload,
        )
        resp.raise_for_status()

    raw = resp.json()["choices"][0]["message"]["content"]
    return _parse_json_response(raw)