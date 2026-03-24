"""
Reranker service — runs as a SEPARATE FastAPI process on port 5601.
Uses Qwen/Qwen3-Reranker-0.6B (cross-encoder).

Run this file directly on the server:
  uvicorn services.reranker_server:app --port 5601 --host 0.0.0.0
"""

RERANKER_SERVER = '''

'''

# ── Part 2: Reranker client (used inside main pipeline) ────────────────────

import httpx
from settings import settings


async def rerank(query: str, passages: list[str]) -> list[dict]:
    """
    Returns passages sorted by relevance score (highest first).
    Each item: {"passage": str, "score": float}
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{settings.reranker_url}/rerank",
            json={"query": query, "passages": passages},
        )
        resp.raise_for_status()
    return resp.json()["results"]