import logging
import httpx
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

from settings import settings
from db import opensearch as os_client
from services.embedder import embed
from services.reranker import rerank

log = logging.getLogger(__name__)


# ── State ──────────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    query: str
    candidates: list[dict]
    reranked: list[dict]
    prompt: str
    response: str
    source_page_ids: list[int]


# ── Nodes ──────────────────────────────────────────────────────────────────

async def retrieve_node(state: RAGState) -> RAGState:
    query_vec = embed(state["query"])
    hits = await os_client.hybrid_search(
        query_text=state["query"],
        embedding=query_vec,
        top_k=settings.top_k_retrieve,
        knn_weight=0.7,
    )
    log.info("[retrieve] found %d candidates", len(hits))
    state["candidates"] = hits
    return state


def _route_after_retrieve(state: RAGState) -> Literal["rerank", "no_results"]:
    return "rerank" if state["candidates"] else "no_results"


async def rerank_node(state: RAGState) -> RAGState:
    passages = [c["text"] for c in state["candidates"]]
    ranked = await rerank(query=state["query"], passages=passages)

    text_to_meta = {c["text"]: c for c in state["candidates"]}
    reranked = []
    for item in ranked[: settings.top_k_rerank]:
        meta = text_to_meta.get(item["passage"], {})
        reranked.append({**meta, "score": item["score"]})

    log.info("[rerank] kept %d / %d", len(reranked), len(state["candidates"]))
    state["reranked"] = reranked
    state["source_page_ids"] = [r["page_id"] for r in reranked if "page_id" in r]
    return state


def build_prompt_node(state: RAGState) -> RAGState:
    context_blocks = [
        f"[Page {r.get('page_number')} — {r.get('filename')}]\n{r.get('text', '')}"
        for r in state["reranked"]
    ]
    context = "\n\n---\n\n".join(context_blocks)

    state["prompt"] = (
        "You are a precise document assistant. "
        "Answer the question using ONLY the context below. "
        "If the answer is not present, say: 'I could not find this in the provided documents.'\n\n"
        f"### Context\n{context}\n\n"
        f"### Question\n{state['query']}\n\n"
        "### Answer"
    )
    return state


async def generate_node(state: RAGState) -> RAGState:
    payload = {
        "model": settings.llm_model,
        "messages": [{"role": "user", "content": state["prompt"]}],
        "temperature": 0.2,
        "max_tokens": 1024,
    }
    async with httpx.AsyncClient(timeout=120.0) as http:
        resp = await http.post(
            f"{settings.llm_url}/v1/chat/completions", json=payload
        )
        resp.raise_for_status()

    state["response"] = resp.json()["choices"][0]["message"]["content"]
    return state


def no_results_node(state: RAGState) -> RAGState:
    state["response"] = (
        "No relevant pages were found in the indexed documents for your query. "
        "Please make sure documents have been ingested and try rephrasing."
    )
    state["reranked"] = []
    state["source_page_ids"] = []
    return state


# ── Graph assembly ─────────────────────────────────────────────────────────

def build_rag_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve",     retrieve_node)
    graph.add_node("rerank",       rerank_node)
    graph.add_node("build_prompt", build_prompt_node)
    graph.add_node("generate",     generate_node)
    graph.add_node("no_results",   no_results_node)

    graph.set_entry_point("retrieve")

    # conditional edge: skip rerank/generate if nothing was retrieved
    graph.add_conditional_edges(
        "retrieve",
        _route_after_retrieve,
        {"rerank": "rerank", "no_results": "no_results"},
    )
    graph.add_edge("rerank",       "build_prompt")
    graph.add_edge("build_prompt", "generate")
    graph.add_edge("generate",     END)
    graph.add_edge("no_results",   END)

    return graph.compile()


rag_graph = build_rag_graph()


async def run_rag(query: str) -> dict:
    initial: RAGState = {
        "query": query,
        "candidates": [],
        "reranked": [],
        "prompt": "",
        "response": "",
        "source_page_ids": [],
    }
    final = await rag_graph.ainvoke(initial)

    return {
        "response": final["response"],
        "source_page_ids": final["source_page_ids"],
        "sources": [
            {
                "page_number": r.get("page_number"),
                "filename":    r.get("filename"),
                "score":       round(r.get("score", 0.0), 4),
                "excerpt":     r.get("text", "")[:300],
            }
            for r in final["reranked"]
        ],
    }
