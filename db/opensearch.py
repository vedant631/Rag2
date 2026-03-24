from opensearchpy import AsyncOpenSearch, NotFoundError
from settings import settings

client = AsyncOpenSearch(
    hosts=[settings.opensearch_url],
    use_ssl=False,
    verify_certs=False,
    http_compress=True,
)

INDEX_BODY = {
    "settings": {
        "index": {
            "knn": True,
            "knn.algo_param.ef_search": 100,
        }
    },
    "mappings": {
        "properties": {
            "page_id":     {"type": "integer"},
            "document_id": {"type": "integer"},
            "page_number": {"type": "integer"},
            "filename":    {"type": "keyword"},
            "text":        {"type": "text", "analyzer": "standard"},
            "embedding": {
                "type":      "knn_vector",
                "dimension": settings.opensearch_dims,
                "method": {
                    "name":       "hnsw",
                    "space_type": "cosinesimil",
                    "engine":     "nmslib",
                    "parameters": {"ef_construction": 128, "m": 16},
                },
            },
        }
    },
}

_SOURCE_FIELDS = ["page_id", "document_id", "page_number", "filename", "text"]


async def ensure_index():
    exists = await client.indices.exists(index=settings.opensearch_index)
    if not exists:
        await client.indices.create(index=settings.opensearch_index, body=INDEX_BODY)
        print(f"[opensearch] created index '{settings.opensearch_index}'")
    else:
        print(f"[opensearch] index '{settings.opensearch_index}' already exists")


async def upsert_page(doc_id: str, body: dict) -> None:
    await client.index(
        index=settings.opensearch_index,
        id=doc_id,
        body=body,
        refresh=True,
    )


async def knn_search(embedding: list[float], top_k: int) -> list[dict]:
    """Pure vector KNN search."""
    query = {
        "size": top_k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": embedding,
                    "k": top_k,
                }
            }
        },
        "_source": _SOURCE_FIELDS,
    }
    resp = await client.search(index=settings.opensearch_index, body=query)
    return [hit["_source"] for hit in resp["hits"]["hits"]]



async def hybrid_search(
    query_text: str,
    embedding: list[float],
    top_k: int,
    knn_weight: float = 0.7,
) -> list[dict]:
    """
    Hybrid BM25 + KNN search using score-level fusion.
    Runs both queries, normalises scores to [0,1], fuses with weights, deduplicates.
    knn_weight: how much to weight vector score vs BM25 (0.7 means 70% vector).
    """
    bm25_weight = 1.0 - knn_weight

    # --- run both queries concurrently ---
    import asyncio

    knn_query = {
        "size": top_k * 2,
        "query": {"knn": {"embedding": {"vector": embedding, "k": top_k * 2}}},
        "_source": _SOURCE_FIELDS,
    }
    bm25_query = {
        "size": top_k * 2,
        "query": {"match": {"text": {"query": query_text}}},
        "_source": _SOURCE_FIELDS,
    }

    knn_resp, bm25_resp = await asyncio.gather(
        client.search(index=settings.opensearch_index, body=knn_query),
        client.search(index=settings.opensearch_index, body=bm25_query),
    )

    knn_hits  = knn_resp["hits"]["hits"]
    bm25_hits = bm25_resp["hits"]["hits"]

    # --- normalise scores ---
    def normalise(hits: list[dict]) -> dict[str, float]:
        if not hits:
            return {}
        scores = [h["_score"] for h in hits]
        min_s, max_s = min(scores), max(scores)
        rng = max_s - min_s or 1.0
        return {h["_id"]: (h["_score"] - min_s) / rng for h in hits}

    knn_scores  = normalise(knn_hits)
    bm25_scores = normalise(bm25_hits)

    # --- fuse ---
    all_ids = set(knn_scores) | set(bm25_scores)
    fused: dict[str, float] = {
        doc_id: knn_weight * knn_scores.get(doc_id, 0.0)
                + bm25_weight * bm25_scores.get(doc_id, 0.0)
        for doc_id in all_ids
    }

    # --- build result list sorted by fused score ---
    id_to_source: dict[str, dict] = {
        h["_id"]: h["_source"] for h in knn_hits + bm25_hits
    }
    results = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [id_to_source[doc_id] for doc_id, _ in results if doc_id in id_to_source]


async def delete_document_pages(opensearch_ids: list[str]) -> None:
    """Remove all pages for a document from the index."""
    if not opensearch_ids:
        return
    await client.delete_by_query(
        index=settings.opensearch_index,
        body={"query": {"ids": {"values": opensearch_ids}}},
    )