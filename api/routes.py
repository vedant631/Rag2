import logging
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import httpx

from settings import settings
from db.postgres import get_session, QueryLog, Document, Page
from db import opensearch as os_client
from pipe.ingest import ingest_document
from pipe.rag_frame import run_rag

log = logging.getLogger(__name__)
router = APIRouter()


# ── Health ─────────────────────────────────────────────────────────────────

@router.get("/health", summary="Overall system health check")
async def health():
    """
    Checks reachability of: OpenSearch, vision model, reranker.
    Returns per-service status so you can see which service is down.
    """
    status: dict[str, str] = {}

    async with httpx.AsyncClient(timeout=5.0) as http:
        # OpenSearch
        try:
            r = await http.get(f"{settings.opensearch_url}/_cluster/health")
            status["opensearch"] = r.json().get("status", "unknown")
        except Exception as e:
            status["opensearch"] = f"unreachable: {e}"

        # Vision model
        try:
            r = await http.get(f"{settings.vision_url}/health")
            status["vision_model"] = "ok" if r.status_code == 200 else f"http {r.status_code}"
        except Exception as e:
            status["vision_model"] = f"unreachable: {e}"

        # Reranker
        try:
            r = await http.get(f"{settings.reranker_url}/health")
            status["reranker"] = "ok" if r.status_code == 200 else f"http {r.status_code}"
        except Exception as e:
            status["reranker"] = f"unreachable: {e}"

    overall = "ok" if all("ok" in v or "green" in v or "yellow" in v for v in status.values()) else "degraded"
    return {"status": overall, "services": status}


# ── Ingest ─────────────────────────────────────────────────────────────────

@router.post("/ingest", summary="Upload and ingest a PDF document")
async def ingest(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(400, "Uploaded file is empty")

    result = await ingest_document(
        filename=file.filename,
        pdf_bytes=pdf_bytes,
        session=session,
    )

    if result["ingested"] == 0:
        raise HTTPException(
            500,
            f"All {result['total_pages']} pages failed to ingest. "
            f"Check vision model at {settings.vision_url}.",
        )

    return result   # {document_id, total_pages, ingested, failed}


# ── Query ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str


@router.post("/query", summary="Ask a question over ingested documents")
async def query(
    req: QueryRequest,
    session: AsyncSession = Depends(get_session),
):
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty")

    result = await run_rag(req.query)

    log_entry = QueryLog(
        query=req.query,
        retrieved_page_ids=result["source_page_ids"],
        llm_response=result["response"],
    )
    session.add(log_entry)
    await session.commit()

    return result


# ── Documents ──────────────────────────────────────────────────────────────

@router.get("/documents", summary="List all ingested documents")
async def list_documents(session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(Document).order_by(Document.created_at.desc()))
    docs = result.scalars().all()
    return [
        {
            "id": d.id,
            "filename": d.filename,
            "total_pages": d.total_pages,
            "created_at": d.created_at.isoformat(),
        }
        for d in docs
    ]


@router.get("/documents/{doc_id}/pages", summary="List pages for a document")
async def list_pages(doc_id: int, session: AsyncSession = Depends(get_session)):
    result = await session.execute(
        select(Page)
        .where(Page.document_id == doc_id)
        .order_by(Page.page_number)
    )
    pages = result.scalars().all()
    if not pages:
        raise HTTPException(404, f"No pages found for document {doc_id}")
    return [
        {
            "id": p.id,
            "page_number": p.page_number,
            "normalized_text": p.normalized_text,
            "raw_json": p.raw_json,
            "opensearch_id": p.opensearch_id,
        }
        for p in pages
    ]


@router.delete("/documents/{doc_id}", summary="Delete a document and all its pages")
async def delete_document(doc_id: int, session: AsyncSession = Depends(get_session)):
    doc = await session.get(Document, doc_id)
    if not doc:
        raise HTTPException(404, f"Document {doc_id} not found")

    result = await session.execute(select(Page).where(Page.document_id == doc_id))
    pages = result.scalars().all()

    os_ids = [p.opensearch_id for p in pages if p.opensearch_id]
    await os_client.delete_document_pages(os_ids)

    for p in pages:
        await session.delete(p)
    await session.delete(doc)
    await session.commit()

    return {"deleted": doc_id, "pages_removed": len(pages)}


# ── Query history ──────────────────────────────────────────────────────────

@router.get("/queries", summary="Recent query history")
async def query_history(
    limit: int =30,
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(QueryLog)
        .order_by(QueryLog.created_at.desc())
        .limit(limit)
    )
    logs = result.scalars().all()
    return [
        {
            "id": q.id,
            "query": q.query,
            "llm_response": q.llm_response,
            "retrieved_page_ids": q.retrieved_page_ids,
            "created_at": q.created_at.isoformat(),
        }
        for q in logs
    ]