"""
Ingestion pipeline:
  PDF bytes → split pages → image per page → vision model → normalize →
  save to Postgres → embed → upsert to OpenSearch

Per-page errors are caught and logged; a failed page is saved with error info
so the rest of the document still ingests. The caller receives a summary.
"""
import io
import uuid
import logging

from pdf2image import convert_from_bytes
from sqlalchemy.ext.asyncio import AsyncSession

from db.postgres import Document, Page
from db import opensearch as os_client
from services.vision import parse_page_image
from services.embedder import embed
from pipe.normalize import page_json_to_text

log = logging.getLogger(__name__)


def _page_to_png(pil_image) -> bytes:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue()


async def ingest_document(
    filename: str,
    pdf_bytes: bytes,
    session: AsyncSession,
) -> dict:
    """
    Ingest a PDF. Returns:
      {
        "document_id": int,
        "total_pages": int,
        "ingested": int,      # pages that succeeded
        "failed": [int],      # page numbers that errored
      }
    """
    images = convert_from_bytes(pdf_bytes, dpi=150)
    total = len(images)

    doc = Document(filename=filename, total_pages=total)
    session.add(doc)
    await session.flush()

    ingested, failed = 0, []

    for page_num, pil_img in enumerate(images, start=1):
        try:
            png_bytes = _page_to_png(pil_img)

            # Vision model
            raw_json = await parse_page_image(png_bytes)

            normalized_text = page_json_to_text(raw_json)
            if not normalized_text.strip():
                normalized_text = f"[Page {page_num} — no extractable text]"

            os_doc_id = str(uuid.uuid4())

            page = Page(
                document_id=doc.id,
                page_number=page_num,
                raw_json=raw_json,
                normalized_text=normalized_text,
                opensearch_id=os_doc_id,
            )
            session.add(page)
            await session.flush()

            # Embed + index
            vector = embed(normalized_text)
            await os_client.upsert_page(
                doc_id=os_doc_id,
                body={
                    "page_id":     page.id,
                    "document_id": doc.id,
                    "page_number": page_num,
                    "filename":    filename,
                    "text":        normalized_text,
                    "embedding":   vector,
                },
            )

            ingested += 1
            log.info("[ingest] %s page %d/%d ✓", filename, page_num, total)

        except Exception as exc:
            failed.append(page_num)
            log.error(
                "[ingest] %s page %d/%d FAILED: %s",
                filename, page_num, total, exc, exc_info=True,
            )
            # Save a placeholder so the page record still exists
            session.add(Page(
                document_id=doc.id,
                page_number=page_num,
                raw_json={"error": str(exc)},
                normalized_text=f"[Page {page_num} ingestion failed: {exc}]",
                opensearch_id=None,
            ))
            await session.flush()
            continue

    await session.commit()

    return {
        "document_id": doc.id,
        "total_pages": total,
        "ingested": ingested,
        "failed": failed,
    }