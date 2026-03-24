from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Text, DateTime,
    ForeignKey, JSON, func
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from settings import settings

Base = declarative_base()
engine = create_async_engine(settings.postgres_dsn, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    total_pages = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    pages = relationship("Page", back_populates="document", cascade="all, delete")


class Page(Base):
    __tablename__ = "pages"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    page_number = Column(Integer, nullable=False)
    raw_json = Column(JSON)           # structured output from vision model
    normalized_text = Column(Text)    # flattened plain text for indexing
    opensearch_id = Column(String)    # the doc id in opensearch
    created_at = Column(DateTime, default=datetime.utcnow)
    document = relationship("Document", back_populates="pages")


class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True)
    query = Column(Text, nullable=False)
    retrieved_page_ids = Column(JSON)   # list of Page.id used as context
    llm_response = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session