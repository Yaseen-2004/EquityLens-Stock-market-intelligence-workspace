"""SQLAlchemy ORM models: a company, and one row per company per trading day."""
from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    BigInteger, Date, DateTime, Float, ForeignKey, Integer,
    LargeBinary, String, Text, UniqueConstraint, func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Company(Base):
    """One listed company we track."""

    __tablename__ = "companies"

    id:     Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    name:   Mapped[str] = mapped_column(String(128))
    currency:   Mapped[str | None] = mapped_column(String(8), default="USD")
    exchange:   Mapped[str | None] = mapped_column(String(32), nullable=True)
    market_cap: Mapped[float | None] = mapped_column(Float, nullable=True)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )

    prices: Mapped[list["PriceHistory"]] = relationship(
        back_populates="company",
        cascade="all, delete-orphan",
        order_by="PriceHistory.date",
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Company {self.symbol}>"


class DocumentChunk(Base):
    """One embedded chunk of generated text about the market data.

    This is the vector store. The source rows live in `price_history`;
    insertdata.py turns them into readable prose, splits that into chunks, and
    stores each chunk beside its embedding so questions can be answered by
    meaning rather than by exact keyword.

    The vector is a float32 numpy array kept as a BLOB - at a few hundred
    chunks an exact scan is instant, so there is no index to keep in sync and
    no second database to run.
    """

    __tablename__ = "document_chunks"
    __table_args__ = (
        # Re-running the pipeline must replace a chunk, not duplicate it.
        UniqueConstraint("doc_id", "chunk_index", name="uq_doc_chunk"),
    )

    id:          Mapped[int] = mapped_column(Integer, primary_key=True)
    doc_id:      Mapped[str] = mapped_column(String(128), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    source:      Mapped[str] = mapped_column(String(32), index=True)
    symbol:      Mapped[str | None] = mapped_column(String(16), nullable=True,
                                                    index=True)
    title:       Mapped[str] = mapped_column(String(256))
    text:        Mapped[str] = mapped_column(Text)
    embedding:   Mapped[bytes] = mapped_column(LargeBinary)
    dim:         Mapped[int] = mapped_column(Integer)
    model:       Mapped[str] = mapped_column(String(128))
    created_at:  Mapped[dt.datetime] = mapped_column(DateTime,
                                                     default=func.now())

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<DocumentChunk {self.doc_id}#{self.chunk_index}>"


class PriceHistory(Base):
    """A single daily OHLCV bar for one company."""

    __tablename__ = "price_history"
    __table_args__ = (
        # The pair (company, date) is the natural key: it makes re-running the
        # ingest idempotent instead of duplicating rows.
        UniqueConstraint("company_id", "date", name="uq_company_date"),
    )

    id:         Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        ForeignKey("companies.id", ondelete="CASCADE"), index=True
    )
    date:   Mapped[dt.date] = mapped_column(Date, index=True)
    open:   Mapped[float | None] = mapped_column(Float)
    high:   Mapped[float | None] = mapped_column(Float)
    low:    Mapped[float | None] = mapped_column(Float)
    close:  Mapped[float] = mapped_column(Float)
    volume: Mapped[int | None] = mapped_column(BigInteger)

    company: Mapped[Company] = relationship(back_populates="prices")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<PriceHistory {self.company_id} {self.date} {self.close}>"
