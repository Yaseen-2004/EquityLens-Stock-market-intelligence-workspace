"""Engine / session plumbing, kept separate so scripts and Flask share it."""
from __future__ import annotations

import os
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

import config
from models import Base

os.makedirs(config.DATA_DIR, exist_ok=True)

engine = create_engine(config.DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_db() -> None:
    """Create tables if they do not exist yet."""
    Base.metadata.create_all(engine)


@contextmanager
def session_scope() -> Session:
    """Transactional scope: commit on success, roll back on error."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
