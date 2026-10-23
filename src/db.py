"""Shared paths and database helpers. Every other script imports from here."""
import sqlite3
from pathlib import Path

from sqlalchemy import create_engine, event

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "db" / "ufc.db"
SCHEMA_PATH = ROOT / "sql" / "schema.sql"
RAW_DIR = ROOT / "data" / "raw"

# Children first, so foreign keys never block the drops.
DROP_SQL = """
DROP TABLE IF EXISTS fight_prefight_stats;
DROP TABLE IF EXISTS fight_rounds;
DROP TABLE IF EXISTS fights;
DROP TABLE IF EXISTS events;
DROP TABLE IF EXISTS fighters;
"""


def get_engine():
    """SQLAlchemy engine for pandas. Foreign keys are switched on for every connection."""
    DB_PATH.parent.mkdir(exist_ok=True)
    engine = create_engine(f"sqlite:///{DB_PATH.as_posix()}")

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_conn, _record):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON")
        cur.close()

    return engine


def init_schema(drop_first=False):
    """Create the tables from sql/schema.sql. drop_first=True wipes and rebuilds them."""
    DB_PATH.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    try:
        if drop_first:
            con.executescript(DROP_SQL)
        con.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        con.commit()
    finally:
        con.close()