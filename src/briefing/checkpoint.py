"""SQLite checkpointer for durable research-graph threads."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

DEFAULT_CHECKPOINT_PATH = Path("data/checkpoints.sqlite")


def open_sqlite_checkpointer(
    path: Path | str = DEFAULT_CHECKPOINT_PATH,
) -> SqliteSaver:
    """Open a SqliteSaver. Caller owns the process lifetime of the connection."""
    if str(path) == ":memory:":
        conn = sqlite3.connect(":memory:", check_same_thread=False)
    else:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(file_path, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver
