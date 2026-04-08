"""
OpenEnv-Orbital-Command | scoring/leaderboard.py
SQLite-backed leaderboard for tracking and comparing episode results.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import sys
from pathlib import Path
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from env.models import EpisodeResult, LeaderboardEntry

DB_PATH = Path(__file__).parent.parent / "data" / "leaderboard.db"


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS episodes (
            episode_id     TEXT PRIMARY KEY,
            agent_name     TEXT NOT NULL,
            model_name     TEXT NOT NULL,
            task_id        INTEGER NOT NULL,
            task_name      TEXT NOT NULL,
            normalized_score REAL NOT NULL,
            final_score    REAL NOT NULL,
            grade          TEXT NOT NULL,
            data_dl_gb     REAL,
            data_ow_gb     REAL,
            sats_survived  INTEGER,
            total_sats     INTEGER,
            req_fulfilled  INTEGER,
            req_missed     INTEGER,
            total_steps    INTEGER,
            duration_sec   REAL,
            timestamp      TEXT NOT NULL,
            grader_json    TEXT,
            reward_json    TEXT
        )
    """)
    conn.commit()
    return conn


def submit_result(
    result: EpisodeResult,
    agent_name: str = "Unknown",
    model_name: str = "RuleBased",
) -> str:
    """Save an episode result to the leaderboard. Returns episode_id."""
    episode_id = str(uuid.uuid4())
    conn = _get_conn()
    conn.execute("""
        INSERT INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        episode_id,
        agent_name,
        model_name,
        result.task_id,
        result.task_name,
        result.normalized_score,
        result.final_score,
        result.grade,
        result.data_downlinked_gb,
        result.data_overwritten_gb,
        result.satellites_survived,
        result.total_satellites,
        result.requests_fulfilled,
        result.requests_missed,
        result.total_steps,
        result.duration_seconds,
        result.timestamp,
        json.dumps(result.grader_breakdown),
        json.dumps(result.reward_history[-50:]),  # last 50 steps
    ))
    conn.commit()
    conn.close()
    return episode_id


def get_leaderboard(
    task_id: Optional[int] = None,
    top_n: int = 20,
) -> List[LeaderboardEntry]:
    """Retrieve top N entries, optionally filtered by task."""
    conn = _get_conn()
    where = f"WHERE task_id = {task_id}" if task_id else ""
    rows = conn.execute(f"""
        SELECT agent_name, model_name, task_id, normalized_score, grade, timestamp, episode_id
        FROM episodes
        {where}
        ORDER BY normalized_score DESC
        LIMIT {top_n}
    """).fetchall()
    conn.close()

    return [
        LeaderboardEntry(
            rank=i + 1,
            agent_name=row[0],
            model_name=row[1],
            task_id=row[2],
            normalized_score=row[3],
            grade=row[4],
            timestamp=row[5],
            episode_id=row[6],
        )
        for i, row in enumerate(rows)
    ]


def get_task_stats(task_id: int) -> Dict:
    """Return aggregate statistics for a task."""
    conn = _get_conn()
    row = conn.execute("""
        SELECT
            COUNT(*) as runs,
            AVG(normalized_score) as avg_score,
            MAX(normalized_score) as best_score,
            AVG(data_dl_gb) as avg_dl,
            AVG(total_steps) as avg_steps
        FROM episodes WHERE task_id = ?
    """, (task_id,)).fetchone()
    conn.close()
    if not row or row[0] == 0:
        return {}
    return {
        "total_runs":    row[0],
        "avg_score":     round(row[1] or 0, 4),
        "best_score":    round(row[2] or 0, 4),
        "avg_dl_gb":     round(row[3] or 0, 2),
        "avg_steps":     round(row[4] or 0, 1),
    }


def clear_leaderboard():
    conn = _get_conn()
    conn.execute("DELETE FROM episodes")
    conn.commit()
    conn.close()
