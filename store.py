"""Results store: every inspection and every operator verdict, kept for audit.

SQLite is enough for a line: a few hundred rows a minute, one file, no server.
The same schema maps straight onto Postgres when the plant wants it central.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    coil_id TEXT,
    frame INTEGER,
    position_m REAL,
    decision TEXT NOT NULL,
    anomaly_score REAL,
    n_detections INTEGER,
    max_conf REAL,
    classes TEXT,
    detections TEXT,
    reason TEXT,
    image_path TEXT,
    total_ms REAL
);
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id INTEGER NOT NULL REFERENCES inspections(id),
    ts REAL NOT NULL,
    operator TEXT,
    verdict TEXT NOT NULL,          -- CONFIRM | REJECT | RELABEL
    corrected_class TEXT,
    note TEXT
);
CREATE INDEX IF NOT EXISTS idx_insp_decision ON inspections(decision);
CREATE INDEX IF NOT EXISTS idx_insp_coil ON inspections(coil_id);
"""


class ResultStore:
    def __init__(self, path: str | Path = "data/steelsight.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def log_inspection(self, record: dict, coil_id: Optional[str] = None, frame: Optional[int] = None,
                       position_m: Optional[float] = None, image_path: Optional[str] = None) -> int:
        cur = self.conn.execute(
            """INSERT INTO inspections (ts, coil_id, frame, position_m, decision, anomaly_score, n_detections,
                                        max_conf, classes, detections, reason, image_path, total_ms)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                time.time(), coil_id, frame, position_m, record["decision"], record.get("anomaly_score"),
                record.get("n_detections", 0), record.get("max_conf", 0.0), json.dumps(record.get("classes", [])),
                json.dumps(record.get("detections", [])), record.get("reason"), image_path,
                (record.get("timings_ms") or {}).get("total_ms"),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def log_review(self, inspection_id: int, verdict: str, operator: str = "operator",
                   corrected_class: Optional[str] = None, note: Optional[str] = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO reviews (inspection_id, ts, operator, verdict, corrected_class, note) VALUES (?,?,?,?,?,?)",
            (inspection_id, time.time(), operator, verdict, corrected_class, note),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def recent(self, limit: int = 200, decision: Optional[str] = None, coil_id: Optional[str] = None) -> list[dict]:
        q, args = "SELECT * FROM inspections", []
        where = []
        if decision:
            where.append("decision = ?"); args.append(decision)
        if coil_id:
            where.append("coil_id = ?"); args.append(coil_id)
        if where:
            q += " WHERE " + " AND ".join(where)
        q += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        return [dict(r) for r in self.conn.execute(q, args)]

    def pending_reviews(self, limit: int = 100) -> list[dict]:
        q = """SELECT i.* FROM inspections i
               LEFT JOIN reviews r ON r.inspection_id = i.id
               WHERE i.decision IN ('REVIEW','UNKNOWN') AND r.id IS NULL
               ORDER BY i.id DESC LIMIT ?"""
        return [dict(r) for r in self.conn.execute(q, (limit,))]

    def review_stats(self) -> dict:
        rows = self.conn.execute("SELECT verdict, COUNT(*) AS n FROM reviews GROUP BY verdict").fetchall()
        stats = {r["verdict"]: r["n"] for r in rows}
        total = sum(stats.values())
        confirmed = stats.get("CONFIRM", 0) + stats.get("RELABEL", 0)
        stats["precision_on_reviewed"] = round(confirmed / total, 3) if total else None
        return stats

    def coils(self) -> list[str]:
        return [r["coil_id"] for r in self.conn.execute(
            "SELECT DISTINCT coil_id FROM inspections WHERE coil_id IS NOT NULL ORDER BY coil_id")]

    def export_training_labels(self, out_dir: str | Path, class_names: list[str], image_hw: tuple[int, int]) -> int:
        """Confirmed / relabelled reviews -> YOLO label files for the next fine-tune.

        This is the active-learning loop: every operator verdict becomes a
        training example, so the model improves on exactly the cases it got wrong.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        name_to_id = {n: i for i, n in enumerate(class_names)}
        rows = self.conn.execute(
            """SELECT i.id, i.detections, i.image_path, r.verdict, r.corrected_class
               FROM inspections i JOIN reviews r ON r.inspection_id = i.id
               WHERE r.verdict IN ('CONFIRM','RELABEL') AND i.image_path IS NOT NULL"""
        ).fetchall()
        h, w = image_hw
        n = 0
        for r in rows:
            dets = json.loads(r["detections"] or "[]")
            lines = []
            for d in dets:
                name = r["corrected_class"] if r["verdict"] == "RELABEL" and r["corrected_class"] else d["name"]
                if name not in name_to_id:
                    continue
                x1, y1, x2, y2 = d["xyxy"]
                lines.append(f"{name_to_id[name]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            if lines:
                (out / (Path(r["image_path"]).stem + ".txt")).write_text("\n".join(lines))
                n += 1
        return n
