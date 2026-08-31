"""In-memory + JSON-file investigation storage.

Deliberately simple for the hackathon: one JSON file per investigation in
``data/investigations/`` so reports survive a backend restart. Isolated
behind this class so a real database can replace it later.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import List, Optional


class InvestigationStorage:
    def __init__(self, directory: str) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._load_existing()

    def _load_existing(self) -> None:
        for path in sorted(self._directory.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                self._records[record["id"]] = record
            except (json.JSONDecodeError, KeyError, OSError):
                continue  # skip corrupt files rather than fail startup

    def save(self, record: dict) -> None:
        with self._lock:
            self._records[record["id"]] = record
            path = self._directory / f"{record['id']}.json"
            path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    def get(self, investigation_id: str) -> Optional[dict]:
        with self._lock:
            return self._records.get(investigation_id)

    def list_all(self) -> List[dict]:
        with self._lock:
            records = list(self._records.values())
        records.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return records
