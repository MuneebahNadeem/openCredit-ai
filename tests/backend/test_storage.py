"""Tests for JSON-file investigation storage."""

from __future__ import annotations

import json

from backend.app.services.storage import InvestigationStorage


def _record(i: str) -> dict:
    return {
        "id": i,
        "created_at": "2026-08-31T00:00:00+00:00",
        "status": "completed",
        "business": {"name": f"Business {i}"},
        "result": {"trustworthiness": {"level": "moderate"}},
    }


class TestStorage:
    def test_save_and_get_roundtrip(self, tmp_path):
        storage = InvestigationStorage(str(tmp_path / "inv"))
        storage.save(_record("inv_a"))
        loaded = storage.get("inv_a")
        assert loaded == _record("inv_a")

    def test_get_missing_returns_none(self, tmp_path):
        storage = InvestigationStorage(str(tmp_path / "inv"))
        assert storage.get("inv_none") is None

    def test_overwrite_on_same_id(self, tmp_path):
        storage = InvestigationStorage(str(tmp_path / "inv"))
        storage.save(_record("inv_a"))
        updated = _record("inv_a")
        updated["status"] = "failed"
        storage.save(updated)
        assert storage.get("inv_a")["status"] == "failed"

    def test_loads_existing_files_on_startup(self, tmp_path):
        dir_path = tmp_path / "inv"
        dir_path.mkdir()
        (dir_path / "inv_old.json").write_text(
            json.dumps(_record("inv_old")), encoding="utf-8"
        )
        storage = InvestigationStorage(str(dir_path))
        assert storage.get("inv_old")["id"] == "inv_old"

    def test_list_all_returns_records(self, tmp_path):
        storage = InvestigationStorage(str(tmp_path / "inv"))
        storage.save(_record("inv_a"))
        storage.save(_record("inv_b"))
        ids = {r["id"] for r in storage.list_all()}
        assert ids == {"inv_a", "inv_b"}

    def test_corrupt_file_is_skipped(self, tmp_path):
        dir_path = tmp_path / "inv"
        dir_path.mkdir()
        (dir_path / "inv_bad.json").write_text("{not valid json", encoding="utf-8")
        storage = InvestigationStorage(str(dir_path))
        assert storage.get("inv_bad") is None
        assert storage.list_all() == []

    def test_creates_directory_if_missing(self, tmp_path):
        storage = InvestigationStorage(str(tmp_path / "nested" / "inv"))
        storage.save(_record("inv_a"))
        assert storage.get("inv_a")["id"] == "inv_a"
