import importlib.util
import sqlite3
import tempfile
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]


def load_db_module():
    spec = importlib.util.spec_from_file_location("db_under_test", BACKEND_DIR / "db.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class DatabaseSourceFieldTests(unittest.TestCase):
    def test_init_adds_notebook_owner_column_without_losing_legacy_data(self):
        db = load_db_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            db.DB_PATH = str(Path(temp_dir) / "notebooks.db")
            conn = sqlite3.connect(db.DB_PATH)
            conn.execute(
                "CREATE TABLE notebooks (id TEXT PRIMARY KEY, name TEXT, icon TEXT, updated_at TEXT)"
            )
            conn.execute(
                "INSERT INTO notebooks (id, name, icon, updated_at) VALUES ('legacy', '舊筆記本', '📓', 'now')"
            )
            conn.commit()
            conn.close()

            db.init_db()

            conn = sqlite3.connect(db.DB_PATH)
            columns = {row[1] for row in conn.execute("PRAGMA table_info(notebooks)").fetchall()}
            row = conn.execute("SELECT id, name, user_id FROM notebooks WHERE id = 'legacy'").fetchone()
            conn.close()

        self.assertIn("user_id", columns)
        self.assertEqual(row, ("legacy", "舊筆記本", None))

    def test_init_migrates_existing_sources_as_audio(self):
        db = load_db_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            db.DB_PATH = str(Path(temp_dir) / "notebooks.db")
            conn = sqlite3.connect(db.DB_PATH)
            conn.execute(
                """
                CREATE TABLE sources (
                    id TEXT PRIMARY KEY,
                    notebook_id TEXT,
                    filename TEXT,
                    added_at TEXT
                )
                """
            )
            conn.execute(
                "INSERT INTO sources (id, notebook_id, filename, added_at) VALUES ('s1', 'n1', 'old.wav', 'now')"
            )
            conn.commit()
            conn.close()

            db.init_db()

            conn = sqlite3.connect(db.DB_PATH)
            row = conn.execute(
                "SELECT source_type, mime_type, content_segments_json FROM sources WHERE id='s1'"
            ).fetchone()
            conn.close()
        self.assertEqual(row[0], "audio")
        self.assertIsNone(row[1])
        self.assertIsNone(row[2])

    def test_document_source_round_trip(self):
        db = load_db_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            db.DB_PATH = str(Path(temp_dir) / "notebooks.db")
            db.init_db()
            notebook = db.create_notebook_record()
            db.insert_source_record(
                notebook["id"],
                "source-id",
                "notes.md",
                "文件內容",
                "[]",
                "quick",
                "pending",
                None,
                source_type="document",
                mime_type="text/markdown",
                content_segments_json='[{"segment_id":"lines-1-1","text":"文件內容"}]',
            )

            details = db.get_notebook_details_data(notebook["id"])
            content = db.get_source_content_data(notebook["id"], "source-id")

        self.assertEqual(details["sources"][0]["source_type"], "document")
        self.assertTrue(details["sources"][0]["has_content"])
        self.assertEqual(content["segments"][0]["segment_id"], "lines-1-1")


if __name__ == "__main__":
    unittest.main()
