import json
import sqlite3
import uuid
from datetime import datetime


DB_PATH = "notebooks.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS notebooks (
            id TEXT PRIMARY KEY,
            name TEXT,
            icon TEXT,
            updated_at TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sources (
            id TEXT PRIMARY KEY,
            notebook_id TEXT,
            filename TEXT,
            added_at TEXT,
            transcript_text TEXT,
            timed_segments TEXT,
            transcript_updated_at TEXT,
            indexed_at TEXT,
            FOREIGN KEY(notebook_id) REFERENCES notebooks(id) ON DELETE CASCADE
        )
    ''')
    cursor.execute("PRAGMA table_info(sources)")
    source_columns = [row[1] for row in cursor.fetchall()]
    source_migrations = {
        "transcript_text": "ALTER TABLE sources ADD COLUMN transcript_text TEXT",
        "timed_segments": "ALTER TABLE sources ADD COLUMN timed_segments TEXT",
        "transcript_updated_at": "ALTER TABLE sources ADD COLUMN transcript_updated_at TEXT",
        "indexed_at": "ALTER TABLE sources ADD COLUMN indexed_at TEXT",
    }
    for column_name, migration_sql in source_migrations.items():
        if column_name not in source_columns:
            cursor.execute(migration_sql)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            notebook_id TEXT,
            sender TEXT,
            text TEXT,
            created_at TEXT,
            references_json TEXT,
            FOREIGN KEY(notebook_id) REFERENCES notebooks(id) ON DELETE CASCADE
        )
    ''')
    cursor.execute("PRAGMA table_info(messages)")
    message_columns = [row[1] for row in cursor.fetchall()]
    if "references_json" not in message_columns:
        cursor.execute("ALTER TABLE messages ADD COLUMN references_json TEXT")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS suggested_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            notebook_id TEXT,
            source_filename TEXT,
            question TEXT,
            priority INTEGER,
            used INTEGER DEFAULT 0,
            created_at TEXT,
            FOREIGN KEY(notebook_id) REFERENCES notebooks(id) ON DELETE CASCADE
        )
    ''')
    cursor.execute("PRAGMA table_info(suggested_questions)")
    suggested_question_columns = [row[1] for row in cursor.fetchall()]
    if "used" not in suggested_question_columns:
        cursor.execute("ALTER TABLE suggested_questions ADD COLUMN used INTEGER DEFAULT 0")
    conn.commit()
    conn.close()


def get_notebooks_data():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, icon, updated_at FROM notebooks ORDER BY updated_at DESC")
    notebooks = []
    for row in cursor.fetchall():
        nid = row[0]
        cursor.execute("SELECT COUNT(*) FROM sources WHERE notebook_id=?", (nid,))
        source_count = cursor.fetchone()[0]
        notebooks.append({
            "id": nid,
            "name": row[1],
            "icon": row[2],
            "updated_at": row[3],
            "source_count": source_count
        })
    conn.close()
    return notebooks


def create_notebook_record():
    nid = str(uuid.uuid4())
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    name = "未命名筆記本"
    icon = "📓"

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO notebooks (id, name, icon, updated_at) VALUES (?, ?, ?, ?)",
        (nid, name, icon, now)
    )
    conn.commit()
    conn.close()

    return {"id": nid, "name": name, "icon": icon, "updated_at": now, "source_count": 0}


def update_notebook_name(notebook_id, name):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE notebooks SET name = ?, updated_at = ? WHERE id = ?",
        (name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), notebook_id)
    )
    conn.commit()
    conn.close()


def get_notebook_details_data(notebook_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            id,
            filename,
            added_at,
            transcript_updated_at,
            indexed_at,
            CASE WHEN TRIM(COALESCE(transcript_text, '')) != '' THEN 1 ELSE 0 END
        FROM sources
        WHERE notebook_id = ?
        ORDER BY added_at ASC
        """,
        (notebook_id,)
    )
    sources = [
        {
            "id": r[0],
            "filename": r[1],
            "added_at": r[2],
            "transcript_updated_at": r[3],
            "indexed_at": r[4],
            "has_transcript": bool(r[5])
        }
        for r in cursor.fetchall()
    ]

    cursor.execute(
        "SELECT sender, text, created_at, references_json FROM messages WHERE notebook_id = ? ORDER BY id ASC",
        (notebook_id,)
    )
    messages = []
    for row in cursor.fetchall():
        references = []
        if row[3]:
            try:
                parsed_references = json.loads(row[3])
                if isinstance(parsed_references, list):
                    references = parsed_references
            except Exception:
                references = []
        messages.append({
            "sender": row[0],
            "text": row[1],
            "created_at": row[2],
            "references": references
        })

    cursor.execute(
        """
        SELECT id, question, priority, source_filename, created_at
        FROM suggested_questions
        WHERE notebook_id = ?
          AND used = 0
          AND created_at = (
              SELECT MAX(created_at)
              FROM suggested_questions
              WHERE notebook_id = ?
          )
        ORDER BY priority ASC
        """,
        (notebook_id, notebook_id)
    )
    suggested_questions = [
        {"id": r[0], "question": r[1], "priority": r[2], "source_filename": r[3], "created_at": r[4]}
        for r in cursor.fetchall()
    ]

    conn.close()
    return {"sources": sources, "messages": messages, "suggested_questions": suggested_questions}


def mark_suggested_question_used_record(notebook_id, question_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE suggested_questions SET used = 1 WHERE id = ? AND notebook_id = ?",
        (question_id, notebook_id)
    )
    conn.commit()
    conn.close()


def get_source_transcript_data(notebook_id, source_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, filename, transcript_text, transcript_updated_at, indexed_at
        FROM sources
        WHERE id = ? AND notebook_id = ?
        """,
        (source_id, notebook_id)
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    transcript_text = row[2] or ""
    return {
        "id": row[0],
        "filename": row[1],
        "transcript_text": transcript_text,
        "has_transcript": bool(transcript_text.strip()),
        "transcript_updated_at": row[3],
        "indexed_at": row[4]
    }


def get_source_update_info(notebook_id, source_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT filename, transcript_text, timed_segments
        FROM sources
        WHERE id = ? AND notebook_id = ?
        """,
        (source_id, notebook_id)
    )
    row = cursor.fetchone()
    conn.close()
    return row


def get_source_audio_info(notebook_id, source_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, filename
        FROM sources
        WHERE id = ? AND notebook_id = ?
        """,
        (source_id, notebook_id)
    )
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return {"id": row[0], "filename": row[1]}


def get_source_filenames(notebook_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT filename FROM sources WHERE notebook_id = ?", (notebook_id,))
    filenames = [row[0] for row in cursor.fetchall()]
    conn.close()
    return filenames


def update_source_filename_record(notebook_id, source_id, filename):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT filename
        FROM sources
        WHERE id = ? AND notebook_id = ?
        """,
        (source_id, notebook_id)
    )
    row = cursor.fetchone()
    if not row:
        conn.close()
        return None

    old_filename = row[0]
    cursor.execute(
        """
        UPDATE sources
        SET filename = ?
        WHERE id = ? AND notebook_id = ?
        """,
        (filename, source_id, notebook_id)
    )
    cursor.execute(
        """
        UPDATE suggested_questions
        SET source_filename = ?
        WHERE notebook_id = ? AND source_filename = ?
        """,
        (filename, notebook_id, old_filename)
    )
    cursor.execute("UPDATE notebooks SET updated_at = ? WHERE id = ?", (now, notebook_id))
    conn.commit()
    conn.close()
    return {"old_filename": old_filename, "filename": filename, "updated_at": now}


def update_source_transcript_record(notebook_id, source_id, transcript_text):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE sources
        SET transcript_text = ?, transcript_updated_at = ?, indexed_at = ?
        WHERE id = ? AND notebook_id = ?
        """,
        (transcript_text, now, now, source_id, notebook_id)
    )
    cursor.execute("UPDATE notebooks SET updated_at = ? WHERE id = ?", (now, notebook_id))
    conn.commit()
    conn.close()
    return now


def delete_notebook_record(notebook_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")
    cursor.execute("DELETE FROM notebooks WHERE id = ?", (notebook_id,))
    conn.commit()
    conn.close()


def insert_source_record(notebook_id, source_id, filename, transcript_text, timed_segments_json):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO sources
            (id, notebook_id, filename, added_at, transcript_text, timed_segments, transcript_updated_at, indexed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (source_id, notebook_id, filename, now, transcript_text, timed_segments_json, now, now)
    )
    cursor.execute("UPDATE notebooks SET updated_at = ? WHERE id = ?", (now, notebook_id))
    conn.commit()
    conn.close()
    return now


def insert_ai_summary_message(notebook_id, filename, structured_knowledge):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_result = f"**《{filename}》重點摘要**\n\n{structured_knowledge}"
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO messages (notebook_id, sender, text, created_at, references_json) VALUES (?, ?, ?, ?, ?)",
        (notebook_id, "AI", formatted_result, now, None)
    )
    conn.commit()
    conn.close()


def save_suggested_questions(notebook_id, source_filename, questions):
    """儲存最新推薦問題；若本次沒有題目，就清空畫面顯示用的舊題目"""
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "DELETE FROM suggested_questions WHERE notebook_id = ?",
        (notebook_id,)
    )
    saved_questions = []
    for priority, question in enumerate(questions, start=1):
        cursor.execute(
            "INSERT INTO suggested_questions (notebook_id, source_filename, question, priority, used, created_at) VALUES (?, ?, ?, ?, 0, ?)",
            (notebook_id, source_filename, question, priority, now)
        )
        saved_questions.append({
            "id": cursor.lastrowid,
            "question": question,
            "priority": priority,
            "source_filename": source_filename,
            "created_at": now
        })
    conn.commit()
    conn.close()
    return saved_questions


def get_recent_messages(notebook_id, limit=6):
    """讀取指定筆記本最近 N 筆對話紀錄，用於多輪對話記憶"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT sender, text FROM messages WHERE notebook_id = ? ORDER BY id DESC LIMIT ?",
            (notebook_id, limit)
        )
        rows = cursor.fetchall()
        conn.close()
        rows.reverse()
        return [{"sender": r[0], "text": r[1]} for r in rows]
    except Exception:
        return []


def append_qa_messages(notebook_id, question, answer, references=None):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    references_json = json.dumps(references or [], ensure_ascii=False)
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO messages (notebook_id, sender, text, created_at, references_json) VALUES (?, ?, ?, ?, ?)",
        (notebook_id, "User", question, now, None)
    )
    cursor.execute(
        "INSERT INTO messages (notebook_id, sender, text, created_at, references_json) VALUES (?, ?, ?, ?, ?)",
        (notebook_id, "AI", answer, now, references_json)
    )
    cursor.execute("UPDATE notebooks SET updated_at = ? WHERE id = ?", (now, notebook_id))
    conn.commit()
    conn.close()
