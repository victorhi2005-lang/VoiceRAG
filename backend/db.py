import json
import os
import sqlite3
import uuid
from datetime import datetime


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "notebooks.db")


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT NOT NULL UNIQUE COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS notebooks (
            id TEXT PRIMARY KEY,
            name TEXT,
            icon TEXT,
            updated_at TEXT,
            user_id TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')
    cursor.execute("PRAGMA table_info(notebooks)")
    notebook_columns = [row[1] for row in cursor.fetchall()]
    if "user_id" not in notebook_columns:
        cursor.execute("ALTER TABLE notebooks ADD COLUMN user_id TEXT REFERENCES users(id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_notebooks_user_id ON notebooks(user_id)")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            token_hash TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            expires_at INTEGER NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at)")
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
        "analysis_mode": "ALTER TABLE sources ADD COLUMN analysis_mode TEXT",
        "analysis_status": "ALTER TABLE sources ADD COLUMN analysis_status TEXT",
        "analysis_json": "ALTER TABLE sources ADD COLUMN analysis_json TEXT",
        "analysis_updated_at": "ALTER TABLE sources ADD COLUMN analysis_updated_at TEXT",
        "source_type": "ALTER TABLE sources ADD COLUMN source_type TEXT DEFAULT 'audio'",
        "mime_type": "ALTER TABLE sources ADD COLUMN mime_type TEXT",
        "content_segments_json": "ALTER TABLE sources ADD COLUMN content_segments_json TEXT",
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
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS diagrams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            notebook_id TEXT,
            title TEXT,
            diagram_type TEXT,
            prompt TEXT,
            mermaid_code TEXT,
            diagram_data_json TEXT,
            references_json TEXT,
            created_at TEXT,
            updated_at TEXT,
            FOREIGN KEY(notebook_id) REFERENCES notebooks(id) ON DELETE CASCADE
        )
    ''')
    cursor.execute("PRAGMA table_info(diagrams)")
    diagram_columns = [row[1] for row in cursor.fetchall()]
    if "diagram_data_json" not in diagram_columns:
        cursor.execute("ALTER TABLE diagrams ADD COLUMN diagram_data_json TEXT")
    conn.commit()
    conn.close()


def get_notebooks_data(user_id=None):
    conn = get_connection()
    cursor = conn.cursor()
    if user_id is None:
        cursor.execute("SELECT id, name, icon, updated_at FROM notebooks ORDER BY updated_at DESC")
    else:
        cursor.execute(
            "SELECT id, name, icon, updated_at FROM notebooks WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        )
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


def create_notebook_record(user_id=None):
    nid = str(uuid.uuid4())
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    name = "未命名筆記本"
    icon = "📓"

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO notebooks (id, name, icon, updated_at, user_id) VALUES (?, ?, ?, ?, ?)",
        (nid, name, icon, now, user_id)
    )
    conn.commit()
    conn.close()

    return {"id": nid, "name": name, "icon": icon, "updated_at": now, "source_count": 0}


def create_user_record(username, password_hash):
    user_id = str(uuid.uuid4())
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("BEGIN IMMEDIATE")
        is_first_user = cursor.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
        cursor.execute(
            "INSERT INTO users (id, username, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (user_id, username, password_hash, now),
        )
        if is_first_user:
            cursor.execute(
                "UPDATE notebooks SET user_id = ? WHERE user_id IS NULL",
                (user_id,),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return {"id": user_id, "username": username, "created_at": now}


def get_user_by_username(username):
    conn = get_connection()
    row = conn.execute(
        "SELECT id, username, password_hash, created_at FROM users WHERE username = ? COLLATE NOCASE",
        (username,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {"id": row[0], "username": row[1], "password_hash": row[2], "created_at": row[3]}


def get_user_by_id(user_id):
    conn = get_connection()
    row = conn.execute(
        "SELECT id, username, created_at FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {"id": row[0], "username": row[1], "created_at": row[2]}


def create_session_record(token_hash, user_id, created_at, expires_at):
    conn = get_connection()
    conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (created_at,))
    conn.execute(
        "INSERT INTO sessions (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
        (token_hash, user_id, created_at, expires_at),
    )
    conn.commit()
    conn.close()


def get_session_user(token_hash, now_timestamp):
    conn = get_connection()
    row = conn.execute(
        """
        SELECT users.id, users.username, users.created_at
        FROM sessions
        JOIN users ON users.id = sessions.user_id
        WHERE sessions.token_hash = ? AND sessions.expires_at > ?
        """,
        (token_hash, now_timestamp),
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {"id": row[0], "username": row[1], "created_at": row[2]}


def delete_session_record(token_hash):
    conn = get_connection()
    conn.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))
    conn.commit()
    conn.close()


def notebook_belongs_to_user(notebook_id, user_id):
    conn = get_connection()
    row = conn.execute(
        "SELECT 1 FROM notebooks WHERE id = ? AND user_id = ?",
        (notebook_id, user_id),
    ).fetchone()
    conn.close()
    return row is not None


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
            analysis_mode,
            analysis_status,
            analysis_updated_at,
            COALESCE(source_type, 'audio'),
            mime_type,
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
            "analysis_mode": r[5],
            "analysis_status": r[6],
            "analysis_updated_at": r[7],
            "source_type": r[8] or "audio",
            "mime_type": r[9] or "",
            "has_transcript": bool(r[10]),
            "has_content": bool(r[10])
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
    diagrams = get_diagrams_data(notebook_id, conn)

    conn.close()
    return {
        "sources": sources,
        "messages": messages,
        "suggested_questions": suggested_questions,
        "diagrams": diagrams,
    }


def parse_references_json(references_json):
    if not references_json:
        return []
    try:
        references = json.loads(references_json)
        return references if isinstance(references, list) else []
    except Exception:
        return []


def parse_diagram_data_json(diagram_data_json):
    if not diagram_data_json:
        return None
    try:
        diagram_data = json.loads(diagram_data_json)
        return diagram_data if isinstance(diagram_data, dict) else None
    except Exception:
        return None


def get_diagrams_data(notebook_id, conn=None):
    should_close = conn is None
    conn = conn or get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, title, diagram_type, prompt, mermaid_code, diagram_data_json, references_json, created_at, updated_at
        FROM diagrams
        WHERE notebook_id = ?
        ORDER BY updated_at DESC, id DESC
        """,
        (notebook_id,)
    )
    diagrams = [
        {
            "id": row[0],
            "title": row[1],
            "diagram_type": row[2],
            "prompt": row[3],
            "mermaid_code": row[4],
            "diagram_data": parse_diagram_data_json(row[5]),
            "references": parse_references_json(row[6]),
            "created_at": row[7],
            "updated_at": row[8],
        }
        for row in cursor.fetchall()
    ]
    if should_close:
        conn.close()
    return diagrams


def insert_diagram_record(notebook_id, title, diagram_type, prompt, mermaid_code, references=None, diagram_data=None):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    references_json = json.dumps(references or [], ensure_ascii=False)
    diagram_data_json = json.dumps(diagram_data, ensure_ascii=False) if isinstance(diagram_data, dict) else None
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO diagrams
            (notebook_id, title, diagram_type, prompt, mermaid_code, diagram_data_json, references_json, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (notebook_id, title, diagram_type, prompt, mermaid_code, diagram_data_json, references_json, now, now)
    )
    diagram_id = cursor.lastrowid
    cursor.execute("UPDATE notebooks SET updated_at = ? WHERE id = ?", (now, notebook_id))
    conn.commit()
    conn.close()
    return {
        "id": diagram_id,
        "title": title,
        "diagram_type": diagram_type,
        "prompt": prompt,
        "mermaid_code": mermaid_code,
        "diagram_data": diagram_data if isinstance(diagram_data, dict) else None,
        "references": references or [],
        "created_at": now,
        "updated_at": now,
    }


def delete_diagram_record(notebook_id, diagram_id):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM diagrams WHERE id = ? AND notebook_id = ?",
        (diagram_id, notebook_id)
    )
    deleted = cursor.rowcount > 0
    if deleted:
        cursor.execute("UPDATE notebooks SET updated_at = ? WHERE id = ?", (now, notebook_id))
    conn.commit()
    conn.close()
    if not deleted:
        return None
    return {"id": diagram_id, "updated_at": now}


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
        SELECT
            id,
            filename,
            transcript_text,
            transcript_updated_at,
            indexed_at,
            analysis_mode,
            analysis_status,
            analysis_updated_at
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
        "indexed_at": row[4],
        "analysis_mode": row[5],
        "analysis_status": row[6],
        "analysis_updated_at": row[7]
    }


def get_source_content_data(notebook_id, source_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            id, filename, transcript_text, content_segments_json,
            COALESCE(source_type, 'audio'), mime_type,
            transcript_updated_at, indexed_at,
            analysis_mode, analysis_status, analysis_updated_at
        FROM sources
        WHERE id = ? AND notebook_id = ?
        """,
        (source_id, notebook_id)
    )
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None

    segments = []
    if row[3]:
        try:
            parsed = json.loads(row[3])
            if isinstance(parsed, list):
                segments = parsed
        except Exception:
            segments = []
    return {
        "id": row[0],
        "filename": row[1],
        "content_text": row[2] or "",
        "segments": segments,
        "source_type": row[4] or "audio",
        "mime_type": row[5] or "",
        "content_updated_at": row[6],
        "indexed_at": row[7],
        "analysis_mode": row[8],
        "analysis_status": row[9],
        "analysis_updated_at": row[10],
    }


def get_source_update_info(notebook_id, source_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            filename,
            transcript_text,
            timed_segments,
            analysis_mode,
            analysis_status,
            analysis_updated_at,
            COALESCE(source_type, 'audio'),
            mime_type,
            content_segments_json
        FROM sources
        WHERE id = ? AND notebook_id = ?
        """,
        (source_id, notebook_id)
    )
    row = cursor.fetchone()
    conn.close()
    return row


def get_source_analysis_input(notebook_id, source_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT filename, transcript_text, timed_segments, COALESCE(source_type, 'audio'), content_segments_json
        FROM sources
        WHERE id = ? AND notebook_id = ?
        """,
        (source_id, notebook_id)
    )
    row = cursor.fetchone()
    conn.close()
    return row


def get_source_filename_suggestion_input(notebook_id, source_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT filename, transcript_text, analysis_json, COALESCE(source_type, 'audio')
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


def get_source_file_info(notebook_id, source_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, filename, COALESCE(source_type, 'audio'), mime_type
        FROM sources
        WHERE id = ? AND notebook_id = ?
        """,
        (source_id, notebook_id)
    )
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row[0],
        "filename": row[1],
        "source_type": row[2] or "audio",
        "mime_type": row[3] or "",
    }


def get_source_delete_info(notebook_id, source_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, filename, COALESCE(source_type, 'audio')
        FROM sources
        WHERE id = ? AND notebook_id = ?
        """,
        (source_id, notebook_id)
    )
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return {"id": row[0], "filename": row[1], "source_type": row[2] or "audio"}


def get_notebook_delete_info(notebook_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, name FROM notebooks WHERE id = ?",
        (notebook_id,)
    )
    notebook_row = cursor.fetchone()
    if not notebook_row:
        conn.close()
        return None

    cursor.execute(
        """
        SELECT id, filename, COALESCE(source_type, 'audio')
        FROM sources
        WHERE notebook_id = ?
        """,
        (notebook_id,)
    )
    sources = [
        {"id": row[0], "filename": row[1], "source_type": row[2] or "audio"}
        for row in cursor.fetchall()
    ]
    conn.close()
    return {
        "id": notebook_row[0],
        "name": notebook_row[1],
        "sources": sources
    }


def get_source_filenames(notebook_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT filename FROM sources WHERE notebook_id = ?", (notebook_id,))
    filenames = [row[0] for row in cursor.fetchall()]
    conn.close()
    return filenames


def get_all_source_filenames(ignore_filename=None):
    conn = get_connection()
    if ignore_filename is None:
        rows = conn.execute("SELECT filename FROM sources").fetchall()
    else:
        rows = conn.execute(
            "SELECT filename FROM sources WHERE filename != ?",
            (ignore_filename,),
        ).fetchall()
    conn.close()
    return [row[0] for row in rows]


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


def update_source_transcript_record(
    notebook_id,
    source_id,
    transcript_text,
    analysis_mode=None,
    analysis_status=None,
    analysis_json=None,
):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection()
    cursor = conn.cursor()
    if analysis_mode is None and analysis_status is None and analysis_json is None:
        cursor.execute(
            """
            UPDATE sources
            SET transcript_text = ?, transcript_updated_at = ?, indexed_at = ?
            WHERE id = ? AND notebook_id = ?
            """,
            (transcript_text, now, now, source_id, notebook_id)
        )
    else:
        cursor.execute(
            """
            UPDATE sources
            SET
                transcript_text = ?,
                transcript_updated_at = ?,
                indexed_at = ?,
                analysis_mode = ?,
                analysis_status = ?,
                analysis_json = ?,
                analysis_updated_at = ?
            WHERE id = ? AND notebook_id = ?
            """,
            (
                transcript_text,
                now,
                now,
                analysis_mode,
                analysis_status,
                analysis_json,
                now,
                source_id,
                notebook_id
            )
        )
    cursor.execute("UPDATE notebooks SET updated_at = ? WHERE id = ?", (now, notebook_id))
    conn.commit()
    conn.close()
    return now


def update_source_analysis_record(notebook_id, source_id, analysis_mode, analysis_status, analysis_json):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE sources
        SET analysis_mode = ?, analysis_status = ?, analysis_json = ?, analysis_updated_at = ?
        WHERE id = ? AND notebook_id = ?
        """,
        (analysis_mode, analysis_status, analysis_json, now, source_id, notebook_id)
    )
    cursor.execute("UPDATE notebooks SET updated_at = ? WHERE id = ?", (now, notebook_id))
    conn.commit()
    conn.close()
    return now


def delete_source_record(notebook_id, source_id):
    source = get_source_delete_info(notebook_id, source_id)
    if not source:
        return None

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM sources WHERE id = ? AND notebook_id = ?",
        (source_id, notebook_id)
    )
    cursor.execute(
        "DELETE FROM suggested_questions WHERE notebook_id = ? AND source_filename = ?",
        (notebook_id, source["filename"])
    )
    cursor.execute("UPDATE notebooks SET updated_at = ? WHERE id = ?", (now, notebook_id))
    conn.commit()
    conn.close()
    return {
        "id": source["id"],
        "filename": source["filename"],
        "updated_at": now
    }


def delete_notebook_record(notebook_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")
    cursor.execute("DELETE FROM notebooks WHERE id = ?", (notebook_id,))
    conn.commit()
    conn.close()


def clear_all_records():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")
    cursor.execute("DELETE FROM diagrams")
    cursor.execute("DELETE FROM suggested_questions")
    cursor.execute("DELETE FROM messages")
    cursor.execute("DELETE FROM sources")
    cursor.execute("DELETE FROM notebooks")
    cursor.execute(
        "DELETE FROM sqlite_sequence WHERE name IN ('messages', 'suggested_questions', 'diagrams')"
    )
    conn.commit()
    conn.close()


def get_data_consistency_snapshot(user_id=None):
    conn = get_connection()
    cursor = conn.cursor()
    if user_id is None:
        cursor.execute("SELECT id, name FROM notebooks")
    else:
        cursor.execute("SELECT id, name FROM notebooks WHERE user_id = ?", (user_id,))
    notebooks = [
        {"id": row[0], "name": row[1]}
        for row in cursor.fetchall()
    ]
    if user_id is None:
        cursor.execute("SELECT id, notebook_id, filename, COALESCE(source_type, 'audio') FROM sources")
    else:
        cursor.execute(
            """
            SELECT sources.id, sources.notebook_id, sources.filename, COALESCE(sources.source_type, 'audio')
            FROM sources
            JOIN notebooks ON notebooks.id = sources.notebook_id
            WHERE notebooks.user_id = ?
            """,
            (user_id,),
        )
    sources = [
        {"id": row[0], "notebook_id": row[1], "filename": row[2], "source_type": row[3] or "audio"}
        for row in cursor.fetchall()
    ]
    conn.close()
    return {"notebooks": notebooks, "sources": sources}


def insert_source_record(
    notebook_id,
    source_id,
    filename,
    transcript_text,
    timed_segments_json,
    analysis_mode=None,
    analysis_status=None,
    analysis_json=None,
    source_type="audio",
    mime_type=None,
    content_segments_json=None,
):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO sources
            (
                id,
                notebook_id,
                filename,
                added_at,
                transcript_text,
                timed_segments,
                transcript_updated_at,
                indexed_at,
                analysis_mode,
                analysis_status,
                analysis_json,
                analysis_updated_at,
                source_type,
                mime_type,
                content_segments_json
            )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            source_id,
            notebook_id,
            filename,
            now,
            transcript_text,
            timed_segments_json,
            now,
            now,
            analysis_mode,
            analysis_status,
            analysis_json,
            now if analysis_status else None,
            source_type,
            mime_type,
            content_segments_json,
        )
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
