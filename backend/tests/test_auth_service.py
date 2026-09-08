import importlib.util
import sqlite3
import sys
import tempfile
import time
import unittest
from http.cookies import SimpleCookie
from pathlib import Path
from unittest import mock

from fastapi import HTTPException, Response
from starlette.requests import Request


BACKEND_DIR = Path(__file__).resolve().parents[1]


def load_modules():
    db_spec = importlib.util.spec_from_file_location("db", BACKEND_DIR / "db.py")
    db_module = importlib.util.module_from_spec(db_spec)
    assert db_spec and db_spec.loader
    db_spec.loader.exec_module(db_module)

    with mock.patch.dict(sys.modules, {"db": db_module}):
        auth_spec = importlib.util.spec_from_file_location("auth_under_test", BACKEND_DIR / "auth_service.py")
        auth_module = importlib.util.module_from_spec(auth_spec)
        assert auth_spec and auth_spec.loader
        auth_spec.loader.exec_module(auth_module)
    return db_module, auth_module


def request_with_cookie(cookie_name, token):
    return Request({
        "type": "http",
        "method": "GET",
        "path": "/api/auth/me",
        "headers": [(b"cookie", f"{cookie_name}={token}".encode("ascii"))],
    })


db_module, auth_module = load_modules()


class AuthServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db, self.auth = db_module, auth_module
        self.db.DB_PATH = str(Path(self.temp_dir.name) / "notebooks.db")
        self.db.init_db()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_first_user_claims_legacy_notebooks_and_users_are_isolated(self):
        legacy_notebooks = [self.db.create_notebook_record() for _ in range(4)]

        first = self.auth.register_user("StudentA", "secret123")
        second = self.auth.register_user("StudentB", "secret456")
        second_notebook = self.db.create_notebook_record(second["id"])

        self.assertEqual(
            {item["id"] for item in self.db.get_notebooks_data(first["id"])},
            {item["id"] for item in legacy_notebooks},
        )
        self.assertEqual(
            [item["id"] for item in self.db.get_notebooks_data(second["id"])],
            [second_notebook["id"]],
        )
        self.assertTrue(self.db.notebook_belongs_to_user(legacy_notebooks[0]["id"], first["id"]))
        self.assertFalse(self.db.notebook_belongs_to_user(legacy_notebooks[0]["id"], second["id"]))
        with self.assertRaises(HTTPException) as forbidden:
            self.auth.require_notebook_owner(legacy_notebooks[0]["id"], second["id"])
        self.assertEqual(forbidden.exception.status_code, 404)
        del forbidden

    def test_password_and_session_token_are_only_stored_as_hashes(self):
        user = self.auth.register_user("student", "secret123")
        response = Response()
        self.auth.start_session(response, user["id"])

        cookie = SimpleCookie()
        cookie.load(response.headers["set-cookie"])
        raw_token = cookie[self.auth.SESSION_COOKIE_NAME].value
        self.assertTrue(cookie[self.auth.SESSION_COOKIE_NAME]["httponly"])
        self.assertEqual(cookie[self.auth.SESSION_COOKIE_NAME]["samesite"].lower(), "lax")
        self.assertFalse(cookie[self.auth.SESSION_COOKIE_NAME]["secure"])
        self.assertFalse(cookie[self.auth.SESSION_COOKIE_NAME]["max-age"])

        conn = sqlite3.connect(self.db.DB_PATH)
        password_hash = conn.execute("SELECT password_hash FROM users WHERE id = ?", (user["id"],)).fetchone()[0]
        stored_token = conn.execute("SELECT token_hash FROM sessions WHERE user_id = ?", (user["id"],)).fetchone()[0]
        conn.close()

        self.assertNotEqual(password_hash, "secret123")
        self.assertNotEqual(stored_token, raw_token)
        self.assertEqual(stored_token, self.auth.hash_session_token(raw_token))
        self.assertEqual(self.auth.require_user(request_with_cookie(self.auth.SESSION_COOKIE_NAME, raw_token))["id"], user["id"])

    def test_login_validation_duplicate_username_and_expired_session(self):
        user = self.auth.register_user("Student", "secret123")
        self.assertEqual(self.auth.authenticate_user("student", "secret123")["id"], user["id"])

        with self.assertRaises(HTTPException) as duplicate:
            self.auth.register_user("STUDENT", "another123")
        self.assertEqual(duplicate.exception.status_code, 409)
        del duplicate

        with self.assertRaises(HTTPException) as wrong_password:
            self.auth.authenticate_user("Student", "incorrect")
        self.assertEqual(wrong_password.exception.status_code, 401)
        del wrong_password

        token = "expired-session-token"
        now = int(time.time())
        self.db.create_session_record(self.auth.hash_session_token(token), user["id"], now - 20, now - 10)
        with self.assertRaises(HTTPException) as expired:
            self.auth.require_user(request_with_cookie(self.auth.SESSION_COOKIE_NAME, token))
        self.assertEqual(expired.exception.status_code, 401)
        del expired

    def test_username_and_password_rules(self):
        invalid_cases = [
            ("ab", "secret123"),
            ("has space", "secret123"),
            ("student", "12345"),
            ("student", "密" * 25),
        ]
        for username, password in invalid_cases:
            with self.subTest(username=username, password_length=len(password)):
                with self.assertRaises(HTTPException) as invalid:
                    self.auth.register_user(username, password)
                self.assertEqual(invalid.exception.status_code, 422)

    def test_missing_cookie_is_unauthorized(self):
        request = Request({"type": "http", "method": "GET", "path": "/api/notebooks/", "headers": []})
        with self.assertRaises(HTTPException) as unauthorized:
            self.auth.require_user(request)
        self.assertEqual(unauthorized.exception.status_code, 401)


if __name__ == "__main__":
    unittest.main()
