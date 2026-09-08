import hashlib
import secrets
import sqlite3
import time

import bcrypt
from fastapi import HTTPException, Request, Response, status

from db import (
    create_session_record,
    create_user_record,
    delete_session_record,
    get_session_user,
    get_user_by_username,
    notebook_belongs_to_user,
)


SESSION_COOKIE_NAME = "voicerag_session"
SESSION_LIFETIME_SECONDS = 8 * 60 * 60


def validate_credentials(username: str, password: str) -> tuple[str, bytes]:
    normalized_username = username.strip()
    if not 3 <= len(normalized_username) <= 20 or any(char.isspace() for char in normalized_username):
        raise HTTPException(
            status_code=422,
            detail="使用者名稱需為 3～20 字，且不可包含空白。",
        )

    password_bytes = password.encode("utf-8")
    if not 6 <= len(password_bytes) <= 72:
        raise HTTPException(
            status_code=422,
            detail="密碼長度需為 6～72 bytes。",
        )
    return normalized_username, password_bytes


def hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def start_session(response: Response, user_id: str) -> None:
    token = secrets.token_urlsafe(32)
    now = int(time.time())
    create_session_record(
        hash_session_token(token),
        user_id,
        now,
        now + SESSION_LIFETIME_SECONDS,
    )
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=False,
        samesite="lax",
        path="/",
    )


def register_user(username: str, password: str) -> dict:
    normalized_username, password_bytes = validate_credentials(username, password)
    password_hash = bcrypt.hashpw(password_bytes, bcrypt.gensalt()).decode("utf-8")
    try:
        return create_user_record(normalized_username, password_hash)
    except sqlite3.IntegrityError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="這個使用者名稱已被使用。",
        ) from None


def authenticate_user(username: str, password: str) -> dict:
    normalized_username = username.strip()
    password_bytes = password.encode("utf-8")
    user = get_user_by_username(normalized_username)
    if not user or len(password_bytes) > 72:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="使用者名稱或密碼錯誤。",
        )
    try:
        password_matches = bcrypt.checkpw(password_bytes, user["password_hash"].encode("utf-8"))
    except ValueError:
        password_matches = False
    if not password_matches:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="使用者名稱或密碼錯誤。",
        )
    return {"id": user["id"], "username": user["username"], "created_at": user["created_at"]}


def require_user(request: Request) -> dict:
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="請先登入。")
    user = get_session_user(hash_session_token(token), int(time.time()))
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="登入已逾時，請重新登入。")
    return user


def require_notebook_owner(notebook_id: str, user_id: str) -> None:
    if not notebook_belongs_to_user(notebook_id, user_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="找不到筆記本。")


def end_session(request: Request, response: Response) -> None:
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if token:
        delete_session_record(hash_session_token(token))
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")
