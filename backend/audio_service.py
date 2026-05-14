import gc
import json
import os
import re
import shutil
import uuid

import ollama
import torch

from db import (
    get_source_audio_info,
    get_source_filenames,
    insert_ai_summary_message,
    insert_source_record,
    save_suggested_questions,
    update_source_filename_record,
)
from models import whisper_model
from rag_service import generate_suggested_questions, index_source_transcript, update_source_filename_metadata


UPLOAD_DIR = "audio_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def sanitize_filename(filename, fallback="audio", extension=None):
    original = os.path.basename(filename or "")
    stem, ext = os.path.splitext(original)
    if extension is not None:
        ext = extension
    if ext and not ext.startswith("."):
        ext = f".{ext}"

    cleaned_stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem).strip(" ._")
    cleaned_stem = re.sub(r"\s+", " ", cleaned_stem)
    if not cleaned_stem:
        cleaned_stem = fallback
    if len(cleaned_stem) > 80:
        cleaned_stem = cleaned_stem[:80].rstrip(" ._")

    cleaned_ext = re.sub(r'[^A-Za-z0-9.]', "", ext or "")
    return f"{cleaned_stem}{cleaned_ext or '.webm'}"


def ensure_unique_filename(filename, existing_filenames=None, ignore_filename=None):
    existing = set(existing_filenames or [])
    stem, ext = os.path.splitext(filename)
    candidate = filename
    counter = 2
    while candidate in existing or (
        candidate != ignore_filename and os.path.exists(os.path.join(UPLOAD_DIR, candidate))
    ):
        candidate = f"{stem}-{counter}{ext}"
        counter += 1
    return candidate


def save_uploaded_audio(file, filename):
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return file_path


def get_audio_file_path(notebook_id, source_id):
    source = get_source_audio_info(notebook_id, source_id)
    if not source:
        return None

    filename = os.path.basename(source["filename"] or "")
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        return None
    return {"path": file_path, "filename": filename}


def get_fallback_recording_filename(notebook_id, extension):
    existing = set(get_source_filenames(notebook_id))
    counter = 1
    while True:
        filename = sanitize_filename(f"錄音{counter}{extension}", fallback=f"錄音{counter}", extension=extension)
        if filename not in existing and not os.path.exists(os.path.join(UPLOAD_DIR, filename)):
            return filename
        counter += 1


def clean_ai_filename(raw_text):
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").replace("text\n", "", 1).replace("json\n", "", 1).strip()
    cleaned = cleaned.splitlines()[0].strip() if cleaned else ""
    cleaned = cleaned.strip('"').strip("'").strip()
    cleaned = re.sub(r"\.(mp3|m4a|wav|webm|ogg|mp4)$", "", cleaned, flags=re.IGNORECASE).strip()
    if not cleaned or cleaned in {"無法判斷", "無法命名", "錄音", "audio"}:
        return ""
    return cleaned


def get_fallback_ai_filename(notebook_id, extension, fallback_filename=None, current_filename=None):
    if fallback_filename:
        filename = sanitize_filename(fallback_filename, fallback="audio", extension=extension)
        return ensure_unique_filename(filename, get_source_filenames(notebook_id), ignore_filename=current_filename)
    return get_fallback_recording_filename(notebook_id, extension)


def generate_ai_audio_filename(notebook_id, transcript_text, extension, fallback_filename=None, current_filename=None):
    fallback_filename = get_fallback_ai_filename(notebook_id, extension, fallback_filename, current_filename)
    compact_text = re.sub(r"\s+", "", transcript_text or "")
    if len(compact_text) < 20:
        return fallback_filename

    prompt = f"""
請根據以下錄音逐字稿，產生一個適合當音檔名稱的繁體中文短檔名。

規則：
1. 只輸出檔名本身，不要副檔名。
2. 長度 4 到 18 個中文字左右。
3. 不要使用 / \\ : * ? " < > | 等檔名禁用符號。
4. 如果內容太短、太雜或無法判斷主題，請只輸出：無法判斷。

逐字稿：
{transcript_text}
"""
    try:
        response = ollama.chat(model="qwen3:14b", messages=[{"role": "user", "content": prompt + "\n/no_think"}])
        title = clean_ai_filename(response["message"]["content"])
    except Exception:
        title = ""

    if not title:
        return fallback_filename

    filename = sanitize_filename(title, fallback=os.path.splitext(fallback_filename)[0], extension=extension)
    return ensure_unique_filename(filename, get_source_filenames(notebook_id), ignore_filename=current_filename)


def rename_audio_file(current_path, new_filename):
    new_path = os.path.join(UPLOAD_DIR, new_filename)
    if os.path.abspath(current_path) == os.path.abspath(new_path):
        return current_path
    if os.path.exists(new_path):
        new_filename = ensure_unique_filename(new_filename)
        new_path = os.path.join(UPLOAD_DIR, new_filename)
    os.replace(current_path, new_path)
    return new_path


def rename_source_filename(notebook_id, source_id, requested_filename):
    source = get_source_audio_info(notebook_id, source_id)
    if not source:
        return {"status": "error", "message": "找不到指定來源。"}

    old_filename = source["filename"]
    old_ext = os.path.splitext(old_filename)[1] or ".webm"
    new_filename = sanitize_filename(requested_filename, fallback="錄音", extension=old_ext)
    existing = [name for name in get_source_filenames(notebook_id) if name != old_filename]
    new_filename = ensure_unique_filename(new_filename, existing, ignore_filename=old_filename)

    if new_filename == old_filename:
        return {
            "status": "success",
            "changed": False,
            "source": {"id": source_id, "filename": old_filename}
        }

    old_path = os.path.join(UPLOAD_DIR, os.path.basename(old_filename))
    if os.path.exists(old_path):
        rename_audio_file(old_path, new_filename)

    updated = update_source_filename_record(notebook_id, source_id, new_filename)
    if not updated:
        return {"status": "error", "message": "找不到指定來源。"}

    try:
        update_source_filename_metadata(notebook_id, source_id, new_filename)
    except Exception:
        pass

    return {
        "status": "success",
        "changed": True,
        "source": {
            "id": source_id,
            "filename": new_filename,
            "updated_at": updated["updated_at"]
        }
    }


def transcribe_audio(file_path):
    try:
        ollama.generate(model="qwen3:14b", prompt="", keep_alive=0)
        torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass

    segments, info = whisper_model.transcribe(
        file_path,
        beam_size=5,
        language="zh",
        initial_prompt="這是一段繁體中文的台灣口音逐字稿：",
        vad_filter=True,
        batch_size=16
    )
    timed_segments = [
        {
            "text": segment.text,
            "start": segment.start,
            "end": segment.end
        }
        for segment in segments
    ]
    transcript_text = "".join([segment["text"] for segment in timed_segments])
    torch.cuda.empty_cache()
    gc.collect()
    return transcript_text, timed_segments


def generate_summary(transcript_text):
    prompt = f"""
        你是一個專業的 AI 知識分析助手。請閱讀以下的口述語音逐字稿，
        進行深入的訊息分析，並給出一份精煉的「整體重點摘要」。
        請用流暢的段落來總結核心訊息，幫助讀者快速掌握整段語音的精華。
        
        ⚠️ 絕對要求：請務必使用「繁體中文 (Traditional Chinese)」輸出，嚴禁出現簡體字！
        
        語音逐字稿內容：\n{transcript_text}
        """
    response = ollama.chat(model="qwen3:14b", messages=[{"role": "user", "content": prompt + "\n/no_think"}])
    return response["message"]["content"]


def process_audio_upload(notebook_id, file, auto_filename=False, fallback_filename=None):
    source_id = str(uuid.uuid4())
    original_ext = os.path.splitext(file.filename or "")[1] or ".webm"
    if auto_filename:
        upload_filename = ensure_unique_filename(
            sanitize_filename(f"recording-{source_id[:8]}{original_ext}", fallback="recording", extension=original_ext)
        )
    else:
        upload_filename = ensure_unique_filename(
            sanitize_filename(file.filename, fallback="audio", extension=original_ext),
            get_source_filenames(notebook_id)
        )
    file_path = save_uploaded_audio(file, upload_filename)

    try:
        transcript_text, timed_segments = transcribe_audio(file_path)
    except Exception as e:
        return {"status": "error", "message": f"語音辨識失敗: {str(e)}"}

    filename = upload_filename
    if auto_filename:
        filename = generate_ai_audio_filename(
            notebook_id,
            transcript_text,
            original_ext,
            fallback_filename=fallback_filename,
            current_filename=upload_filename
        )
        try:
            file_path = rename_audio_file(file_path, filename)
        except Exception:
            filename = upload_filename

    try:
        chunks_count = index_source_transcript(notebook_id, source_id, filename, transcript_text, timed_segments)
        source_saved_at = insert_source_record(
            notebook_id,
            source_id,
            filename,
            transcript_text,
            json.dumps(timed_segments, ensure_ascii=False)
        )
    except Exception as e:
        return {"status": "error", "message": f"寫入向量資料庫失敗: {str(e)}"}
    finally:
        gc.collect()

    try:
        structured_knowledge = generate_summary(transcript_text)
    except Exception as e:
        return {"status": "error", "message": f"AI 摘要失敗: {str(e)}"}

    suggested_questions = generate_suggested_questions(transcript_text)
    saved_suggested_questions = []

    try:
        insert_ai_summary_message(notebook_id, filename, structured_knowledge)
        saved_suggested_questions = save_suggested_questions(notebook_id, filename, suggested_questions)
    except Exception:
        pass

    return {
        "filename": filename,
        "status": "success",
        "message": f"✅ 成功存入 {chunks_count} 個知識片段。",
        "raw_transcript": transcript_text,
        "structured_knowledge": structured_knowledge,
        "suggested_questions": saved_suggested_questions,
        "source": {
            "id": source_id,
            "filename": filename,
            "added_at": source_saved_at,
            "has_transcript": True,
            "transcript_updated_at": source_saved_at,
            "indexed_at": source_saved_at
        }
    }
