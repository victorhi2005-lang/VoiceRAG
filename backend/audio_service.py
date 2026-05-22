import gc
import json
import os
import re
import shutil
import uuid
from typing import Any, cast

from db import (
    delete_notebook_record,
    delete_source_record,
    get_data_consistency_snapshot,
    get_notebook_delete_info,
    get_source_analysis_input,
    get_source_audio_info,
    get_source_delete_info,
    get_source_filename_suggestion_input,
    get_source_filenames,
    insert_ai_summary_message,
    insert_source_record,
    save_suggested_questions,
    update_source_analysis_record,
    update_source_filename_record,
)
from models import (
    clear_cuda_memory,
    get_whisper_batch_size,
    get_whisper_model,
    unload_whisper_model,
)
from llm_service import generate_text, is_gemini_provider, is_ollama_provider, normalize_llm_provider, stop_llm_generation
from rag_service import (
    clear_orphan_chroma_segments,
    build_source_analysis,
    delete_notebook_collection,
    delete_source_from_index,
    get_chroma_consistency_snapshot,
    index_source_analysis_documents,
    index_source_transcript,
    should_use_deep_analysis,
    update_source_filename_metadata,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "audio_uploads")
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


def get_audio_path_for_filename(filename):
    safe_filename = os.path.basename(filename or "")
    if not safe_filename:
        return None

    upload_root = os.path.abspath(UPLOAD_DIR)
    file_path = os.path.abspath(os.path.join(upload_root, safe_filename))
    if os.path.commonpath([upload_root, file_path]) != upload_root:
        return None
    return file_path


def delete_audio_file(filename):
    file_path = get_audio_path_for_filename(filename)
    if not file_path or not os.path.exists(file_path):
        return {"deleted": False, "missing": True, "path": file_path}
    if not os.path.isfile(file_path):
        return {"deleted": False, "missing": False, "path": file_path}

    os.remove(file_path)
    return {"deleted": True, "missing": False, "path": file_path}


def list_audio_upload_files():
    if not os.path.isdir(UPLOAD_DIR):
        return []
    return sorted(
        filename
        for filename in os.listdir(UPLOAD_DIR)
        if os.path.isfile(os.path.join(UPLOAD_DIR, filename))
    )


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
    cleaned = re.sub(r'^[「『【\[\(（\s]+|[」』】\]\)）\s]+$', "", cleaned).strip()
    if not cleaned or cleaned in {"無法判斷", "無法命名", "錄音", "audio"}:
        return ""
    return cleaned


def get_filename_stem(filename):
    stem, _ = os.path.splitext(os.path.basename(filename or ""))
    stem = re.sub(r"\s+", " ", stem).strip(" ._-'\"")
    return stem


def is_generic_filename_stem(stem):
    compact = re.sub(r"[\s._\-()（）\[\]【】]+", "", stem or "").lower()
    if len(compact) < 3:
        return True

    generic_names = {
        "audio", "voice", "memo", "record", "recording", "untitled",
        "newrecording", "sound", "source", "file", "錄音", "音訊", "聲音",
        "未命名", "新錄音", "錄製音訊"
    }
    if compact in generic_names:
        return True
    if re.fullmatch(r"(recording|record|audio|voice|memo|source|file)[a-z0-9]*", compact):
        return True
    if re.fullmatch(r"(錄音|音訊|聲音|未命名|新錄音)\d*", compact):
        return True
    if re.fullmatch(r"[0-9a-f]{8,32}", compact):
        return True
    if re.fullmatch(r"\d{6,}", compact):
        return True
    if re.fullmatch(r"\d{4}\d{1,2}\d{1,2}\d*", compact):
        return True
    return False


def build_filename_context_sample(transcript_text, analysis=None, sample_size=900):
    if analysis:
        global_summary = cast(dict[str, Any], analysis.get("global_summary") or {})
        chapters = cast(list[dict[str, Any]], analysis.get("chapters") or [])
        analysis_parts = [
            f"全局標題：{global_summary.get('overall_title', '')}",
            f"整體摘要：{global_summary.get('overview', '')}",
        ]
        core_themes = global_summary.get("core_themes") or []
        if core_themes:
            analysis_parts.append(f"核心主題：{'、'.join(str(item) for item in core_themes[:5])}")
        chapter_titles = [
            str(chapter.get("title", "")).strip()
            for chapter in chapters
            if str(chapter.get("title", "")).strip()
        ]
        if chapter_titles:
            analysis_parts.append(f"章節標題：{'、'.join(chapter_titles[:8])}")
        analysis_context = "\n".join(part for part in analysis_parts if part.strip())
        if analysis_context.strip():
            return analysis_context

    text = re.sub(r"\s+", " ", transcript_text or "").strip()
    if len(text) <= sample_size * 3:
        return text

    middle_start = max(0, len(text) // 2 - sample_size // 2)
    return "\n".join([
        f"開頭片段：{text[:sample_size]}",
        f"中段片段：{text[middle_start:middle_start + sample_size]}",
        f"結尾片段：{text[-sample_size:]}"
    ])


def has_filename_topic_overlap(original_stem, ai_title):
    original = re.sub(r"[\s._\-()（）\[\]【】]+", "", original_stem or "").lower()
    title = re.sub(r"[\s._\-()（）\[\]【】]+", "", ai_title or "").lower()
    if not original or not title:
        return False
    if original in title or title in original:
        return True

    generic_chars = set("的與和及之研究探討分析摘要重點錄音音訊")
    original_chars = {char for char in original if char not in generic_chars}
    title_chars = {char for char in title if char not in generic_chars}
    if not original_chars or not title_chars:
        return False

    overlap = original_chars & title_chars
    return len(overlap) / max(len(original_chars), 1) >= 0.35


def get_fallback_ai_filename(notebook_id, extension, fallback_filename=None, current_filename=None):
    if fallback_filename:
        filename = sanitize_filename(fallback_filename, fallback="audio", extension=extension)
        return ensure_unique_filename(filename, get_source_filenames(notebook_id), ignore_filename=current_filename)
    return get_fallback_recording_filename(notebook_id, extension)


def generate_ai_audio_filename(
    notebook_id,
    transcript_text,
    extension,
    fallback_filename=None,
    current_filename=None,
    analysis=None,
    llm_provider=None,
    respect_original_context=True,
):
    original_stem = get_filename_stem(fallback_filename)
    is_generic_original = is_generic_filename_stem(original_stem)
    fallback_filename = get_fallback_ai_filename(notebook_id, extension, fallback_filename, current_filename)
    compact_text = re.sub(r"\s+", "", transcript_text or "")
    if len(compact_text) < 20:
        return fallback_filename

    naming_context = build_filename_context_sample(transcript_text, analysis)
    if is_generic_original:
        prompt = f"""
請根據以下錄音逐字稿片段，產生一個精準、自然、適合當音檔名稱的繁體中文短檔名。

規則：
1. 只輸出檔名本身，不要副檔名。
2. 長度 6 到 18 個中文字左右。
3. 不要使用 / \\ : * ? " < > | 等檔名禁用符號。
4. 請抓整體主題，不要只用單一細節、例子或開頭片段命名。
5. 如果內容太短、太雜或無法判斷主題，請只輸出：無法判斷。

逐字稿片段：
{naming_context}
"""
    else:
        prompt = f"""
請根據原始檔名與逐字稿片段，精修出一個更自然、清楚、適合當音檔名稱的繁體中文短檔名。

原始檔名主題：{original_stem}

規則：
1. 只輸出檔名本身，不要副檔名。
2. 長度 6 到 18 個中文字左右。
3. 不要使用 / \\ : * ? " < > | 等檔名禁用符號。
4. 原始檔名已經提供主題方向，請只做修飾、簡化或補明確性，不可以大幅改變主題。
5. 如果逐字稿與原始檔名方向不衝突，請優先保留原始檔名的核心詞。
6. 如果無法精修，請輸出原始檔名主題本身。

逐字稿片段：
{naming_context}
"""
    try:
        title = clean_ai_filename(generate_text(prompt, llm_provider))
    except Exception:
        title = ""

    if not title:
        return fallback_filename
    if respect_original_context and not is_generic_original and not has_filename_topic_overlap(original_stem, title):
        return fallback_filename

    filename = sanitize_filename(title, fallback=os.path.splitext(fallback_filename)[0], extension=extension)
    return ensure_unique_filename(filename, get_source_filenames(notebook_id), ignore_filename=current_filename)


def generate_analysis_title_filename(notebook_id, analysis, extension, fallback_filename, current_filename=None):
    global_summary = analysis.get("global_summary") if isinstance(analysis, dict) else {}
    if not isinstance(global_summary, dict):
        return ""

    title = clean_ai_filename(str(global_summary.get("overall_title") or ""))
    generic_titles = {"重點摘要", "錄音重點摘要", "目前無法產生整體摘要", "無法判斷"}
    if not title or title in generic_titles:
        return ""

    fallback_stem = os.path.splitext(fallback_filename or "audio")[0]
    filename = sanitize_filename(title, fallback=fallback_stem, extension=extension)
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


def suggest_source_filename(notebook_id, source_id, llm_provider=None):
    llm_provider = normalize_llm_provider(llm_provider)
    row = get_source_filename_suggestion_input(notebook_id, source_id)
    if not row:
        return {"status": "error", "message": "找不到指定來源。"}

    current_filename, transcript_text, analysis_json = row
    transcript_text = (transcript_text or "").strip()
    if not transcript_text:
        return {"status": "error", "message": "此來源沒有可用逐字稿，無法使用 AI 取檔名。"}

    extension = os.path.splitext(current_filename or "")[1] or ".webm"
    analysis = None
    if analysis_json:
        try:
            parsed_analysis = json.loads(analysis_json)
            if isinstance(parsed_analysis, dict):
                analysis = parsed_analysis
        except Exception:
            analysis = None

    suggested_filename = ""
    if analysis:
        suggested_filename = generate_analysis_title_filename(
            notebook_id,
            analysis,
            extension,
            current_filename or "audio",
            current_filename=current_filename,
        )

    if not suggested_filename:
        suggested_filename = generate_ai_audio_filename(
            notebook_id,
            transcript_text,
            extension,
            fallback_filename=current_filename,
            current_filename=current_filename,
            analysis=analysis,
            llm_provider=llm_provider,
            respect_original_context=False,
        )

    stop_local_llm_after_use(llm_provider)
    return {
        "status": "success",
        "filename": suggested_filename,
    }


def delete_source_data(notebook_id, source_id):
    source = get_source_delete_info(notebook_id, source_id)
    if not source:
        return {"status": "error", "message": "找不到指定來源。"}

    deleted_chunks = delete_source_from_index(notebook_id, source_id)
    deleted_record = delete_source_record(notebook_id, source_id)
    if not deleted_record:
        return {"status": "error", "message": "找不到指定來源。"}

    audio_result = delete_audio_file(source["filename"])
    return {
        "status": "success",
        "source": {
            "id": source_id,
            "filename": source["filename"]
        },
        "deleted_chunks": deleted_chunks,
        "audio_deleted": audio_result["deleted"],
        "audio_missing": audio_result["missing"]
    }


def delete_notebook_data(notebook_id):
    notebook = get_notebook_delete_info(notebook_id)
    if not notebook:
        return {"status": "error", "message": "Notebook not found"}

    delete_notebook_collection(notebook_id)

    audio_results = []
    for source in notebook["sources"]:
        audio_result = delete_audio_file(source["filename"])
        audio_results.append({
            "source_id": source["id"],
            "filename": source["filename"],
            "deleted": audio_result["deleted"],
            "missing": audio_result["missing"]
        })

    delete_notebook_record(notebook_id)
    return {
        "status": "success",
        "notebook": {
            "id": notebook_id,
            "name": notebook["name"]
        },
        "deleted_sources": len(notebook["sources"]),
        "audio_results": audio_results
    }


def check_data_consistency():
    sqlite_snapshot = get_data_consistency_snapshot()
    chroma_snapshot = cast(dict[str, Any], get_chroma_consistency_snapshot())
    audio_files = set(list_audio_upload_files())

    notebooks = sqlite_snapshot["notebooks"]
    sources = sqlite_snapshot["sources"]
    notebook_ids = {notebook["id"] for notebook in notebooks}
    source_ids = {source["id"] for source in sources}
    source_filenames = {os.path.basename(source["filename"] or "") for source in sources}
    chroma_collections = set(chroma_snapshot["collections"])
    source_chunks_by_notebook = cast(dict[str, dict[str, Any]], chroma_snapshot["source_chunks_by_notebook"])
    issues = []

    for source in sources:
        filename = os.path.basename(source["filename"] or "")
        audio_path = get_audio_path_for_filename(filename)
        if not audio_path or not os.path.exists(audio_path):
            issues.append({
                "type": "sqlite_source_missing_audio",
                "severity": "warning",
                "notebook_id": source["notebook_id"],
                "source_id": source["id"],
                "filename": source["filename"],
                "message": "SQLite 有來源紀錄，但 audio_uploads 找不到對應音檔。"
            })

        chunk_info = source_chunks_by_notebook.get(source["notebook_id"], {})
        notebook_chunks = cast(dict[str, int], chunk_info.get("source_counts", {}))
        if notebook_chunks.get(source["id"], 0) == 0:
            issues.append({
                "type": "sqlite_source_missing_chroma_chunks",
                "severity": "warning",
                "notebook_id": source["notebook_id"],
                "source_id": source["id"],
                "filename": source["filename"],
                "message": "SQLite 有來源紀錄，但 ChromaDB 沒有對應知識片段。"
            })

    for notebook in notebooks:
        collection_name = f"notebook_{notebook['id']}"
        if collection_name not in chroma_collections:
            issues.append({
                "type": "sqlite_notebook_missing_chroma_collection",
                "severity": "info",
                "notebook_id": notebook["id"],
                "notebook_name": notebook["name"],
                "collection": collection_name,
                "message": "SQLite 有筆記本紀錄，但 ChromaDB 沒有對應 collection。"
            })

    for collection_name in chroma_collections:
        notebook_id = collection_name.replace("notebook_", "", 1)
        if notebook_id not in notebook_ids:
            issues.append({
                "type": "orphan_chroma_collection",
                "severity": "warning",
                "notebook_id": notebook_id,
                "collection": collection_name,
                "message": "ChromaDB 有 collection，但 SQLite 沒有對應筆記本。"
            })

    for notebook_id, chunk_info in source_chunks_by_notebook.items():
        source_counts = cast(dict[str, int], chunk_info.get("source_counts", {}))
        for source_id, chunk_count in source_counts.items():
            if source_id not in source_ids:
                issues.append({
                    "type": "orphan_chroma_chunks",
                    "severity": "warning",
                    "notebook_id": notebook_id,
                    "source_id": source_id,
                    "chunk_count": chunk_count,
                    "message": "ChromaDB 有 source_id chunks，但 SQLite sources 找不到對應來源。"
                })
        missing_source_id_chunks = int(chunk_info.get("missing_source_id_chunks", 0))
        if missing_source_id_chunks:
            issues.append({
                "type": "chroma_chunks_missing_source_id",
                "severity": "warning",
                "notebook_id": notebook_id,
                "chunk_count": missing_source_id_chunks,
                "message": "ChromaDB 有 chunks 缺少 source_id metadata。"
            })

    for filename in sorted(audio_files - source_filenames):
        issues.append({
            "type": "orphan_audio_file",
            "severity": "info",
            "filename": filename,
            "message": "audio_uploads 有音檔，但 SQLite sources 沒有對應紀錄。"
        })

    for error in chroma_snapshot["errors"]:
        issues.append({
            "type": "chroma_collection_read_error",
            "severity": "warning",
            "collection": error["collection"],
            "message": error["message"]
        })

    return {
        "status": "ok" if not issues else "warning",
        "summary": {
            "notebooks": len(notebooks),
            "sources": len(sources),
            "audio_files": len(audio_files),
            "chroma_collections": len(chroma_collections),
            "issues": len(issues)
        },
        "issues": issues
    }


def clear_orphan_chroma_data():
    chroma_result = clear_orphan_chroma_segments()
    return {
        "status": "success",
        "needs_restart": bool(chroma_result["needs_restart"]),
        "summary": {
            "chroma_segment_dirs_deleted": len(chroma_result["deleted_segment_dirs"]),
            "chroma_segment_dirs_remaining": len(chroma_result["remaining_orphan_segment_dirs"]),
        },
        "chroma": chroma_result
    }


def stop_local_llm_after_use(llm_provider=None):
    if not is_ollama_provider(llm_provider):
        return
    try:
        stop_llm_generation("ollama")
    except Exception:
        pass


def transcribe_audio(file_path, llm_provider=None):
    if is_ollama_provider(llm_provider):
        try:
            stop_llm_generation("ollama")
            clear_cuda_memory()
        except Exception:
            pass

    try:
        whisper_model = get_whisper_model()
        segments, info = whisper_model.transcribe(
            file_path,
            beam_size=5,
            language="zh",
            initial_prompt="這是一段繁體中文的台灣口音逐字稿：",
            vad_filter=True,
            batch_size=get_whisper_batch_size(llm_provider)
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
        return transcript_text, timed_segments
    finally:
        if is_ollama_provider(llm_provider):
            unload_whisper_model()
            gc.collect()


def generate_summary(transcript_text, llm_provider=None):
    prompt = f"""
        你是一個專業的 AI 知識分析助手。請閱讀以下的口述語音逐字稿，
        進行深入的訊息分析，並給出一份精煉的「整體重點摘要」。
        請用流暢的段落來總結核心訊息，幫助讀者快速掌握整段語音的精華。
        
        ⚠️ 絕對要求：請務必使用「繁體中文 (Traditional Chinese)」輸出，嚴禁出現簡體字！
        
        語音逐字稿內容：\n{transcript_text}
        """
    return generate_text(prompt, llm_provider)


def process_audio_upload(notebook_id, file, auto_filename=False, fallback_filename=None, llm_provider=None):
    llm_provider = normalize_llm_provider(llm_provider)
    source_id = str(uuid.uuid4())
    original_ext = os.path.splitext(file.filename or "")[1] or ".webm"
    if auto_filename:
        upload_filename = ensure_unique_filename(
            sanitize_filename(
                fallback_filename or file.filename or f"recording-{source_id[:8]}{original_ext}",
                fallback="recording",
                extension=original_ext
            ),
            get_source_filenames(notebook_id)
        )
    else:
        upload_filename = ensure_unique_filename(
            sanitize_filename(file.filename, fallback="audio", extension=original_ext),
            get_source_filenames(notebook_id)
        )
    file_path = save_uploaded_audio(file, upload_filename)

    try:
        transcript_text, timed_segments = transcribe_audio(file_path, llm_provider)
    except Exception as e:
        return {"status": "error", "message": f"語音辨識失敗: {str(e)}"}

    filename = upload_filename
    analysis_mode = "deep" if should_use_deep_analysis(transcript_text, timed_segments) else "quick"
    try:
        chunks_count = index_source_transcript(notebook_id, source_id, filename, transcript_text, timed_segments, llm_provider)
        source_saved_at = insert_source_record(
            notebook_id,
            source_id,
            filename,
            transcript_text,
            json.dumps(timed_segments, ensure_ascii=False),
            analysis_mode,
            "pending",
            None
        )
    except Exception as e:
        stop_local_llm_after_use(llm_provider)
        return {"status": "error", "message": f"寫入向量資料庫失敗: {str(e)}"}
    finally:
        gc.collect()

    stop_local_llm_after_use(llm_provider)

    return {
        "filename": filename,
        "status": "success",
        "message": f"音檔已上傳並建立 {chunks_count} 個逐字稿索引片段。",
        "raw_transcript": transcript_text,
        "structured_knowledge": "",
        "suggested_questions": [],
        "analysis_mode": analysis_mode,
        "analysis_status": "pending",
        "source": {
            "id": source_id,
            "filename": filename,
            "added_at": source_saved_at,
            "has_transcript": True,
            "transcript_updated_at": source_saved_at,
            "indexed_at": source_saved_at,
            "analysis_mode": analysis_mode,
            "analysis_status": "pending",
            "analysis_updated_at": source_saved_at
        }
    }

def reanalyze_source_data(notebook_id, source_id, llm_provider=None):
    llm_provider = normalize_llm_provider(llm_provider)
    row = get_source_analysis_input(notebook_id, source_id)
    if not row:
        return {"status": "error", "message": "找不到指定來源。"}

    filename, transcript_text, timed_segments_json = row
    transcript_text = (transcript_text or "").strip()
    if not transcript_text:
        return {"status": "error", "message": "此來源沒有可用逐字稿，請重新上傳音檔。"}

    try:
        timed_segments = json.loads(timed_segments_json) if timed_segments_json else []
        analysis = build_source_analysis(filename, transcript_text, timed_segments, llm_provider)
        analysis_chunk_count = index_source_analysis_documents(notebook_id, source_id, filename, analysis, llm_provider)
        analysis_json = json.dumps(
            {key: value for key, value in analysis.items() if key != "structured_knowledge"},
            ensure_ascii=False
        )
        updated_at = update_source_analysis_record(
            notebook_id,
            source_id,
            analysis.get("mode"),
            "completed",
            analysis_json
        )
        saved_suggested_questions = save_suggested_questions(
            notebook_id,
            filename,
            cast(list[str], analysis.get("suggested_questions") or [])
        )
        insert_ai_summary_message(notebook_id, filename, str(analysis.get("structured_knowledge", "")))
        stop_local_llm_after_use(llm_provider)
    except Exception as e:
        try:
            update_source_analysis_record(notebook_id, source_id, None, "failed", "")
        except Exception:
            pass
        stop_local_llm_after_use(llm_provider)
        return {"status": "error", "message": f"重新深度分析失敗: {str(e)}"}

    return {
        "status": "success",
        "message": f"已重新產生分析，並建立 {analysis_chunk_count} 個摘要片段。",
        "structured_knowledge": analysis.get("structured_knowledge", ""),
        "suggested_questions": saved_suggested_questions,
        "analysis_mode": analysis.get("mode"),
        "analysis_status": "completed",
        "analysis_chunk_count": analysis_chunk_count,
        "source": {
            "id": source_id,
            "filename": filename,
            "has_transcript": True,
            "analysis_mode": analysis.get("mode"),
            "analysis_status": "completed",
            "analysis_updated_at": updated_at
        }
    }
