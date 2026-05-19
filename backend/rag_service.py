import json
import os
import re
import sqlite3
import uuid
from collections.abc import Iterable
from typing import Any, cast

import chromadb
import jieba
import numpy as np
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from db import (
    append_qa_messages,
    get_recent_messages,
    get_source_update_info,
    save_suggested_questions,
    update_source_transcript_record,
)
from models import (
    EMBEDDING_BATCH_SIZE,
    RERANKER_BATCH_SIZE,
    clear_cuda_memory,
    get_embeddings_model,
    get_reranker,
    unload_whisper_model,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)


Metadata = dict[str, Any]
OLLAMA_LLM_MODEL = "qwen3.5:9b-q4_K_M"
REFERENCE_MAX_COUNT = 2
REFERENCE_SCORE_MARGIN = 1.25
REFERENCE_MIN_QUESTION_OVERLAP_SCORE = 1.3
REFERENCE_MIN_ANSWER_OVERLAP_SCORE = 2.5

REFERENCE_STOPWORDS = {
    "這個", "這些", "那個", "哪些", "如何", "為何", "為什麼", "什麼", "是否",
    "可以", "目前", "根據", "資料", "資料庫", "錄音", "紀錄", "內容", "回答",
    "來源", "參考", "問題", "使用者", "相關", "說明", "指出", "提到", "表示",
    "以及", "並且", "如果", "因為", "所以", "其中", "此外", "仍然", "需要",
}

NO_ANSWER_PHRASES = (
    "根據目前資料庫的錄音紀錄，並未提及此資訊",
    "根據目前資料庫的錄音紀錄，並未提及",
    "目前資料庫的錄音紀錄並未提及",
    "資料庫的錄音紀錄並未提及",
    "並未提及此資訊",
)
NO_ANSWER_MESSAGE = "根據目前資料庫的錄音紀錄，並未提及此資訊。"
ANSWER_CONTEXT_MIN_COUNT = 3
ANSWER_CONTEXT_MAX_COUNT = 5
ANSWER_CONTEXT_SCORE_MARGIN = 1.0
ANSWERABILITY_STRONG_BEST_OVERLAP = 3.5
ANSWERABILITY_STRONG_TOTAL_OVERLAP = 6.5

ANALYSIS_VERSION = "1.0"
LONG_AUDIO_SECONDS = 10 * 60
LONG_TRANSCRIPT_CHARS = 5000
CHAPTER_TARGET_SECONDS = 4 * 60
CHAPTER_MIN_SECONDS = 3 * 60
CHAPTER_MAX_SECONDS = 5 * 60
SUMMARY_DOC_TYPES = {"global_summary", "chapter_summary"}
TRANSCRIPT_DOC_TYPE = "transcript_chunk"
GLOBAL_QUESTION_KEYWORDS = (
    "摘要", "總結", "重點", "大意", "主題", "核心", "整體", "全篇", "全段", "這份",
    "這段", "錄音主要", "內容主要", "深度解析", "解析", "介紹", "潤色", "改寫",
    "心得", "架構", "脈絡", "整理", "統整", "說明這個", "說明一下",
)


def get_notebook_collection(notebook_id: str) -> Any:
    return chroma_client.get_or_create_collection(name=f"notebook_{notebook_id}")


# 每個筆記本維護一個獨立的 BM25 索引，存在記憶體中
bm25_indices: dict[str, Any] = {}   # {notebook_id: BM25Okapi 實例}
bm25_docs: dict[str, list[str]] = {}      # {notebook_id: [原始文件列表]}
bm25_metas: dict[str, list[Metadata]] = {}     # {notebook_id: [metadata 列表]}


def tokenize_chinese(text: str) -> list[str]:
    """jieba 中文分詞，過濾空白"""
    return [w for w in jieba.cut(text) if w.strip()]


def get_meaningful_tokens(text: str) -> set[str]:
    """取出適合用來判斷引用相關性的關鍵詞，降低常見虛詞干擾。"""
    normalized_text = re.sub(r"https?://\S+", " ", text or "")
    tokens: set[str] = set()
    for token in tokenize_chinese(normalized_text):
        cleaned = re.sub(r"[^\w\u4e00-\u9fff]", "", token).lower().strip()
        if len(cleaned) < 2:
            continue
        if cleaned in REFERENCE_STOPWORDS:
            continue
        tokens.add(cleaned)
    return tokens


def score_token_overlap(tokens: set[str]) -> float:
    """讓較具體的長詞略高分，作為 reranker 之外的輔助訊號。"""
    return sum(1.0 + min(len(token), 6) * 0.15 for token in tokens)


def normalize_reranker_scores(raw_scores: Any) -> list[float]:
    if isinstance(raw_scores, (int, float)):
        return [float(raw_scores)]
    return [float(score) for score in cast(Iterable[Any], raw_scores)]


def is_no_answer_response(answer: str) -> bool:
    """判斷整段回答是否屬於「資料庫沒有答案」，避免顯示硬湊的來源。"""
    compact_answer = re.sub(r"\s+", "", answer or "")
    return any(phrase in compact_answer for phrase in NO_ANSWER_PHRASES)


def stop_ollama_model() -> dict[str, Any]:
    """嘗試立即卸載目前 LLM，作為停止回答的簡單版後端中止機制。"""
    try:
        ollama.generate(model=OLLAMA_LLM_MODEL, prompt="", keep_alive=0)
        return {
            "status": "success",
            "message": "已送出停止 Ollama 模型的請求。",
            "model": OLLAMA_LLM_MODEL
        }
    except Exception as error:
        return {
            "status": "error",
            "message": f"停止 Ollama 模型失敗：{error}",
            "model": OLLAMA_LLM_MODEL
        }


def normalize_text_list(value: Any, limit: int | None = None) -> list[str]:
    if isinstance(value, list):
        items = [str(item).strip() for item in value]
    elif isinstance(value, str):
        items = [
            re.sub(r"^\s*[-*•\d.、)）]+\s*", "", line).strip()
            for line in value.splitlines()
        ]
    else:
        items = []

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        cleaned.append(item)
        if limit is not None and len(cleaned) >= limit:
            break
    return cleaned


def extract_json_object(raw_text: str) -> dict[str, Any]:
    cleaned = (raw_text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = re.sub(r"^(json|JSON)\s*", "", cleaned).strip()

    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        return {}

    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def get_transcript_duration(timed_segments: list[dict[str, Any]]) -> float:
    duration = 0.0
    for segment in timed_segments or []:
        try:
            duration = max(duration, float(segment.get("end", 0) or 0))
        except Exception:
            continue
    return duration


def should_use_deep_analysis(transcript_text: str, timed_segments: list[dict[str, Any]]) -> bool:
    return (
        get_transcript_duration(timed_segments) >= LONG_AUDIO_SECONDS
        or len((transcript_text or "").strip()) >= LONG_TRANSCRIPT_CHARS
    )


def build_chapter_inputs(transcript_text: str, timed_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if timed_segments:
        chapters: list[dict[str, Any]] = []
        current_segments: list[dict[str, Any]] = []
        current_start: float | None = None

        for segment in timed_segments:
            text = str(segment.get("text", "")).strip()
            if not text:
                continue
            try:
                start = float(segment.get("start", 0) or 0)
                end = float(segment.get("end", start) or start)
            except Exception:
                start = 0.0
                end = start

            if current_start is None:
                current_start = start

            current_segments.append({"text": text, "start": start, "end": end})
            current_duration = end - current_start
            if current_duration >= CHAPTER_TARGET_SECONDS or current_duration >= CHAPTER_MAX_SECONDS:
                if current_duration >= CHAPTER_MIN_SECONDS:
                    chapter_text = "".join(item["text"] for item in current_segments).strip()
                    chapters.append({
                        "index": len(chapters) + 1,
                        "start_time": current_start,
                        "end_time": end,
                        "time_range": get_time_range_text(current_start, end),
                        "text": chapter_text
                    })
                    current_segments = []
                    current_start = None

        if current_segments:
            start = current_start if current_start is not None else float(current_segments[0]["start"])
            end = float(current_segments[-1]["end"])
            chapter_text = "".join(item["text"] for item in current_segments).strip()
            if chapter_text:
                if chapters and end - start < 90:
                    previous = chapters[-1]
                    previous["text"] = f"{previous['text']}{chapter_text}"
                    previous["end_time"] = end
                    previous["time_range"] = get_time_range_text(previous["start_time"], end)
                else:
                    chapters.append({
                        "index": len(chapters) + 1,
                        "start_time": start,
                        "end_time": end,
                        "time_range": get_time_range_text(start, end),
                        "text": chapter_text
                    })
        if chapters:
            return chapters

    chunks = semantic_chunk(transcript_text, similarity_threshold=0.62, max_chunk_size=3500, min_chunk_size=600)
    return [
        {
            "index": index,
            "start_time": -1.0,
            "end_time": -1.0,
            "time_range": "時間未記錄",
            "text": chunk
        }
        for index, chunk in enumerate(chunks, start=1)
    ]


def generate_short_summary(transcript_text: str) -> str:
    prompt = f"""
你是一個專業的 AI 知識分析助手。請閱讀以下的口述語音逐字稿，
進行深入的訊息分析，並給出一份精煉的「整體重點摘要」。

輸出格式請嚴格遵守：
【整體重點摘要】

接著只寫一段流暢的摘要文字，幫助讀者快速掌握整段語音的精華。

規則：
1. 請務必使用繁體中文。
2. 不要使用編號清單。
3. 不要使用項目符號。
4. 不要輸出「核心主題」、「深度解析」、「重要細節」、「分段重點」等額外區塊。
5. 不要輸出 Markdown 標題。

語音逐字稿內容：
{transcript_text}
"""
    response = ollama.chat(model=OLLAMA_LLM_MODEL, messages=[{"role": "user", "content": prompt + "\n/no_think"}])
    return response["message"]["content"]


def analyze_chapter_with_llm(chapter: dict[str, Any]) -> dict[str, Any]:
    prompt = f"""
你是一個專業的長音檔知識整理助手。請分析下方這一段逐字稿，並只輸出 JSON 物件。

JSON 欄位固定如下：
{{
  "title": "本段標題，12 字以內",
  "summary": "本段摘要，使用繁體中文，2 到 4 句",
  "key_points": ["重點一", "重點二", "重點三"],
  "keywords": ["關鍵詞一", "關鍵詞二"],
  "questions": ["可延伸問題一？"]
}}

規則：
1. 必須使用繁體中文。
2. 不要輸出 Markdown，不要輸出 JSON 以外的文字。
3. 只根據逐字稿內容整理，不要自行補充外部知識。

段落時間：{chapter["time_range"]}
逐字稿：
{chapter["text"]}
"""
    response = ollama.chat(model=OLLAMA_LLM_MODEL, messages=[{"role": "user", "content": prompt + "\n/no_think"}])
    parsed = extract_json_object(response["message"]["content"])
    title = str(parsed.get("title") or f"第 {chapter['index']} 段重點").strip()
    summary = str(parsed.get("summary") or response["message"]["content"]).strip()
    return {
        "index": chapter["index"],
        "title": title[:40],
        "time_range": chapter["time_range"],
        "start_time": float(chapter.get("start_time", -1.0)),
        "end_time": float(chapter.get("end_time", -1.0)),
        "summary": summary,
        "key_points": normalize_text_list(parsed.get("key_points"), limit=5),
        "keywords": normalize_text_list(parsed.get("keywords"), limit=8),
        "questions": normalize_text_list(parsed.get("questions"), limit=3),
    }


def generate_global_analysis(chapters: list[dict[str, Any]]) -> dict[str, Any]:
    chapter_payload = [
        {
            "index": chapter["index"],
            "title": chapter["title"],
            "time_range": chapter["time_range"],
            "summary": chapter["summary"],
            "key_points": chapter["key_points"],
            "keywords": chapter["keywords"],
        }
        for chapter in chapters
    ]
    prompt = f"""
你是一個 NotebookLM 風格的知識摘要助手。請根據多個段落分析結果，產生整份錄音的全局理解，並只輸出 JSON 物件。

JSON 欄位固定如下：
{{
  "overall_title": "整份錄音標題",
  "overview": "整體摘要，3 到 5 句",
  "core_themes": ["核心主題一", "核心主題二", "核心主題三"],
  "deep_insights": ["深度解析一", "深度解析二", "深度解析三"],
  "important_details": ["重要細節一", "重要細節二"],
  "suggested_questions": ["推薦問題一？", "推薦問題二？", "推薦問題三？"]
}}

規則：
1. 必須使用繁體中文。
2. 不要輸出 Markdown，不要輸出 JSON 以外的文字。
3. 摘要要重視全局脈絡，不要只挑其中一段。
4. 推薦問題最多 3 題，且要能幫助使用者理解整份內容。

段落分析結果：
{json.dumps(chapter_payload, ensure_ascii=False)}
"""
    response = ollama.chat(model=OLLAMA_LLM_MODEL, messages=[{"role": "user", "content": prompt + "\n/no_think"}])
    parsed = extract_json_object(response["message"]["content"])
    return {
        "overall_title": str(parsed.get("overall_title") or "錄音重點摘要").strip(),
        "overview": str(parsed.get("overview") or response["message"]["content"]).strip(),
        "core_themes": normalize_text_list(parsed.get("core_themes"), limit=5),
        "deep_insights": normalize_text_list(parsed.get("deep_insights"), limit=5),
        "important_details": normalize_text_list(parsed.get("important_details"), limit=6),
        "suggested_questions": normalize_text_list(parsed.get("suggested_questions"), limit=3),
    }


def format_analysis_summary(analysis: dict[str, Any]) -> str:
    global_summary = cast(dict[str, Any], analysis.get("global_summary") or {})
    overview = str(global_summary.get("overview") or "").strip()
    if overview:
        return overview

    return str(global_summary.get("overall_title") or "目前無法產生整體摘要。").strip()


def build_source_analysis(filename: str, transcript_text: str, timed_segments: list[dict[str, Any]]) -> dict[str, Any]:
    mode = "deep" if should_use_deep_analysis(transcript_text, timed_segments) else "quick"
    duration_seconds = get_transcript_duration(timed_segments)

    if mode == "quick":
        summary = generate_short_summary(transcript_text)
        suggested_questions = generate_suggested_questions(transcript_text)
        analysis = {
            "version": ANALYSIS_VERSION,
            "mode": "quick",
            "filename": filename,
            "duration_seconds": duration_seconds,
            "chapters": [],
            "global_summary": {
                "overall_title": "重點摘要",
                "overview": summary,
                "core_themes": [],
                "deep_insights": [],
                "important_details": [],
                "suggested_questions": suggested_questions,
            },
            "suggested_questions": suggested_questions,
        }
        analysis["structured_knowledge"] = format_analysis_summary(analysis)
        return analysis

    chapter_inputs = build_chapter_inputs(transcript_text, timed_segments)
    chapters = [analyze_chapter_with_llm(chapter) for chapter in chapter_inputs]
    global_summary = generate_global_analysis(chapters)
    suggested_questions = normalize_text_list(global_summary.get("suggested_questions"), limit=3)
    if not suggested_questions:
        suggested_questions = normalize_text_list(
            [question for chapter in chapters for question in chapter.get("questions", [])],
            limit=3
        )

    analysis = {
        "version": ANALYSIS_VERSION,
        "mode": "deep",
        "filename": filename,
        "duration_seconds": duration_seconds,
        "chapters": chapters,
        "global_summary": global_summary,
        "suggested_questions": suggested_questions,
    }
    analysis["structured_knowledge"] = format_analysis_summary(analysis)
    return analysis


def rebuild_bm25_index(notebook_id: str) -> None:
    """從 ChromaDB 重建指定筆記本的 BM25 索引"""
    try:
        collection = get_notebook_collection(notebook_id)
        if collection.count() == 0:
            clear_bm25_index(notebook_id)
            return
        all_data = cast(dict[str, Any], collection.get(include=["documents", "metadatas"]))
        docs = cast(list[str], all_data.get("documents") or [])
        if not docs:
            clear_bm25_index(notebook_id)
            return
        metadatas = cast(list[Metadata], all_data.get("metadatas") or [{} for _ in docs])
        bm25_docs[notebook_id] = docs
        bm25_metas[notebook_id] = metadatas
        tokenized = [tokenize_chinese(doc) for doc in docs]
        bm25_indices[notebook_id] = BM25Okapi(tokenized)
    except Exception:
        pass


def clear_bm25_index(notebook_id: str) -> None:
    bm25_indices.pop(notebook_id, None)
    bm25_docs.pop(notebook_id, None)
    bm25_metas.pop(notebook_id, None)


def get_collection_segment_ids(collection_name: str) -> list[str]:
    db_path = os.path.join(CHROMA_PATH, "chroma.sqlite3")
    if not os.path.exists(db_path):
        return []

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT s.id
            FROM segments s
            JOIN collections c ON s.collection = c.id
            WHERE c.name = ?
            """,
            (collection_name,)
        )
        segment_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        return segment_ids
    except Exception:
        return []


def delete_chroma_segment_directory(segment_id: str) -> bool:
    try:
        uuid.UUID(segment_id)
    except Exception:
        return False

    root = os.path.abspath(CHROMA_PATH)
    segment_path = os.path.abspath(os.path.join(root, segment_id))
    if os.path.commonpath([root, segment_path]) != root:
        return False
    if not os.path.isdir(segment_path):
        return False

    allowed_files = {"data_level0.bin", "header.bin", "length.bin", "link_lists.bin"}
    children = os.listdir(segment_path)
    for child in children:
        child_path = os.path.join(segment_path, child)
        if os.path.isdir(child_path) or child not in allowed_files:
            return False

    try:
        for child in children:
            os.remove(os.path.join(segment_path, child))
        os.rmdir(segment_path)
        return True
    except OSError:
        return False


def get_all_chroma_segment_ids() -> list[str]:
    db_path = os.path.join(CHROMA_PATH, "chroma.sqlite3")
    if not os.path.exists(db_path):
        return []

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM segments")
        segment_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        return segment_ids
    except Exception:
        return []


def delete_orphan_chroma_segment_directories() -> list[str]:
    active_segment_ids = set(get_all_chroma_segment_ids())
    deleted_dirs: list[str] = []
    root = os.path.abspath(CHROMA_PATH)
    if not os.path.isdir(root):
        return deleted_dirs

    for name in os.listdir(root):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        try:
            uuid.UUID(name)
        except Exception:
            continue
        if name in active_segment_ids:
            continue
        try:
            if delete_chroma_segment_directory(name):
                deleted_dirs.append(name)
        except Exception:
            pass

    return deleted_dirs


def get_orphan_chroma_segment_directories() -> list[str]:
    active_segment_ids = set(get_all_chroma_segment_ids())
    orphan_dirs: list[str] = []
    root = os.path.abspath(CHROMA_PATH)
    if not os.path.isdir(root):
        return orphan_dirs

    for name in os.listdir(root):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        try:
            uuid.UUID(name)
        except Exception:
            continue
        if name not in active_segment_ids:
            orphan_dirs.append(name)

    return orphan_dirs


def clear_orphan_chroma_segments() -> dict[str, Any]:
    deleted_segment_dirs = delete_orphan_chroma_segment_directories()
    remaining_orphan_segment_dirs = get_orphan_chroma_segment_directories()
    return {
        "deleted_segment_dirs": deleted_segment_dirs,
        "remaining_orphan_segment_dirs": remaining_orphan_segment_dirs,
        "needs_restart": bool(remaining_orphan_segment_dirs)
    }


def delete_notebook_collection(notebook_id: str) -> None:
    collection_name = f"notebook_{notebook_id}"
    segment_ids = get_collection_segment_ids(collection_name)
    deleted_collection = False
    try:
        chroma_client.delete_collection(name=collection_name)
        deleted_collection = True
    except Exception:
        pass
    if deleted_collection:
        for segment_id in segment_ids:
            try:
                delete_chroma_segment_directory(segment_id)
            except Exception:
                pass
    delete_orphan_chroma_segment_directories()
    clear_bm25_index(notebook_id)


def rrf_fusion(dense_docs: list[str], bm25_result_docs: list[str], k: int = 60) -> list[str]:
    """使用 Reciprocal Rank Fusion (RRF) 融合兩組排名結果"""
    scores: dict[str, float] = {}
    for rank, doc in enumerate(dense_docs):
        scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank + 1)
    for rank, doc in enumerate(bm25_result_docs):
        scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank + 1)
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in sorted_docs]


def split_sentences(text):
    """將中文文本按句子分割"""
    sentences = re.split(r'(?<=[\u3002\uff01\uff1f\uff1b\u000a])', text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]


def split_sentences_with_offsets(text: str) -> list[dict[str, Any]]:
    """將文字切成句子並保留在原文中的位置，用於引用證據片段。"""
    sentences: list[dict[str, Any]] = []
    for match in re.finditer(r'[^。！？；\n]+[。！？；\n]?', text):
        sentence = match.group(0).strip()
        if len(sentence) <= 5:
            continue
        sentence_start = match.start() + len(match.group(0)) - len(match.group(0).lstrip())
        sentence_end = match.end()
        sentences.append({
            "text": sentence,
            "start": sentence_start,
            "end": sentence_end
        })

    if not sentences and text.strip():
        stripped = text.strip()
        stripped_start = text.find(stripped)
        sentences.append({
            "text": stripped,
            "start": stripped_start,
            "end": stripped_start + len(stripped)
        })

    return sentences


def score_reference_sentence(question_tokens: set[str], sentence: str) -> float:
    sentence_tokens = set(tokenize_chinese(sentence))
    if not sentence_tokens:
        return 0.0
    overlap = question_tokens & sentence_tokens
    return len(overlap) + sum(len(token) for token in overlap) * 0.1


def select_reference_evidence(question: str, doc: str, metadata: Metadata) -> dict[str, Any]:
    """從檢索 chunk 中挑出較短的證據句，保留後續引用追溯需要的片段資訊。"""
    try:
        chunk_char_start = int(metadata.get("char_start", -1)) if metadata else -1
    except Exception:
        chunk_char_start = -1
    try:
        chunk_char_end = int(metadata.get("char_end", -1)) if metadata else -1
    except Exception:
        chunk_char_end = -1
    if not doc:
        return {
            "char_start": chunk_char_start,
            "char_end": chunk_char_end,
            "excerpt": ""
        }

    sentences = split_sentences_with_offsets(doc)
    if not sentences:
        return {
            "char_start": chunk_char_start,
            "char_end": chunk_char_end,
            "excerpt": doc[:220]
        }

    question_tokens = set(tokenize_chinese(question))
    best_sentence = max(
        sentences,
        key=lambda item: score_reference_sentence(question_tokens, item["text"])
    )

    selected_start = best_sentence["start"]
    selected_end = best_sentence["end"]
    selected_text = best_sentence["text"]

    # 句子太短時，補上下一句，讓證據片段更完整；但避免變成整個大 chunk。
    if len(selected_text) < 80:
        sentence_index = sentences.index(best_sentence)
        if sentence_index + 1 < len(sentences):
            next_sentence = sentences[sentence_index + 1]
            candidate_text = doc[selected_start:next_sentence["end"]].strip()
            if len(candidate_text) <= 220:
                selected_end = next_sentence["end"]
                selected_text = candidate_text

    if len(selected_text) > 260:
        selected_end = selected_start + 260
        selected_text = doc[selected_start:selected_end].strip()

    if chunk_char_start >= 0:
        return {
            "char_start": chunk_char_start + selected_start,
            "char_end": chunk_char_start + selected_end,
            "excerpt": selected_text
        }

    return {
        "char_start": -1,
        "char_end": -1,
        "excerpt": selected_text
    }


def build_reference_candidate(question: str, doc: str, metadata: Metadata, reranker_score: float) -> dict[str, Any]:
    label = get_reference_label(metadata)
    evidence = select_reference_evidence(question, doc, metadata)
    return {
        "doc": doc,
        "reranker_score": reranker_score,
        "reference": {
            "label": label,
            "source_id": str(metadata.get("source_id", "")) if metadata else "",
            "source": metadata.get("source", "未知來源") if metadata else "未知來源",
            "doc_type": metadata.get("doc_type", TRANSCRIPT_DOC_TYPE) if metadata else TRANSCRIPT_DOC_TYPE,
            "chunk_index": metadata.get("chunk_index", -1) if metadata else -1,
            "time_range": metadata.get("time_range", "時間未記錄") if metadata else "時間未記錄",
            "char_start": evidence["char_start"],
            "char_end": evidence["char_end"],
            "excerpt": evidence["excerpt"]
        }
    }


def filter_relevant_references(
    question: str,
    answer: str,
    candidates: list[dict[str, Any]],
    limit: int = REFERENCE_MAX_COUNT
) -> list[dict[str, Any]]:
    """只保留真正支撐問題與答案的來源，避免把次相關檢索結果列給使用者。"""
    if is_no_answer_response(answer):
        return []

    question_tokens = get_meaningful_tokens(question)
    answer_tokens = get_meaningful_tokens(answer)
    if not question_tokens or not answer_tokens:
        return []

    prepared_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        doc = str(candidate.get("doc", ""))
        reference = cast(dict[str, Any], candidate.get("reference", {}))
        if not reference.get("source_id"):
            continue

        doc_tokens = get_meaningful_tokens(doc)
        if not doc_tokens:
            continue

        question_overlap = doc_tokens & question_tokens
        answer_overlap = doc_tokens & answer_tokens
        question_overlap_score = score_token_overlap(question_overlap)
        answer_overlap_score = score_token_overlap(answer_overlap)
        if (
            question_overlap_score < REFERENCE_MIN_QUESTION_OVERLAP_SCORE
            or answer_overlap_score < REFERENCE_MIN_ANSWER_OVERLAP_SCORE
        ):
            continue

        prepared_candidates.append({
            "doc": doc,
            "reference": reference,
            "question_overlap_score": question_overlap_score,
            "answer_overlap_score": answer_overlap_score
        })

    if not prepared_candidates:
        return []

    reference_query = f"問題：{question}\n回答：{answer}"
    pairs: list[tuple[str, str]] = [(reference_query, item["doc"]) for item in prepared_candidates]
    try:
        semantic_scores = normalize_reranker_scores(
            get_reranker().predict(cast(Any, pairs), batch_size=RERANKER_BATCH_SIZE)
        )
    except Exception:
        semantic_scores = [
            item["question_overlap_score"] + item["answer_overlap_score"]
            for item in prepared_candidates
        ]

    scored_candidates: list[tuple[float, dict[str, Any]]] = []
    for candidate, semantic_score in zip(prepared_candidates, semantic_scores):
        score = (
            float(semantic_score)
            + float(candidate["question_overlap_score"]) * 0.15
            + float(candidate["answer_overlap_score"]) * 0.2
        )
        scored_candidates.append((score, cast(dict[str, Any], candidate["reference"])))

    if not scored_candidates:
        return []

    best_score = max(score for score, _ in scored_candidates)
    best_by_source: dict[str, tuple[float, dict[str, Any]]] = {}
    for relevance_score, reference in scored_candidates:
        if relevance_score < best_score - REFERENCE_SCORE_MARGIN:
            continue
        source_key = str(reference["source_id"])
        previous = best_by_source.get(source_key)
        if previous is None or relevance_score > previous[0]:
            best_by_source[source_key] = (relevance_score, reference)

    ranked_references = sorted(best_by_source.values(), key=lambda item: item[0], reverse=True)
    return [reference for _, reference in ranked_references[:limit]]


def semantic_chunk(text, similarity_threshold=0.65, max_chunk_size=600, min_chunk_size=80):
    """基於語意相似度的智慧切分。"""
    sentences = split_sentences(text)

    if len(sentences) <= 2:
        return [text] if text.strip() else []

    sentence_embeddings = get_embeddings_model().encode(sentences, batch_size=EMBEDDING_BATCH_SIZE)

    similarities = []
    for i in range(len(sentence_embeddings) - 1):
        vec_a = sentence_embeddings[i]
        vec_b = sentence_embeddings[i + 1]
        cos_sim = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        similarities.append(float(cos_sim))

    chunks = []
    current_chunk = sentences[0]

    for i in range(len(similarities)):
        next_sentence = sentences[i + 1]
        should_split = False

        if similarities[i] < similarity_threshold:
            should_split = True

        if len(current_chunk) + len(next_sentence) > max_chunk_size:
            should_split = True

        if should_split and len(current_chunk) >= min_chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = next_sentence
        else:
            current_chunk += next_sentence

    if current_chunk.strip():
        if len(current_chunk.strip()) < min_chunk_size and chunks:
            chunks[-1] += current_chunk.strip()
        else:
            chunks.append(current_chunk.strip())

    return chunks if chunks else [text]


def format_timestamp(seconds):
    """將秒數格式化為 MM:SS 或 HH:MM:SS"""
    if seconds is None:
        return None

    total_seconds = max(0, round(float(seconds)))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def get_time_range_text(start_time, end_time):
    """回傳適合顯示的時間區間文字"""
    start_text = format_timestamp(start_time)
    end_text = format_timestamp(end_time)
    if start_text and end_text:
        return f"{start_text}-{end_text}"
    return "時間未記錄"


def get_time_for_char_position(position, segment_ranges, edge="start"):
    """依文字位置推估對應的 Whisper segment 時間"""
    if not segment_ranges:
        return None

    for item in segment_ranges:
        if item["char_start"] <= position < item["char_end"]:
            return item["start"] if edge == "start" else item["end"]

    if position <= segment_ranges[0]["char_start"]:
        return segment_ranges[0]["start"] if edge == "start" else segment_ranges[0]["end"]
    return segment_ranges[-1]["start"] if edge == "start" else segment_ranges[-1]["end"]


def build_chunk_metadata(chunks, timed_segments, source_filename, source_id=None):
    """將 chunk 依文字位置對應回 Whisper 時間戳，供來源引用使用"""
    transcript_text = "".join(segment["text"] for segment in timed_segments)
    segment_ranges = []
    char_position = 0
    for segment in timed_segments:
        text = segment["text"]
        char_start = char_position
        char_end = char_start + len(text)
        segment_ranges.append({
            "char_start": char_start,
            "char_end": char_end,
            "start": segment["start"],
            "end": segment["end"]
        })
        char_position = char_end

    metadata_list = []
    search_position = 0
    for i, chunk in enumerate(chunks):
        chunk_text = chunk.strip()
        found_at = transcript_text.find(chunk_text, search_position) if chunk_text else -1
        if found_at == -1:
            found_at = search_position
        chunk_end = min(found_at + len(chunk_text), len(transcript_text))
        start_time = get_time_for_char_position(found_at, segment_ranges, edge="start")
        end_time = get_time_for_char_position(max(found_at, chunk_end - 1), segment_ranges, edge="end")
        metadata_list.append({
            "source": source_filename,
            "source_id": source_id or "",
            "doc_type": TRANSCRIPT_DOC_TYPE,
            "chunk_index": i,
            "char_start": found_at,
            "char_end": chunk_end,
            "start_time": float(start_time) if start_time is not None else -1.0,
            "end_time": float(end_time) if end_time is not None else -1.0,
            "time_range": get_time_range_text(start_time, end_time)
        })
        search_position = chunk_end

    return metadata_list


def chunk_transcript_text(transcript_text):
    """依目前 RAG 規則切分逐字稿，供上傳與重新索引共用"""
    if len(transcript_text) > 200:
        return semantic_chunk(
            transcript_text,
            similarity_threshold=0.65,
            max_chunk_size=600,
            min_chunk_size=80
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", "，", " "]
    )
    return text_splitter.split_text(transcript_text)


def delete_source_chunks(collection, source_id):
    """刪除指定來源的舊向量片段，不直接操作 ChromaDB 檔案"""
    deleted_count = count_source_chunks(collection, source_id)
    try:
        collection.delete(where={"source_id": source_id})
        return deleted_count
    except Exception:
        pass

    try:
        existing = collection.get(include=["metadatas"])
        ids_to_delete = [
            doc_id
            for doc_id, metadata in zip(existing.get("ids", []), existing.get("metadatas", []))
            if metadata and metadata.get("source_id") == source_id
        ]
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
    except Exception:
        pass
    return deleted_count


def count_source_chunks(collection, source_id):
    try:
        existing = cast(dict[str, Any], collection.get(
            where={"source_id": source_id},
            include=["metadatas"]
        ))
        return len(existing.get("ids") or [])
    except Exception:
        pass

    try:
        existing = cast(dict[str, Any], collection.get(include=["metadatas"]))
        return sum(
            1
            for metadata in existing.get("metadatas", [])
            if metadata and metadata.get("source_id") == source_id
        )
    except Exception:
        return 0


def delete_source_from_index(notebook_id, source_id):
    collection = get_notebook_collection(notebook_id)
    deleted_count = delete_source_chunks(collection, source_id)
    rebuild_bm25_index(notebook_id)
    return deleted_count


def delete_source_analysis_chunks(collection, source_id: str) -> int:
    try:
        existing = cast(dict[str, Any], collection.get(
            where={"source_id": source_id},
            include=["metadatas"]
        ))
        ids = cast(list[str], existing.get("ids") or [])
        metadatas = cast(list[Metadata], existing.get("metadatas") or [])
        ids_to_delete = [
            doc_id
            for doc_id, metadata in zip(ids, metadatas)
            if metadata and metadata.get("doc_type") in SUMMARY_DOC_TYPES
        ]
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)
    except Exception:
        return 0


def get_chroma_collection_names():
    names: list[str] = []
    try:
        for collection in chroma_client.list_collections():
            if isinstance(collection, str):
                names.append(collection)
            else:
                names.append(collection.name)
    except Exception:
        pass
    return names


def get_chroma_consistency_snapshot():
    collection_names = get_chroma_collection_names()
    notebook_collections = [
        name
        for name in collection_names
        if name.startswith("notebook_")
    ]
    source_chunks_by_notebook: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []

    for collection_name in notebook_collections:
        notebook_id = collection_name.replace("notebook_", "", 1)
        try:
            collection = chroma_client.get_collection(name=collection_name)
            data = cast(dict[str, Any], collection.get(include=["metadatas"]))
            source_counts: dict[str, int] = {}
            missing_source_id_chunks = 0
            for metadata in data.get("metadatas", []) or []:
                source_id = str((metadata or {}).get("source_id", "")).strip()
                if source_id:
                    source_counts[source_id] = source_counts.get(source_id, 0) + 1
                else:
                    missing_source_id_chunks += 1
            source_chunks_by_notebook[notebook_id] = {
                "source_counts": source_counts,
                "missing_source_id_chunks": missing_source_id_chunks
            }
        except Exception as error:
            errors.append({
                "collection": collection_name,
                "message": str(error)
            })

    return {
        "collections": notebook_collections,
        "source_chunks_by_notebook": source_chunks_by_notebook,
        "errors": errors
    }


def build_global_summary_document(analysis: dict[str, Any]) -> str:
    global_summary = cast(dict[str, Any], analysis.get("global_summary") or {})
    parts = [
        f"全局摘要標題：{global_summary.get('overall_title', '')}",
        f"整體摘要：{global_summary.get('overview', '')}",
    ]
    for label, key in (
        ("核心主題", "core_themes"),
        ("深度解析", "deep_insights"),
        ("重要細節", "important_details"),
    ):
        items = normalize_text_list(global_summary.get(key))
        if items:
            parts.append(f"{label}：{'；'.join(items)}")
    return "\n".join(part for part in parts if part.strip())


def build_chapter_summary_document(chapter: dict[str, Any]) -> str:
    parts = [
        f"章節：{chapter.get('title', '')}",
        f"時間：{chapter.get('time_range', '時間未記錄')}",
        f"摘要：{chapter.get('summary', '')}",
    ]
    key_points = normalize_text_list(chapter.get("key_points"))
    keywords = normalize_text_list(chapter.get("keywords"))
    if key_points:
        parts.append(f"重點：{'；'.join(key_points)}")
    if keywords:
        parts.append(f"關鍵詞：{'、'.join(keywords)}")
    return "\n".join(part for part in parts if part.strip())


def index_source_analysis_documents(notebook_id, source_id, filename, analysis):
    collection = get_notebook_collection(notebook_id)
    delete_source_analysis_chunks(collection, source_id)

    documents: list[str] = []
    metadatas: list[Metadata] = []
    ids: list[str] = []
    global_doc = build_global_summary_document(analysis)
    if global_doc.strip():
        documents.append(global_doc)
        ids.append(f"{source_id}_global_summary_{uuid.uuid4().hex[:6]}")
        metadatas.append({
            "source": filename,
            "source_id": source_id,
            "doc_type": "global_summary",
            "chunk_index": -1,
            "chapter_index": -1,
            "char_start": -1,
            "char_end": -1,
            "start_time": -1.0,
            "end_time": -1.0,
            "time_range": "全段錄音",
            "analysis_version": str(analysis.get("version", ANALYSIS_VERSION))
        })

    for chapter in cast(list[dict[str, Any]], analysis.get("chapters") or []):
        chapter_doc = build_chapter_summary_document(chapter)
        if not chapter_doc.strip():
            continue
        chapter_index = int(chapter.get("index", len(documents)))
        documents.append(chapter_doc)
        ids.append(f"{source_id}_chapter_{chapter_index}_{uuid.uuid4().hex[:6]}")
        metadatas.append({
            "source": filename,
            "source_id": source_id,
            "doc_type": "chapter_summary",
            "chunk_index": chapter_index,
            "chapter_index": chapter_index,
            "char_start": -1,
            "char_end": -1,
            "start_time": float(chapter.get("start_time", -1.0)),
            "end_time": float(chapter.get("end_time", -1.0)),
            "time_range": chapter.get("time_range", "時間未記錄"),
            "analysis_version": str(analysis.get("version", ANALYSIS_VERSION))
        })

    if not documents:
        rebuild_bm25_index(notebook_id)
        return 0

    vectors = [
        get_embeddings_model().encode(document, batch_size=EMBEDDING_BATCH_SIZE).tolist()
        for document in documents
    ]
    for doc_id, vector, document, metadata in zip(ids, vectors, documents, metadatas):
        collection.add(
            ids=[doc_id],
            embeddings=[vector],
            documents=[document],
            metadatas=[metadata]
        )

    rebuild_bm25_index(notebook_id)
    return len(documents)


def index_source_transcript(notebook_id, source_id, filename, transcript_text, timed_segments):
    """將單一來源逐字稿重新切分、向量化並寫回 ChromaDB"""
    transcript_text = transcript_text.strip()
    if not transcript_text:
        raise ValueError("逐字稿不可為空")

    chunks = chunk_transcript_text(transcript_text)
    if not chunks:
        raise ValueError("逐字稿內容不足，無法建立索引")

    chunk_metadatas = build_chunk_metadata(chunks, timed_segments, filename, source_id)
    vectors = [
        get_embeddings_model().encode(chunk, batch_size=EMBEDDING_BATCH_SIZE).tolist()
        for chunk in chunks
    ]

    collection = get_notebook_collection(notebook_id)
    delete_source_chunks(collection, source_id)
    for i, chunk in enumerate(chunks):
        chunk_id = f"{source_id}_chunk_{i}_{uuid.uuid4().hex[:6]}"
        collection.add(
            ids=[chunk_id],
            embeddings=[vectors[i]],
            documents=[chunk],
            metadatas=[chunk_metadatas[i]]
        )

    rebuild_bm25_index(notebook_id)
    return len(chunks)


def update_source_filename_metadata(notebook_id, source_id, filename):
    collection = get_notebook_collection(notebook_id)
    existing = cast(dict[str, Any], collection.get(
        where={"source_id": source_id},
        include=["metadatas"]
    ))
    ids = cast(list[str], existing.get("ids") or [])
    metadatas = cast(list[Metadata], existing.get("metadatas") or [])
    if not ids:
        return 0

    updated_metadatas = []
    for metadata in metadatas:
        updated_metadata = dict(metadata or {})
        updated_metadata["source"] = filename
        updated_metadatas.append(updated_metadata)

    collection.update(ids=ids, metadatas=updated_metadatas)
    rebuild_bm25_index(notebook_id)
    return len(ids)


def get_reference_label(metadata):
    """將 metadata 轉成使用者看得懂的來源引用文字"""
    if not metadata:
        return "未知來源"

    return metadata.get("source", "未知來源")


def strip_model_source_mentions(answer: str) -> str:
    """移除模型自行寫在正文中的來源文字，引用統一交給系統 references 顯示。"""
    cleaned = answer or ""
    cleaned = re.sub(r"\s*[（(]\s*(?:來源|參考來源|資料來源)\s*[:：][^）)]*[）)]", "", cleaned)
    cleaned = re.sub(r"\s*[（(]\s*(?:資料片段|片段)\s*\d+\s*[）)]", "", cleaned)

    lines = cleaned.splitlines()
    filtered_lines: list[str] = []
    skipping_reference_block = False
    for line in lines:
        stripped = line.strip()
        if re.match(r"^(?:參考來源|來源|資料來源)(?:\s*[:：]\s*)?$", stripped):
            skipping_reference_block = True
            continue
        if skipping_reference_block:
            if not stripped:
                skipping_reference_block = False
                filtered_lines.append(line)
                continue
            if re.match(r"^\d+[.、]\s+", stripped) or re.search(r"\.(?:mp3|m4a|wav|webm)\b", stripped, re.IGNORECASE):
                continue
            skipping_reference_block = False

        if re.match(r"^(?:參考來源|來源|資料來源)\s*[:：]", stripped):
            continue
        filtered_lines.append(line)

    cleaned = "\n".join(filtered_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def parse_suggested_questions(raw_text, limit):
    """解析 LLM 回傳的問題清單，優先讀 JSON，失敗時用行切分 fallback"""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1).replace("JSON\n", "", 1).strip()

    questions = []
    try:
        json_text = cleaned
        if not cleaned.startswith("[") and "[" in cleaned and "]" in cleaned:
            match = re.search(r'\[[\s\S]*\]', cleaned)
            if match:
                json_text = match.group(0)
        data = json.loads(json_text)
        if isinstance(data, dict):
            data = data.get("questions", [])
        if isinstance(data, list):
            questions = [str(item).strip() for item in data]
    except Exception:
        for line in cleaned.splitlines():
            question = re.sub(r'^\s*[-*•\d.、)）]+\s*', '', line).strip()
            if question:
                questions.append(question)

    unique_questions = []
    seen = set()
    for question in questions:
        question = question.strip().strip('"').strip("'")
        if not question or question in seen:
            continue
        seen.add(question)
        unique_questions.append(question)
        if len(unique_questions) >= limit:
            break
    return unique_questions


def generate_suggested_questions(transcript_text):
    """根據逐字稿產生推薦問題。失敗時回傳空清單，不中斷上傳流程。"""
    prompt = f"""
你是一個專業的知識庫問題設計助手。
請根據下方語音逐字稿的內容量、資訊密度與可提問性，自行判斷要產生 0 到 3 個使用者最可能想問的問題。

排序規則：
1. 最能幫助使用者理解整體內容的問題排前面。
2. 重要主題、關鍵觀點、實用資訊排前面。
3. 細節補充或延伸問題排後面。

輸出規則：
1. 必須使用繁體中文。
2. 最多只能輸出 3 題。
3. 如果內容太短、資訊不足，或沒有明確可問重點，請輸出空陣列 []。
4. 只輸出 JSON 字串陣列，例如 ["問題一？", "問題二？"]。
5. 不要輸出 Markdown、編號、解釋或其他文字。

語音逐字稿內容：
{transcript_text}
"""
    try:
        response = ollama.chat(model=OLLAMA_LLM_MODEL, messages=[{"role": "user", "content": prompt + "\n/no_think"}])
        raw_questions = response["message"]["content"]
        return parse_suggested_questions(raw_questions, 3)
    except Exception:
        return []


def is_global_question(question: str) -> bool:
    compact = re.sub(r"\s+", "", question or "")
    return any(keyword in compact for keyword in GLOBAL_QUESTION_KEYWORDS)


def get_summary_candidates(collection, limit: int = 24) -> tuple[list[str], dict[str, Metadata]]:
    try:
        existing = cast(dict[str, Any], collection.get(include=["documents", "metadatas"]))
    except Exception:
        return [], {}

    docs = cast(list[str], existing.get("documents") or [])
    metadatas = cast(list[Metadata], existing.get("metadatas") or [])
    pairs: list[tuple[int, str, Metadata]] = []
    for doc, metadata in zip(docs, metadatas):
        doc_type = (metadata or {}).get("doc_type")
        if doc_type not in SUMMARY_DOC_TYPES:
            continue
        priority = 0 if doc_type == "global_summary" else 1
        pairs.append((priority, doc, metadata or {}))

    pairs.sort(key=lambda item: (
        item[0],
        str(item[2].get("source", "")),
        int(item[2].get("chapter_index", item[2].get("chunk_index", 999999)) or 999999)
    ))
    selected = pairs[:limit]
    docs_out = [doc for _, doc, _ in selected]
    metadata_map = {doc: metadata for _, doc, metadata in selected}
    return docs_out, metadata_map


def merge_ranked_candidates(primary_docs: list[str], secondary_docs: list[str], limit: int) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for doc in primary_docs + secondary_docs:
        if not doc or doc in seen:
            continue
        seen.add(doc)
        merged.append(doc)
        if len(merged) >= limit:
            break
    return merged


def get_context_block_label(metadata: Metadata, index: int) -> str:
    doc_type = metadata.get("doc_type") if metadata else ""
    if doc_type == "global_summary":
        return f"全局摘要 {index}"
    if doc_type == "chapter_summary":
        time_range = metadata.get("time_range", "時間未記錄")
        return f"章節摘要 {index}（{time_range}）"
    return f"原文片段 {index}"


def select_answer_contexts(ranked: list[tuple[float, str]]) -> list[tuple[float, str]]:
    """從 reranker 結果中挑出較可信的少量片段，避免把弱相關雜訊餵給 LLM。"""
    if not ranked:
        return []

    best_score = ranked[0][0]
    selected: list[tuple[float, str]] = []
    for score, doc in ranked:
        if len(selected) < ANSWER_CONTEXT_MIN_COUNT or score >= best_score - ANSWER_CONTEXT_SCORE_MARGIN:
            selected.append((score, doc))
        if len(selected) >= ANSWER_CONTEXT_MAX_COUNT:
            break
    return selected


def normalize_confidence(value: Any, default: str = "low") -> str:
    confidence = str(value or "").strip().lower()
    return confidence if confidence in {"high", "medium", "low"} else default


def parse_answerable_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"true", "yes", "y", "1", "可回答", "可以回答"}


def get_answerability_evidence_signal(question: str, ranked: list[tuple[float, str]]) -> dict[str, Any]:
    """用關鍵詞重疊當作保護機制，避免 LLM 可答性判斷過度保守。"""
    question_tokens = get_meaningful_tokens(question)
    if not question_tokens:
        return {
            "strong_evidence": False,
            "best_overlap_score": 0.0,
            "total_overlap_score": 0.0,
        }

    best_overlap_score = 0.0
    total_overlap_score = 0.0
    for _, doc in ranked:
        doc_tokens = get_meaningful_tokens(doc)
        overlap_score = score_token_overlap(question_tokens & doc_tokens)
        best_overlap_score = max(best_overlap_score, overlap_score)
        total_overlap_score += overlap_score

    strong_evidence = (
        best_overlap_score >= ANSWERABILITY_STRONG_BEST_OVERLAP
        or (
            best_overlap_score >= 3.0
            and total_overlap_score >= ANSWERABILITY_STRONG_TOTAL_OVERLAP
        )
    )
    return {
        "strong_evidence": strong_evidence,
        "best_overlap_score": best_overlap_score,
        "total_overlap_score": total_overlap_score,
    }


def judge_answerability(question: str, retrieved_context: str, history_text: str = "") -> dict[str, Any]:
    """用本地 LLM 先判斷資料是否足以回答，避免弱相關片段造成幻覺。"""
    if not retrieved_context.strip():
        return {
            "answerable": False,
            "confidence": "low",
            "reason": "no_retrieved_context"
        }

    history_section = ""
    if history_text:
        history_section = f"""
【歷史對話】
{history_text}

注意：歷史對話只能用來理解代名詞或追問脈絡，不能當作回答證據。"""

    prompt = f"""你是一個 RAG 可答性判斷器，只負責判斷「參考資料是否足以支撐回答使用者問題」，不要真的回答問題。

判斷規則：
1. 如果參考資料提供了問題的核心主題、相關事實、摘要或可整理的證據，answerable 應該是 true。
2. 問題不需要逐字出現在參考資料中；只要能根據參考資料做合理整理、比較、歸納或說明，就可以回答。
3. 如果只是出現少量相同關鍵字、相似主題、影片標題相關，卻缺少問題所需的核心證據，answerable 必須是 false。
4. 如果需要依靠常識、外部知識、推測、延伸聯想才能回答，answerable 必須是 false。
5. 如果問題是一般知識、天氣、程式、人生建議，但參考資料沒有明確討論，answerable 必須是 false。
6. confidence 只能是 high、medium、low。
7. 只能輸出 JSON，不要輸出 Markdown、解釋文字或其他內容。

輸出格式：
{{
  "answerable": false,
  "confidence": "low",
  "reason": "一句話說明判斷原因"
}}
{history_section}

【參考資料】
{retrieved_context}

【使用者問題】
{question}
"""
    try:
        response = ollama.chat(model=OLLAMA_LLM_MODEL, messages=[{"role": "user", "content": prompt + "\n/no_think"}])
        parsed = extract_json_object(response["message"]["content"])
    except Exception:
        return {
            "answerable": False,
            "confidence": "low",
            "reason": "answerability_judge_failed"
        }
    if "answerable" not in parsed:
        return {
            "answerable": False,
            "confidence": "low",
            "reason": "answerability_parse_failed"
        }

    answerable = parse_answerable_value(parsed.get("answerable"))
    confidence = normalize_confidence(parsed.get("confidence"), default="medium" if answerable else "low")
    return {
        "answerable": answerable,
        "confidence": confidence,
        "reason": str(parsed.get("reason") or "").strip()
    }


class SourceNotFoundError(Exception):
    pass


def update_source_transcript(notebook_id, source_id, transcript_text):
    transcript_text = transcript_text.strip()
    if not transcript_text:
        return {"status": "error", "message": "逐字稿不可為空。"}

    row = get_source_update_info(notebook_id, source_id)
    if not row:
        raise SourceNotFoundError()

    filename, existing_transcript, timed_segments_json = row
    if not existing_transcript or not existing_transcript.strip():
        return {"status": "error", "message": "此來源尚未保存逐字稿，請重新上傳音檔後再編輯。"}

    if transcript_text == existing_transcript.strip():
        return {
            "status": "success",
            "changed": False,
            "message": "逐字稿沒有變更，不需要重新索引。",
            "source": {
                "id": source_id,
                "filename": filename,
                "has_transcript": True
            },
            "chunk_count": 0
        }

    try:
        timed_segments = json.loads(timed_segments_json) if timed_segments_json else []
        analysis = build_source_analysis(filename, transcript_text, timed_segments)
        chunk_count = index_source_transcript(notebook_id, source_id, filename, transcript_text, timed_segments)
        analysis_chunk_count = index_source_analysis_documents(notebook_id, source_id, filename, analysis)
    except Exception as e:
        return {"status": "error", "message": f"重新索引失敗: {str(e)}"}

    analysis_json = json.dumps(
        {key: value for key, value in analysis.items() if key != "structured_knowledge"},
        ensure_ascii=False
    )
    now = update_source_transcript_record(
        notebook_id,
        source_id,
        transcript_text,
        analysis.get("mode"),
        "completed",
        analysis_json
    )
    try:
        save_suggested_questions(notebook_id, filename, analysis.get("suggested_questions", []))
    except Exception:
        pass

    return {
        "status": "success",
        "changed": True,
        "message": f"逐字稿已更新，並重新建立 {chunk_count} 個原文片段與 {analysis_chunk_count} 個摘要片段。",
        "source": {
            "id": source_id,
            "filename": filename,
            "has_transcript": True,
            "transcript_updated_at": now,
            "indexed_at": now,
            "analysis_mode": analysis.get("mode"),
            "analysis_status": "completed",
            "analysis_updated_at": now
        },
        "chunk_count": chunk_count,
        "analysis_chunk_count": analysis_chunk_count,
        "structured_knowledge": analysis.get("structured_knowledge", ""),
        "suggested_questions": analysis.get("suggested_questions", [])
    }


def answer_question(notebook_id: str, question: str) -> dict[str, Any]:
    ollama_was_used = False
    try:
        unload_whisper_model()
        query_vector = cast(
            list[float],
            get_embeddings_model().encode(question, batch_size=EMBEDDING_BATCH_SIZE).tolist(),
        )

        collection = get_notebook_collection(notebook_id)
        if collection.count() == 0:
            return {"status": "error", "message": "此筆記本尚未上傳任何來源，請先上傳音檔。"}

        total_docs = collection.count()
        n_candidates = min(40, total_docs)
        global_question = is_global_question(question)

        dense_results = cast(dict[str, Any], collection.query(
            query_embeddings=[query_vector],
            n_results=n_candidates,
            include=["documents", "metadatas"]
        ))
        dense_docs = cast(list[str], (dense_results.get("documents") or [[]])[0])
        dense_metas = cast(list[Metadata], (dense_results.get("metadatas") or [[]])[0])
        doc_metadata_map: dict[str, Metadata] = {}
        for doc, meta in zip(dense_docs, dense_metas):
            if meta:
                doc_metadata_map[doc] = meta

        bm25_top_docs: list[str] = []
        if notebook_id not in bm25_indices:
            rebuild_bm25_index(notebook_id)

        if notebook_id in bm25_indices and bm25_indices[notebook_id] is not None:
            query_tokens = tokenize_chinese(question)
            bm25_scores = bm25_indices[notebook_id].get_scores(query_tokens)
            scored_indices = [
                (int(index), float(score))
                for index, score in enumerate(bm25_scores)
                if float(score) > 0
            ]
            scored_indices.sort(key=lambda item: item[1], reverse=True)
            top_indices = [index for index, _ in scored_indices[:n_candidates]]
            bm25_top_docs = [bm25_docs[notebook_id][i] for i in top_indices]
            if notebook_id in bm25_metas:
                for i in top_indices:
                    doc_metadata_map[bm25_docs[notebook_id][i]] = bm25_metas[notebook_id][i] or {}

        if bm25_top_docs:
            fused_candidates = rrf_fusion(dense_docs, bm25_top_docs)
        else:
            fused_candidates = dense_docs

        summary_docs, summary_metadata_map = get_summary_candidates(collection)
        doc_metadata_map.update(summary_metadata_map)
        if global_question and summary_docs:
            fused_candidates = merge_ranked_candidates(summary_docs, fused_candidates, limit=60)
        else:
            fused_candidates = merge_ranked_candidates(fused_candidates, summary_docs[:6], limit=50)

        pairs: list[tuple[str, str]] = [(question, doc) for doc in fused_candidates]
        if not pairs:
            return {"status": "error", "message": "目前知識庫沒有可用的文字片段，請重新上傳或重新分析來源。"}

        raw_scores = get_reranker().predict(cast(Any, pairs), batch_size=RERANKER_BATCH_SIZE)
        scores = normalize_reranker_scores(raw_scores)

        ranked = select_answer_contexts(sorted(zip(scores, fused_candidates), key=lambda x: x[0], reverse=True))
        top_documents = [doc for _, doc in ranked]
        reference_candidates: list[dict[str, Any]] = []
        context_blocks: list[str] = []
        seen_reference_keys: set[str] = set()
        for index, (reranker_score, doc) in enumerate(ranked, start=1):
            metadata = doc_metadata_map.get(doc, {})
            label = get_reference_label(metadata)
            context_label = get_context_block_label(metadata, index)
            context_blocks.append(f"[{context_label}｜來源：{label}]\n{doc}")
            source_id = str(metadata.get("source_id", "")) if metadata else ""
            chunk_index = metadata.get("chunk_index", -1) if metadata else -1
            doc_type = metadata.get("doc_type", TRANSCRIPT_DOC_TYPE) if metadata else TRANSCRIPT_DOC_TYPE
            reference_key = f"{source_id or label}:{doc_type}:{chunk_index}"
            if reference_key not in seen_reference_keys:
                seen_reference_keys.add(reference_key)
                reference_candidates.append(build_reference_candidate(question, doc, metadata, reranker_score))
        retrieved_context = "\n\n".join(context_blocks)

        history = get_recent_messages(notebook_id, limit=6)
        history_text = ""
        if history:
            history_lines = []
            for msg in history:
                role = "使用者" if msg["sender"] == "User" else "AI 助理"
                raw_history_text = str(msg["text"])
                history_body = strip_model_source_mentions(raw_history_text) if msg["sender"] == "AI" else raw_history_text
                text = history_body[:300] + "..." if len(history_body) > 300 else history_body
                history_lines.append(f"{role}：{text}")
            history_text = "\n".join(history_lines)

        evidence_signal = get_answerability_evidence_signal(question, ranked)
        ollama_was_used = True
        answerability = judge_answerability(question, retrieved_context, history_text)
        if not answerability["answerable"] and evidence_signal["strong_evidence"]:
            answerability = {
                "answerable": True,
                "confidence": "medium",
                "reason": "retrieved_context_has_strong_keyword_evidence"
            }

        if not answerability["answerable"]:
            references: list[dict[str, Any]] = []
            try:
                append_qa_messages(notebook_id, question, NO_ANSWER_MESSAGE, references)
            except Exception:
                pass

            return {
                "status": "success",
                "question": question,
                "answer": NO_ANSWER_MESSAGE,
                "reference_sources": [],
                "source_files": [],
                "references": references,
                "answerable": False,
                "confidence": answerability["confidence"],
                "refusal_reason": answerability.get("reason") or "no_relevant_evidence"
            }

        rag_prompt = f"""你現在是一個嚴格且專業的「知識庫檢索助理」。
系統已先確認參考資料可能足以回答問題，但你仍然必須【完全且只能】依據下方的【參考資料】作答。

⚠️ 絕對遵守以下規則：
1.【務必】使用「繁體中文」進行輸出，嚴禁出現簡體字！
2. 如果【參考資料】直接支持答案，請用條理清晰、分點說明的方式回答。
3. 如果【參考資料】無法回答問題，請誠實回答：「根據目前資料庫的錄音紀錄，並未提及此資訊」，【絕對不可以】編造答案。
4. 禁止根據常識、影片標題、相似主題、外部知識或你的背景知識延伸回答。
5. 回答正文只寫答案，不要自行輸出「來源」、「參考來源」、「資料來源」、檔名、音檔名稱或資料片段編號。
6. 若使用編號清單，請使用 1、2、3 依序編號，不要每一點都寫成 1。
7. 引用來源會由系統在畫面下方獨立顯示，你不需要也不可以在正文中標註來源。
8. 若有【歷史對話】，只能用來理解追問脈絡，不能把歷史對話當作新的事實來源。
9. 若參考資料同時包含「全局摘要」、「章節摘要」與「原文片段」，回答整體主題、摘要、潤色或深度解析問題時，請優先使用全局摘要與章節摘要，再用原文片段補細節。"""

        if history_text:
            rag_prompt += f"\n\n【歷史對話】：\n{history_text}"

        rag_prompt += f"\n\n【參考資料】：\n{retrieved_context}"
        rag_prompt += f"\n\n【使用者的問題】：\n{question}"

        ollama_was_used = True
        response = ollama.chat(model=OLLAMA_LLM_MODEL, messages=[{"role": "user", "content": rag_prompt + "\n/no_think"}])
        ai_answer = strip_model_source_mentions(response["message"]["content"])
        answerable = not is_no_answer_response(ai_answer)
        references = filter_relevant_references(question, ai_answer, reference_candidates) if answerable else []
        source_files = list(dict.fromkeys(reference["source"] for reference in references))

        try:
            append_qa_messages(notebook_id, question, ai_answer, references)
        except Exception:
            pass

        return {
            "status": "success",
            "question": question,
            "answer": ai_answer,
            "reference_sources": top_documents,
            "source_files": source_files,
            "references": references,
            "answerable": answerable,
            "confidence": answerability["confidence"] if answerable else "low",
            "refusal_reason": "" if answerable else "model_returned_no_answer"
        }

    except Exception as e:
        return {"status": "error", "message": f"問答過程發生錯誤: {str(e)}"}
    finally:
        if ollama_was_used:
            stop_ollama_model()
        clear_cuda_memory()
