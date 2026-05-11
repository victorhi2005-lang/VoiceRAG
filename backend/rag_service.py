import json
import re
import uuid
from collections.abc import Iterable
from typing import Any, cast

import chromadb
import jieba
import numpy as np
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from db import append_qa_messages, get_recent_messages, get_source_update_info, update_source_transcript_record
from models import embeddings_model, reranker


chroma_client = chromadb.PersistentClient(path="./chroma_db")


Metadata = dict[str, Any]


def get_notebook_collection(notebook_id: str) -> Any:
    return chroma_client.get_or_create_collection(name=f"notebook_{notebook_id}")


# 每個筆記本維護一個獨立的 BM25 索引，存在記憶體中
bm25_indices: dict[str, Any] = {}   # {notebook_id: BM25Okapi 實例}
bm25_docs: dict[str, list[str]] = {}      # {notebook_id: [原始文件列表]}
bm25_metas: dict[str, list[Metadata]] = {}     # {notebook_id: [metadata 列表]}


def tokenize_chinese(text: str) -> list[str]:
    """jieba 中文分詞，過濾空白"""
    return [w for w in jieba.cut(text) if w.strip()]


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


def delete_notebook_collection(notebook_id: str) -> None:
    try:
        chroma_client.delete_collection(name=f"notebook_{notebook_id}")
    except Exception:
        pass
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


def semantic_chunk(text, similarity_threshold=0.65, max_chunk_size=600, min_chunk_size=80):
    """基於語意相似度的智慧切分。"""
    sentences = split_sentences(text)

    if len(sentences) <= 2:
        return [text] if text.strip() else []

    sentence_embeddings = embeddings_model.encode(sentences)

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

    total_seconds = max(0, int(round(float(seconds))))
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
            "chunk_index": i,
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
    try:
        collection.delete(where={"source_id": source_id})
        return
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


def index_source_transcript(notebook_id, source_id, filename, transcript_text, timed_segments):
    """將單一來源逐字稿重新切分、向量化並寫回 ChromaDB"""
    transcript_text = transcript_text.strip()
    if not transcript_text:
        raise ValueError("逐字稿不可為空")

    chunks = chunk_transcript_text(transcript_text)
    if not chunks:
        raise ValueError("逐字稿內容不足，無法建立索引")

    chunk_metadatas = build_chunk_metadata(chunks, timed_segments, filename, source_id)
    vectors = [embeddings_model.encode(chunk).tolist() for chunk in chunks]

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


def get_reference_label(metadata):
    """將 metadata 轉成使用者看得懂的來源引用文字"""
    if not metadata:
        return "未知來源（時間未記錄）"

    source = metadata.get("source", "未知來源")
    time_range = metadata.get("time_range")
    if not time_range:
        start_time = metadata.get("start_time")
        end_time = metadata.get("end_time")
        if start_time is not None and float(start_time) >= 0 and end_time is not None and float(end_time) >= 0:
            time_range = get_time_range_text(start_time, end_time)
        else:
            time_range = "時間未記錄"
    return f"{source}（{time_range}）"


def append_reference_summary(answer, references):
    """在 AI 回答後附上系統整理的參考來源，確保回答可追溯"""
    if not references:
        return answer

    lines = ["", "", "參考來源："]
    for i, reference in enumerate(references, start=1):
        lines.append(f"{i}. {reference['label']}")
    return answer.rstrip() + "\n".join(lines)


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
        response = ollama.chat(model="qwen3:14b", messages=[{"role": "user", "content": prompt + "\n/no_think"}])
        raw_questions = response["message"]["content"]
        return parse_suggested_questions(raw_questions, 3)
    except Exception:
        return []


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
        chunk_count = index_source_transcript(notebook_id, source_id, filename, transcript_text, timed_segments)
    except Exception as e:
        return {"status": "error", "message": f"重新索引失敗: {str(e)}"}

    now = update_source_transcript_record(notebook_id, source_id, transcript_text)
    return {
        "status": "success",
        "changed": True,
        "message": f"逐字稿已更新，並重新建立 {chunk_count} 個知識片段。",
        "source": {
            "id": source_id,
            "filename": filename,
            "has_transcript": True,
            "transcript_updated_at": now,
            "indexed_at": now
        },
        "chunk_count": chunk_count
    }


def answer_question(notebook_id: str, question: str) -> dict[str, Any]:
    try:
        query_vector = cast(list[float], embeddings_model.encode(question).tolist())

        collection = get_notebook_collection(notebook_id)
        if collection.count() == 0:
            return {"status": "error", "message": "此筆記本尚未上傳任何來源，請先上傳音檔。"}

        total_docs = collection.count()
        n_candidates = min(20, total_docs)

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
            top_n = min(n_candidates, len(bm25_scores))
            top_indices = [int(i) for i in bm25_scores.argsort()[-top_n:][::-1]]
            bm25_top_docs = [bm25_docs[notebook_id][i] for i in top_indices]
            if notebook_id in bm25_metas:
                for i in top_indices:
                    doc_metadata_map[bm25_docs[notebook_id][i]] = bm25_metas[notebook_id][i] or {}

        if bm25_top_docs:
            fused_candidates = rrf_fusion(dense_docs, bm25_top_docs)
        else:
            fused_candidates = dense_docs

        fused_candidates = fused_candidates[:20]

        pairs: list[tuple[str, str]] = [(question, doc) for doc in fused_candidates]
        raw_scores = reranker.predict(cast(Any, pairs))

        if isinstance(raw_scores, (int, float)):
            scores = [float(raw_scores)]
        else:
            scores = [float(score) for score in cast(Iterable[Any], raw_scores)]

        ranked = sorted(zip(scores, fused_candidates), key=lambda x: x[0], reverse=True)[:5]
        top_documents = [doc for _, doc in ranked]
        references: list[dict[str, Any]] = []
        context_blocks: list[str] = []
        seen_reference_labels: set[str] = set()
        for index, doc in enumerate(top_documents, start=1):
            metadata = doc_metadata_map.get(doc, {})
            label = get_reference_label(metadata)
            context_blocks.append(f"[來源 {index}: {label}]\n{doc}")
            if label not in seen_reference_labels:
                seen_reference_labels.add(label)
                references.append({
                    "label": label,
                    "source": metadata.get("source", "未知來源") if metadata else "未知來源",
                    "time_range": metadata.get("time_range", "時間未記錄") if metadata else "時間未記錄",
                    "start_time": metadata.get("start_time", -1) if metadata else -1,
                    "end_time": metadata.get("end_time", -1) if metadata else -1,
                    "chunk_index": metadata.get("chunk_index", -1) if metadata else -1
                })
        retrieved_context = "\n\n".join(context_blocks)

        source_files = list(dict.fromkeys(reference["source"] for reference in references))
        source_info = "；".join(reference["label"] for reference in references) if references else "未知來源"

        history = get_recent_messages(notebook_id, limit=6)
        history_text = ""
        if history:
            history_lines = []
            for msg in history:
                role = "使用者" if msg["sender"] == "User" else "AI 助理"
                text = msg["text"][:300] + "..." if len(msg["text"]) > 300 else msg["text"]
                history_lines.append(f"{role}：{text}")
            history_text = "\n".join(history_lines)

        rag_prompt = f"""你現在是一個嚴格且專業的「知識庫檢索助理」。
請你【完全且只能】依據下方的【參考資料】來回答使用者的問題。

⚠️ 絕對遵守以下規則：
1.【務必】使用「繁體中文」進行輸出，嚴禁出現簡體字！
2. 如果【參考資料】中有答案，請用條理清晰、分點說明的方式回答。
3. 如果【參考資料】無法回答問題，請誠實回答：「根據目前資料庫的錄音紀錄，並未提及此資訊」，【絕對不可以】編造答案。
4. 回答時請注明參考來源檔案：{source_info}。
5. 若有【歷史對話】，請結合上下文脈絡理解使用者的追問意圖。"""

        if history_text:
            rag_prompt += f"\n\n【歷史對話】：\n{history_text}"

        rag_prompt += f"\n\n【參考資料】：\n{retrieved_context}"
        rag_prompt += f"\n\n【使用者的問題】：\n{question}"

        response = ollama.chat(model="qwen3:14b", messages=[{"role": "user", "content": rag_prompt + "\n/no_think"}])
        ai_answer = append_reference_summary(response["message"]["content"], references)

        try:
            append_qa_messages(notebook_id, question, ai_answer)
        except Exception:
            pass

        return {
            "status": "success",
            "question": question,
            "answer": ai_answer,
            "reference_sources": top_documents,
            "source_files": source_files,
            "references": references
        }

    except Exception as e:
        return {"status": "error", "message": f"問答過程發生錯誤: {str(e)}"}
