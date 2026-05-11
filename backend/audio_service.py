import gc
import json
import os
import shutil
import uuid

import ollama
import torch

from db import insert_ai_summary_message, insert_source_record, save_suggested_questions
from models import whisper_model
from rag_service import generate_suggested_questions, index_source_transcript


UPLOAD_DIR = "audio_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def save_uploaded_audio(file):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return file_path


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
            "start": float(segment.start),
            "end": float(segment.end)
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


def process_audio_upload(notebook_id, file):
    source_id = str(uuid.uuid4())
    file_path = save_uploaded_audio(file)

    try:
        transcript_text, timed_segments = transcribe_audio(file_path)
    except Exception as e:
        return {"status": "error", "message": f"語音辨識失敗: {str(e)}"}

    try:
        chunks_count = index_source_transcript(notebook_id, source_id, file.filename, transcript_text, timed_segments)
        source_saved_at = insert_source_record(
            notebook_id,
            source_id,
            file.filename,
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
        insert_ai_summary_message(notebook_id, file.filename, structured_knowledge)
        saved_suggested_questions = save_suggested_questions(notebook_id, file.filename, suggested_questions)
    except Exception:
        pass

    return {
        "filename": file.filename,
        "status": "success",
        "message": f"✅ 成功存入 {chunks_count} 個知識片段。",
        "raw_transcript": transcript_text,
        "structured_knowledge": structured_knowledge,
        "suggested_questions": saved_suggested_questions,
        "source": {
            "id": source_id,
            "filename": file.filename,
            "added_at": source_saved_at,
            "has_transcript": True,
            "transcript_updated_at": source_saved_at,
            "indexed_at": source_saved_at
        }
    }
