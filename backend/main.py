from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel # 用來定義接收問題的資料格式
import shutil
import os
import sqlite3
import uuid
from datetime import datetime
from sentence_transformers import SentenceTransformer, CrossEncoder
from faster_whisper import WhisperModel, BatchedInferencePipeline
import ollama

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import jieba
import numpy as np
import re
import torch
import gc

app = FastAPI()
# 告訴 FastAPI 網頁檔案放在 static 資料夾裡
app.mount("/static", StaticFiles(directory="static"), name="static")

# 當使用者輸入首頁網址 ("/") 時，回傳 index.html 畫面給他
@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")
    
UPLOAD_DIR = "audio_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ==========================================
# 系統初始化與資料庫設定
# ==========================================
def init_db():
    conn = sqlite3.connect("notebooks.db")
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
            FOREIGN KEY(notebook_id) REFERENCES notebooks(id) ON DELETE CASCADE
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            notebook_id TEXT,
            sender TEXT,
            text TEXT,
            created_at TEXT,
            FOREIGN KEY(notebook_id) REFERENCES notebooks(id) ON DELETE CASCADE
        )
    ''')
    conn.commit()
    conn.close()

init_db()

chroma_client = chromadb.PersistentClient(path="./chroma_db")

def get_notebook_collection(notebook_id: str):
    return chroma_client.get_or_create_collection(name=f"notebook_{notebook_id}")

print("正在載入 Embedding 向量模型 (BAAI/bge-m3)，請稍候...", flush=True)
embeddings_model = SentenceTransformer('BAAI/bge-m3', device='cuda', model_kwargs={'torch_dtype': torch.float16})

print("正在載入 Reranker 精排模型 (BAAI/bge-reranker-v2-m3)，請稍候...", flush=True)
reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', device='cuda', model_kwargs={'torch_dtype': torch.float16})

print("正在載入本地端 Whisper 模型 (large-v3-turbo)，請稍候...", flush=True)
base_model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
model = BatchedInferencePipeline(model=base_model)
print("[OK] 所有 AI 系統與資料庫載入完成！", flush=True)

# ==========================================
# BM25 關鍵字索引管理
# ==========================================
# 每個筆記本維護一個獨立的 BM25 索引，存在記憶體中
bm25_indices = {}   # {notebook_id: BM25Okapi 實例}
bm25_docs = {}      # {notebook_id: [原始文件列表]}

def tokenize_chinese(text):
    """jieba 中文分詞，過濾空白"""
    return [w for w in jieba.cut(text) if w.strip()]

def rebuild_bm25_index(notebook_id):
    """從 ChromaDB 重建指定筆記本的 BM25 索引"""
    try:
        collection = get_notebook_collection(notebook_id)
        if collection.count() == 0:
            return
        all_data = collection.get()
        docs = all_data['documents']
        bm25_docs[notebook_id] = docs
        tokenized = [tokenize_chinese(doc) for doc in docs]
        bm25_indices[notebook_id] = BM25Okapi(tokenized)
    except Exception:
        pass

def rrf_fusion(dense_docs, bm25_result_docs, k=60):
    """使用 Reciprocal Rank Fusion (RRF) 融合兩組排名結果"""
    scores = {}
    # Dense 排名
    for rank, doc in enumerate(dense_docs):
        scores[doc] = scores.get(doc, 0) + 1.0 / (k + rank + 1)
    # BM25 排名
    for rank, doc in enumerate(bm25_result_docs):
        scores[doc] = scores.get(doc, 0) + 1.0 / (k + rank + 1)
    # 依合併分數排序
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in sorted_docs]

# ==========================================
# 語意切分 (Semantic Chunking)
# ==========================================
def split_sentences(text):
    """將中文文本按句子分割"""
    # 按中文標點切分，保留標點
    sentences = re.split(r'(?<=[\u3002\uff01\uff1f\uff1b\u000a])', text)
    # 過濾空白和過短的片段
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
    return sentences

def semantic_chunk(text, embeddings_model, similarity_threshold=0.65, max_chunk_size=600, min_chunk_size=80):
    """
    基於語意相似度的智慧切分。
    將相鄰且語意相近的句子聚合成一個 chunk，在語意跳躍處切斷。
    
    參數：
        text: 原始文字
        embeddings_model: BGE-M3 向量模型
        similarity_threshold: 語意相似度門檻，低於此值則切斷 (0~1)
        max_chunk_size: 單個 chunk 最大字數
        min_chunk_size: 單個 chunk 最小字數
    """
    sentences = split_sentences(text)
    
    # 句子太少時直接回傳
    if len(sentences) <= 2:
        return [text] if text.strip() else []
    
    # 計算每個句子的向量
    sentence_embeddings = embeddings_model.encode(sentences)
    
    # 計算相鄰句子間的餘弦相似度
    similarities = []
    for i in range(len(sentence_embeddings) - 1):
        vec_a = sentence_embeddings[i]
        vec_b = sentence_embeddings[i + 1]
        cos_sim = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        similarities.append(float(cos_sim))
    
    # 根據相似度決定切割點
    chunks = []
    current_chunk = sentences[0]
    
    for i in range(len(similarities)):
        next_sentence = sentences[i + 1]
        
        # 判斷是否應該在此處切斷
        should_split = False
        
        # 條件 1：語意相似度低於門檻，代表話題轉換
        if similarities[i] < similarity_threshold:
            should_split = True
        
        # 條件 2：累積長度超過最大上限，強制切斷
        if len(current_chunk) + len(next_sentence) > max_chunk_size:
            should_split = True
        
        if should_split and len(current_chunk) >= min_chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = next_sentence
        else:
            current_chunk += next_sentence
    
    # 最後一個 chunk
    if current_chunk.strip():
        # 如果最後一段太短，併入前一個
        if len(current_chunk.strip()) < min_chunk_size and chunks:
            chunks[-1] += current_chunk.strip()
        else:
            chunks.append(current_chunk.strip())
    
    return chunks if chunks else [text]

# ==========================================
# API 1：筆記本管理 (Notebook CRUD)
# ==========================================

class NotebookUpdate(BaseModel):
    name: str

@app.get("/api/notebooks/")
async def get_notebooks():
    conn = sqlite3.connect("notebooks.db")
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
    return {"status": "success", "notebooks": notebooks}

@app.post("/api/notebooks/")
async def create_notebook():
    nid = str(uuid.uuid4())
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    name = "未命名筆記本"
    icon = "📓"
    
    conn = sqlite3.connect("notebooks.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO notebooks (id, name, icon, updated_at) VALUES (?, ?, ?, ?)", (nid, name, icon, now))
    conn.commit()
    conn.close()
    
    # 預先建立 ChromaDB Collection
    get_notebook_collection(nid)
    return {"status": "success", "notebook": {"id": nid, "name": name, "icon": icon, "updated_at": now, "source_count": 0}}

@app.put("/api/notebooks/{notebook_id}")
async def update_notebook(notebook_id: str, data: NotebookUpdate):
    conn = sqlite3.connect("notebooks.db")
    cursor = conn.cursor()
    cursor.execute("UPDATE notebooks SET name = ?, updated_at = ? WHERE id = ?", (data.name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), notebook_id))
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.get("/api/notebooks/{notebook_id}")
async def get_notebook_details(notebook_id: str):
    conn = sqlite3.connect("notebooks.db")
    cursor = conn.cursor()
    
    # 取得來源清單
    cursor.execute("SELECT id, filename, added_at FROM sources WHERE notebook_id = ? ORDER BY added_at ASC", (notebook_id,))
    sources = [{"id": r[0], "filename": r[1], "added_at": r[2]} for r in cursor.fetchall()]
    
    # 取得對話紀錄
    cursor.execute("SELECT sender, text, created_at FROM messages WHERE notebook_id = ? ORDER BY id ASC", (notebook_id,))
    messages = [{"sender": r[0], "text": r[1], "created_at": r[2]} for r in cursor.fetchall()]
    
    conn.close()
    return {"status": "success", "sources": sources, "messages": messages}

@app.delete("/api/notebooks/{notebook_id}")
async def delete_notebook(notebook_id: str):
    conn = sqlite3.connect("notebooks.db")
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")
    cursor.execute("DELETE FROM notebooks WHERE id = ?", (notebook_id,))
    conn.commit()
    conn.close()
    
    # 從 ChromaDB 刪除 collection
    try:
        chroma_client.delete_collection(name=f"notebook_{notebook_id}")
    except Exception:
        pass
    
    # 清除 BM25 索引
    bm25_indices.pop(notebook_id, None)
    bm25_docs.pop(notebook_id, None)
        
    return {"status": "success"}

# ==========================================
# API 2：上傳音檔並寫入記憶庫 
# ==========================================

@app.post("/upload-audio/")
async def upload_audio(notebook_id: str = Form(...), file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # 暫時卸載 Ollama 模型，騰出 GPU VRAM 給 Whisper 全速處理
        try:
            ollama.generate(model='qwen3:14b', prompt='', keep_alive=0)
            torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass  # 若 Ollama 未載入也不影響流程

        segments, info = model.transcribe(
            file_path,
            beam_size=5,
            language="zh",
            initial_prompt="這是一段繁體中文的台灣口音逐字稿：",
            vad_filter=True,  # 啟用 VAD 過濾靜音段，減少 VRAM 峰值
            batch_size=16     # 開啟 16 線程批次推論
        )
        transcript_text = "".join([segment.text for segment in segments])
        # 釋放 Whisper 推論時佔用的 VRAM
        torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
            return {"status": "error", "message": f"語音辨識失敗: {str(e)}"}

    try:
        # 1. 將原始逐字稿進行語意切分並寫入向量資料庫
        # 優先使用語意切分，當文字太短時 fallback 到固定切分
        if len(transcript_text) > 200:
            chunks = semantic_chunk(transcript_text, embeddings_model, similarity_threshold=0.65, max_chunk_size=600, min_chunk_size=80)
        else:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n", "。", "！", "？", "，", " "])
            chunks = text_splitter.split_text(transcript_text)

        collection = get_notebook_collection(notebook_id)
        for i, chunk in enumerate(chunks):
            vector = embeddings_model.encode(chunk).tolist()
            chunk_id = f"{file.filename}_chunk_{i}_{uuid.uuid4().hex[:6]}"
            collection.add(ids=[chunk_id], embeddings=[vector], documents=[chunk], metadatas=[{"source": file.filename}])
        
        # 同步重建該筆記本的 BM25 索引
        rebuild_bm25_index(notebook_id)
            
        # 更新 SQLite 紀錄
        conn = sqlite3.connect("notebooks.db")
        cursor = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO sources (id, notebook_id, filename, added_at) VALUES (?, ?, ?, ?)", (str(uuid.uuid4()), notebook_id, file.filename, now))
        cursor.execute("UPDATE notebooks SET updated_at = ? WHERE id = ?", (now, notebook_id))
        conn.commit()
        conn.close()
    except Exception as e:
        return {"status": "error", "message": f"寫入向量資料庫失敗: {str(e)}"}
    finally:
        # 釋放 Embedding 過程中的暫存記憶體
        gc.collect()

    try:
        # 2. 用 AI 針對完整訊息進行分析，產生整體重點摘要供前端顯示
        prompt = f"""
        你是一個專業的 AI 知識分析助手。請閱讀以下的口述語音逐字稿，
        進行深入的訊息分析，並給出一份精煉的「整體重點摘要」。
        請用流暢的段落來總結核心訊息，幫助讀者快速掌握整段語音的精華。
        
        ⚠️ 絕對要求：請務必使用「繁體中文 (Traditional Chinese)」輸出，嚴禁出現簡體字！
        
        語音逐字稿內容：\n{transcript_text}
        """
        response = ollama.chat(model='qwen3:14b', messages=[{'role': 'user', 'content': prompt + '\n/no_think'}])
        structured_knowledge = response['message']['content']
    except Exception as e:
        return {"status": "error", "message": f"AI 摘要失敗: {str(e)}"}

    # 3. 將 AI 摘要存入對話紀錄
    try:
        conn = sqlite3.connect("notebooks.db")
        cursor = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_result = f"**《{file.filename}》重點摘要**\n\n{structured_knowledge}"
        cursor.execute("INSERT INTO messages (notebook_id, sender, text, created_at) VALUES (?, ?, ?, ?)", (notebook_id, 'AI', formatted_result, now))
        conn.commit()
        conn.close()
    except Exception as e:
        pass # 若儲存對話失敗不中斷流程

    return {
        "filename": file.filename, 
        "status": "success",
        "message": f"✅ 成功存入 {len(chunks)} 個知識片段。",
        "raw_transcript": transcript_text,
        "structured_knowledge": structured_knowledge
    }


# API 2：AI 知識問答端點

def get_recent_messages(notebook_id, limit=6):
    """讀取指定筆記本最近 N 筆對話紀錄，用於多輪對話記憶"""
    try:
        conn = sqlite3.connect("notebooks.db")
        cursor = conn.cursor()
        cursor.execute(
            "SELECT sender, text FROM messages WHERE notebook_id = ? ORDER BY id DESC LIMIT ?",
            (notebook_id, limit)
        )
        rows = cursor.fetchall()
        conn.close()
        # 反轉為時間正序 (最舊的在前面)
        rows.reverse()
        return [{"sender": r[0], "text": r[1]} for r in rows]
    except Exception:
        return []


# 定義接收前端問題的格式
class QuestionRequest(BaseModel):
    notebook_id: str
    question: str

@app.post("/ask-question/")
async def ask_question(request: QuestionRequest):
    try:
        # 1. 將使用者的問題轉成數學向量
        query_vector = embeddings_model.encode(request.question).tolist()
        
        # 2. 混合檢索：Dense 向量檢索 + BM25 關鍵字檢索
        collection = get_notebook_collection(request.notebook_id)
        if collection.count() == 0:
            return {"status": "error", "message": "此筆記本尚未上傳任何來源，請先上傳音檔。"}
        
        total_docs = collection.count()
        n_candidates = min(20, total_docs)
        
        # 2a. Dense 向量檢索 (Top-20)，同時獲取 metadata 用於來源追蹤
        dense_results = collection.query(
            query_embeddings=[query_vector],
            n_results=n_candidates,
            include=["documents", "metadatas"]
        )
        dense_docs = dense_results['documents'][0]
        dense_metas = dense_results['metadatas'][0]
        # 建立文件 → 來源檔名的對照表
        doc_source_map = {}
        for doc, meta in zip(dense_docs, dense_metas):
            if meta and 'source' in meta:
                doc_source_map[doc] = meta['source']
        
        # 2b. BM25 關鍵字檢索 (Top-20)
        bm25_top_docs = []
        nb_id = request.notebook_id
        if nb_id not in bm25_indices:
            rebuild_bm25_index(nb_id)
        
        if nb_id in bm25_indices and bm25_indices[nb_id] is not None:
            query_tokens = tokenize_chinese(request.question)
            bm25_scores = bm25_indices[nb_id].get_scores(query_tokens)
            top_n = min(n_candidates, len(bm25_scores))
            top_indices = bm25_scores.argsort()[-top_n:][::-1]
            bm25_top_docs = [bm25_docs[nb_id][i] for i in top_indices]
        
        # 3. 使用 RRF (Reciprocal Rank Fusion) 融合兩組結果
        if bm25_top_docs:
            fused_candidates = rrf_fusion(dense_docs, bm25_top_docs)
        else:
            fused_candidates = dense_docs
        
        # 取前 20 名送入 Reranker
        fused_candidates = fused_candidates[:20]
        
        # 4. 用 Reranker 對融合後的候選片段進行二次精排，取出 Top-5
        pairs = [[request.question, doc] for doc in fused_candidates]
        scores = reranker.predict(pairs)
        
        if isinstance(scores, (int, float)):
            scores = [scores]
        
        ranked = sorted(zip(scores, fused_candidates), key=lambda x: x[0], reverse=True)[:5]
        top_documents = [doc for _, doc in ranked]
        retrieved_context = "\n\n".join(top_documents)
        
        # 彙整 Top-5 的來源檔名
        source_files = list(set(doc_source_map.get(doc, "未知來源") for doc in top_documents))
        source_info = "、".join(source_files)
        
        # 5. 讀取最近對話紀錄，建構多輪對話脈絡
        history = get_recent_messages(request.notebook_id, limit=6)
        history_text = ""
        if history:
            history_lines = []
            for msg in history:
                role = "使用者" if msg['sender'] == 'User' else "AI 助理"
                # 截斷過長的歷史訊息以節省 Token
                text = msg['text'][:300] + "..." if len(msg['text']) > 300 else msg['text']
                history_lines.append(f"{role}：{text}")
            history_text = "\n".join(history_lines)
        
        # 6. 組裝完整的 RAG Prompt（含對話歷史 + 參考資料 + 來源標註）
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
        rag_prompt += f"\n\n【使用者的問題】：\n{request.question}"
        
        # 7. 呼叫 Qwen3-14B 回答
        response = ollama.chat(model='qwen3:14b', messages=[{'role': 'user', 'content': rag_prompt + '\n/no_think'}])
        ai_answer = response['message']['content']
        
        # 8. 將問答存入對話紀錄
        try:
            conn = sqlite3.connect("notebooks.db")
            cursor = conn.cursor()
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("INSERT INTO messages (notebook_id, sender, text, created_at) VALUES (?, ?, ?, ?)", (request.notebook_id, 'User', request.question, now))
            cursor.execute("INSERT INTO messages (notebook_id, sender, text, created_at) VALUES (?, ?, ?, ?)", (request.notebook_id, 'AI', ai_answer, now))
            cursor.execute("UPDATE notebooks SET updated_at = ? WHERE id = ?", (now, request.notebook_id))
            conn.commit()
            conn.close()
        except Exception as e:
            pass

        return {
            "status": "success",
            "question": request.question,
            "answer": ai_answer,
            "reference_sources": top_documents,
            "source_files": source_files
        }
        
    except Exception as e:
        return {"status": "error", "message": f"問答過程發生錯誤: {str(e)}"}