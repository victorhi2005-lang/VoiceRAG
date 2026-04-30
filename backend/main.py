from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel # 用來定義接收問題的資料格式
import shutil
import os
import sqlite3
import uuid
from datetime import datetime
from faster_whisper import WhisperModel
import ollama

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

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

print("正在載入 Embedding 向量模型，請稍候...")
embeddings_model = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")

print("正在載入本地端 Whisper 模型，請稍候...")
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
print("[OK] 所有 AI 系統與資料庫載入完成！")

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
        segments, info = model.transcribe(
            file_path,
            beam_size=5,
            language="zh",
            initial_prompt="這是一段繁體中文的台灣口音逐字稿："
        )
        transcript_text = "".join([segment.text for segment in segments])
    except Exception as e:
            return {"status": "error", "message": f"語音辨識失敗: {str(e)}"}

    try:
        # 1. 將原始逐字稿切片並寫入向量資料庫（寫入專屬的 Notebook Collection）
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n", "。", "！", "？", "，", " "])
        chunks = text_splitter.split_text(transcript_text)

        collection = get_notebook_collection(notebook_id)
        for i, chunk in enumerate(chunks):
            vector = embeddings_model.embed_query(chunk)
            chunk_id = f"{file.filename}_chunk_{i}_{uuid.uuid4().hex[:6]}"
            collection.add(ids=[chunk_id], embeddings=[vector], documents=[chunk], metadatas=[{"source": file.filename}])
            
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

    try:
        # 2. 用 AI 針對完整訊息進行分析，產生整體重點摘要供前端顯示
        prompt = f"""
        你是一個專業的 AI 知識分析助手。請閱讀以下的口述語音逐字稿，
        進行深入的訊息分析，並給出一份精煉的「整體重點摘要」。
        請用流暢的段落來總結核心訊息，幫助讀者快速掌握整段語音的精華。
        
        ⚠️ 絕對要求：請務必使用「繁體中文 (Traditional Chinese)」輸出，嚴禁出現簡體字！
        
        語音逐字稿內容：\n{transcript_text}
        """
        response = ollama.chat(model='qwen2.5', messages=[{'role': 'user', 'content': prompt}])
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


# 定義接收前端問題的格式
class QuestionRequest(BaseModel):
    notebook_id: str
    question: str

@app.post("/ask-question/")
async def ask_question(request: QuestionRequest):
    try:
        # 1. 將使用者的問題轉成數學向量
        query_vector = embeddings_model.embed_query(request.question)
        
        # 2. 去 ChromaDB 指定的筆記本中尋找最相關的 5 段記憶 
        collection = get_notebook_collection(request.notebook_id)
        if collection.count() == 0:
            return {"status": "error", "message": "此筆記本尚未上傳任何來源，請先上傳音檔。"}
            
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=5
        )
        
        # 把找到的記憶片段組合起來
        retrieved_context = "\n\n".join(results['documents'][0])
        
        # 3.Prompt強迫 AI 只能有根據回答
        rag_prompt = f"""
        你現在是一個嚴格且專業的「知識庫檢索助理」。
        請你【完全且只能】依據下方的【參考資料】來回答使用者的問題。

        ⚠️ 絕對遵守以下四條規則：
        1.【務必】使用「繁體中文」進行輸出，嚴禁出現簡體字！
        2. 如果【參考資料】中有答案，請用條理清晰、分點說明的方式（繁體中文）回答。
        3. 如果【參考資料】的內容無法完全回答問題，請誠實回答：「根據目前資料庫的錄音紀錄，並未提及此資訊」，【絕對不可以】使用你自身的常識來腦補或編造答案。
        4. 回答完畢後，請在最後簡述你主要是參考了哪幾段資料。

        【參考資料】：
        {retrieved_context}

        【使用者的問題】：
        {request.question}
        """
        
        # 4. 呼叫qwen回答
        response = ollama.chat(model='qwen2.5', messages=[{'role': 'user', 'content': rag_prompt}])
        ai_answer = response['message']['content']
        
        # 5. 回傳答案與引用的資料來源 
        
        # 6. 將問答存入對話紀錄
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
            "reference_sources": results['documents'][0] # 附上找出來的 5 段小抄
        }
        
    except Exception as e:
        return {"status": "error", "message": f"問答過程發生錯誤: {str(e)}"}