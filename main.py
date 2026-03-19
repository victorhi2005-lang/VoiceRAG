from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import sqlite3
from faster_whisper import WhisperModel
import ollama

# RAG
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# =========================
# FastAPI
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "audio_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# =========================
# SQLite（一般資料庫）
# =========================
conn = sqlite3.connect("data.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS recordings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    transcript TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

# =========================
# ChromaDB（向量資料庫）
# =========================
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="knowledge")

# =========================
# Embedding
# =========================
print("載入 embedding 模型...")
embeddings_model = HuggingFaceEmbeddings(
    model_name="shibing624/text2vec-base-chinese"
)

# =========================
# Whisper
# =========================
print("載入 Whisper...")
model = WhisperModel("base", device="cpu", compute_type="int8")

print("✅ 系統啟動完成")

# =========================
# 🎤 上傳錄音
# =========================
@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 1️⃣ 語音轉文字
    segments, _ = model.transcribe(file_path, beam_size=5, vad_filter=True)
    transcript_text = "".join([seg.text for seg in segments])

    # 2️⃣ 存 SQLite（原始資料）
    cursor.execute(
        "INSERT INTO recordings (filename, transcript) VALUES (?, ?)",
        (file.filename, transcript_text)
    )
    conn.commit()

    # 3️⃣ Ollama 整理知識
    prompt = f"""
    請將以下內容整理成條列式重點知識：

    {transcript_text}

    ⚠️ 請用繁體中文
    """

    response = ollama.chat(
        model='qwen2.5',
        messages=[{"role": "user", "content": prompt}]
    )

    structured_text = response["message"]["content"]

    # 4️⃣ 切塊
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = splitter.split_text(structured_text)

    # 5️⃣ 存 ChromaDB
    for i, chunk in enumerate(chunks):
        vector = embeddings_model.embed_query(chunk)

        collection.add(
            ids=[f"{file.filename}_{i}"],
            embeddings=[vector],
            documents=[chunk],
            metadatas=[{"source": file.filename}]
        )

    return {
        "status": "success",
        "transcript": transcript_text,
        "knowledge": structured_text
    }

# =========================
# 🤖 問答（RAG）
# =========================
@app.post("/ask")
async def ask(data: dict):

    question = data["question"]

    # 1️⃣ 向量化問題
    query_vector = embeddings_model.embed_query(question)

    # 2️⃣ 查詢知識庫
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=3
    )

    docs = results["documents"][0]

    if not docs:
        return {"answer": "目前沒有相關知識"}

    context = "\n".join(docs)

    # 3️⃣ Ollama 回答
    prompt = f"""
    根據以下知識回答問題：

    {context}

    問題：{question}

    ⚠️ 用繁體中文回答
    """

    response = ollama.chat(
        model='qwen2.5',
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "answer": response["message"]["content"]
    }