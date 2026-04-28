from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel # 用來定義接收問題的資料格式
import shutil
import os
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


# 系統初始化區塊 
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="mis_knowledge")

print("正在載入 Embedding 向量模型，請稍候...")
embeddings_model = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")

print("正在載入本地端 Whisper 模型，請稍候...")
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
print("[OK] 所有 AI 系統與資料庫載入完成！")


# API 1：上傳音檔並寫入記憶庫 

@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
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
        prompt = f"""
        你是一個專業的知識整理助手。請將以下的口述逐字稿，
        整理成結構化的重點知識（包含標題與條列式說明），並去除口語化的冗言贅字。
        ⚠️ 絕對要求：請務必使用「繁體中文 (Traditional Chinese)」輸出！
        口述逐字稿內容：\n{transcript_text}
        """
        response = ollama.chat(model='qwen2.5', messages=[{'role': 'user', 'content': prompt}])
        structured_knowledge = response['message']['content']
    except Exception as e:
        return {"status": "error", "message": f"AI 整理失敗: {str(e)}"}

    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n", "。", "！", "？", "，", " "])
        chunks = text_splitter.split_text(structured_knowledge)

        for i, chunk in enumerate(chunks):
            vector = embeddings_model.embed_query(chunk)
            chunk_id = f"{file.filename}_chunk_{i}"
            collection.add(ids=[chunk_id], embeddings=[vector], documents=[chunk], metadatas=[{"source": file.filename}])
    except Exception as e:
        return {"status": "error", "message": f"寫入向量資料庫失敗: {str(e)}"}

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
    question: str

@app.post("/ask-question/")
async def ask_question(request: QuestionRequest):
    try:
        # 1. 將使用者的問題轉成數學向量
        query_vector = embeddings_model.embed_query(request.question)
        
        # 2. 去 ChromaDB 尋找最相關的 5 段記憶 
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
        return {
            "status": "success",
            "question": request.question,
            "answer": ai_answer,
            "reference_sources": results['documents'][0] # 附上找出來的 5 段小抄
        }
        
    except Exception as e:
        return {"status": "error", "message": f"問答過程發生錯誤: {str(e)}"}