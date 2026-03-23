from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel # 📍 新增：用來定義接收問題的資料格式
import shutil #複製檔案
import os #管理資料夾
from faster_whisper import WhisperModel 
import ollama

import chromadb #向量資料庫(存放數字轉向量)
from langchain_text_splitters import RecursiveCharacterTextSplitter #把長文章拆成小段。
from langchain_community.embeddings import HuggingFaceEmbeddings #文字 → 向量

app = FastAPI()
UPLOAD_DIR = "audio_uploads" #設定音檔上傳資料夾
os.makedirs(UPLOAD_DIR, exist_ok=True) #建立資料夾（如果不存在）


# 系統初始化區塊

chroma_client = chromadb.PersistentClient(path="./chroma_db") #建立 ChromaDB 的資料庫連線
collection = chroma_client.get_or_create_collection(name="mis_knowledge") #建立一個 資料集合

print("正在載入 Embedding 向量模型，請稍候...")
embeddings_model = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")

print("正在載入本地端 Whisper 模型，請稍候...")
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
print("✅ 所有 AI 系統與資料庫載入完成！")


# API 1：上傳音檔並寫入記憶庫 

@app.post("/upload-audio/") #建立 API 端點
async def upload_audio(file: UploadFile = File(...)): #UploadFile 接收前端上傳的檔案 File(...) 必填參數
    file_path = os.path.join(UPLOAD_DIR, file.filename) #建立檔案路徑(資料夾+檔名)
    with open(file_path, "wb") as buffer: #開啟檔案寫入(2進位)
        shutil.copyfileobj(file.file, buffer) #把上傳的檔案內容複製到伺服器檔案
        
    try:
        segments, info = model.transcribe(
            file_path,
            beam_size=5, #模型同時嘗試 5 種可能句子
            language="zh",
            initial_prompt="這是一段繁體中文的台灣口音逐字稿：" #提示 Whisper
        )
        transcript_text = "".join([segment.text for segment in segments]) #把每段文字取出後合併
    except Exception as e:
            return {"status": "error", "message": f"語音辨識失敗: {str(e)}"}
#給 LLM 的指令
    try:
        prompt = f"""
        你是一個專業的知識整理助手。請將以下的口述逐字稿，
        整理成結構化的重點知識（包含標題與條列式說明），並去除口語化的冗言贅字。 
        ⚠️ 絕對要求：請務必使用「繁體中文 (Traditional Chinese)」輸出！
        口述逐字稿內容：\n{transcript_text}
        """
        response = ollama.chat(model='qwen2.5', messages=[{'role': 'user', 'content': prompt}]) #呼叫 Ollama LLM
        structured_knowledge = response['message']['content'] #取得整理後的文字
    except Exception as e:
        return {"status": "error", "message": f"AI 整理失敗: {str(e)}"}
#把 AI 整理好的結構化知識切成小段、轉成向量，然後存到向量資料庫
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n", "。", "！", "？", "，", " "])#每段最多 500 個字 每段有 100 字重疊，保留前後文連貫性 優先按段落、換行、標點切割
        chunks = text_splitter.split_text(structured_knowledge)

        for i, chunk in enumerate(chunks):
            vector = embeddings_model.embed_query(chunk)#文字轉成數字向量
            chunk_id = f"{file.filename}_chunk_{i}"#建立 chunk ID 讓資料庫能識別每段文字
            collection.add(ids=[chunk_id], embeddings=[vector], documents=[chunk], metadatas=[{"source": file.filename}]) #把資料存進 Vector Database。
    except Exception as e:
        return {"status": "error", "message": f"寫入向量資料庫失敗: {str(e)}"}

    return {
        "filename": file.filename, 
        "status": "success",
        "message": f"✅ 成功存入 {len(chunks)} 個知識片段。",
        "raw_transcript": transcript_text,
        "structured_knowledge": structured_knowledge
    }


# API 2：AI 知識問答

# 定義接收前端問題的格式
class QuestionRequest(BaseModel):
    question: str

@app.post("/ask-question/")
async def ask_question(request: QuestionRequest):
    try:
        # 1. 將使用者的問題轉成數學向量
        query_vector = embeddings_model.embed_query(request.question)
        
        # 2. 去 ChromaDB 尋找最相關的5段
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=5
        )
        
        # 把找到的記憶片段組合起來
        retrieved_context = "\n\n".join(results['documents'][0])
        
        # 3. Prompt：強迫 AI 只能根據小抄回答
        rag_prompt = f"""
        你現在是一個嚴格且專業的「知識庫檢索助理」。
        請你【完全且只能】依據下方的【參考資料】來回答使用者的問題。

        ⚠️ 絕對遵守以下四條規則：
        1.【務必】使用「繁體中文」進行輸出，嚴禁出現簡體字！
        2. 如果【參考資料】中有答案，請用條理清晰、分點說明的方式（繁體中文）回答。
        3. 如果【參考資料】的內容無法完全回答問題，【絕對不可以】使用你自身的常識來腦補或編造答案。
        4. 回答完畢後，請在最後簡述你主要是參考了哪幾段資料。

        【參考資料】：
        {retrieved_context}

        【使用者的問題】：
        {request.question}
        """
        
        # 4. 呼叫大腦回答
        response = ollama.chat(model='qwen2.5', messages=[{'role': 'user', 'content': rag_prompt}])
        ai_answer = response['message']['content']
        
        # 5. 回傳答案與引用的資料來源 
        return {
            "status": "success",
            "question": request.question,
            "answer": ai_answer,
            "reference_sources": results['documents'][0] # 附上找出來的 3 段小抄
        }
        
    except Exception as e:
        return {"status": "error", "message": f"問答過程發生錯誤: {str(e)}"}
