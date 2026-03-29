from fastapi import FastAPI, UploadFile, File
import shutil
import os
from faster_whisper import WhisperModel
import ollama

app = FastAPI()
UPLOAD_DIR = "audio_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

print("正在載入本地端 Whisper 模型，請稍候...")
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
print("✅ 語音辨識系統載入完成！")

@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # 語音辨識
    try:
        segments, info = model.transcribe(file_path, beam_size=5, language="zh", initial_prompt="這是一段中文的逐字稿：")
        transcript_text = "".join([segment.text for segment in segments])
    except Exception as e:
        return {"status": "error", "message": f"語音辨識失敗: {str(e)}"}

    # AI 知識萃取
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

    return {
        "filename": file.filename, 
        "status": "success",
        "raw_transcript": transcript_text,
        "structured_knowledge": structured_knowledge
    }
