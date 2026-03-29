from fastapi import FastAPI, UploadFile, File
import shutil
import os
from faster_whisper import WhisperModel

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
        
    try:
        segments, info = model.transcribe(
            file_path,
            beam_size=5,
            language="zh",
            initial_prompt="這是一段中文的逐字稿："
        )
        transcript_text = "".join([segment.text for segment in segments])
    except Exception as e:
        return {"status": "error", "message": f"語音辨識失敗: {str(e)}"}

    return {
        "filename": file.filename, 
        "status": "success",
        "raw_transcript": transcript_text
    }
