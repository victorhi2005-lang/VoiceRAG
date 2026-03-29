from fastapi import FastAPI, UploadFile, File
import shutil
import os

app = FastAPI()
UPLOAD_DIR = "audio_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

print("✅ 後端伺服器啟動成功！")

@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    # 將手機傳來的檔案存入資料夾
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {
        "filename": file.filename, 
        "status": "success",
        "message": "音檔上傳成功，等待後續語音辨識處理"
    }
