import sys
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from db import (
    create_notebook_record,
    get_notebook_details_data,
    get_notebooks_data,
    get_source_transcript_data,
    init_db,
    mark_suggested_question_used_record,
    update_notebook_name,
)
from schemas import NotebookUpdate, QuestionRequest, SourceFilenameUpdate, TranscriptAiEditRequest, TranscriptUpdateRequest
from models import preload_models_for_provider
from llm_service import get_default_llm_provider, normalize_llm_provider


app = FastAPI()
STATIC_DIR = BASE_DIR / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_frontend():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.on_event("startup")
async def preload_models_on_startup():
    preload_models_for_provider(get_default_llm_provider())


init_db()

from audio_service import (
    check_data_consistency,
    clear_orphan_chroma_data,
    delete_notebook_data,
    delete_source_data,
    get_audio_file_path,
    process_audio_upload,
    reanalyze_source_data,
    rename_source_filename,
    suggest_source_filename,
)
from llm_service import get_llm_provider_config, stop_llm_generation
from rag_service import (
    SourceNotFoundError,
    ai_edit_source_transcript,
    answer_question,
    get_notebook_collection,
    update_source_transcript,
)


@app.get("/api/notebooks/")
async def get_notebooks():
    return {"status": "success", "notebooks": get_notebooks_data()}


@app.get("/api/llm-providers/")
async def get_llm_providers():
    return get_llm_provider_config()


@app.post("/api/llm-providers/preload")
async def preload_llm_provider(llm_provider: str | None = None):
    provider = normalize_llm_provider(llm_provider)
    preload_models_for_provider(provider)
    return {"status": "success", "provider": provider}


@app.post("/api/notebooks/")
async def create_notebook():
    notebook = create_notebook_record()
    get_notebook_collection(notebook["id"])
    return {"status": "success", "notebook": notebook}


@app.put("/api/notebooks/{notebook_id}")
async def update_notebook(notebook_id: str, data: NotebookUpdate):
    update_notebook_name(notebook_id, data.name)
    return {"status": "success"}


@app.get("/api/notebooks/{notebook_id}")
async def get_notebook_details(notebook_id: str):
    details = get_notebook_details_data(notebook_id)
    return {
        "status": "success",
        "sources": details["sources"],
        "messages": details["messages"],
        "suggested_questions": details["suggested_questions"],
    }


@app.post("/api/notebooks/{notebook_id}/suggested-questions/{question_id}/used")
async def mark_suggested_question_used(notebook_id: str, question_id: int):
    mark_suggested_question_used_record(notebook_id, question_id)
    return {"status": "success"}


@app.get("/api/notebooks/{notebook_id}/sources/{source_id}/transcript")
async def get_source_transcript(notebook_id: str, source_id: str):
    source = get_source_transcript_data(notebook_id, source_id)
    if not source:
        raise HTTPException(status_code=404, detail="找不到指定來源")
    return {"status": "success", "source": source}


@app.put("/api/notebooks/{notebook_id}/sources/{source_id}/transcript")
async def update_source_transcript_endpoint(notebook_id: str, source_id: str, request: TranscriptUpdateRequest):
    try:
        return update_source_transcript(notebook_id, source_id, request.transcript_text, request.llm_provider)
    except SourceNotFoundError:
        raise HTTPException(status_code=404, detail="找不到指定來源")


@app.post("/api/notebooks/{notebook_id}/sources/{source_id}/transcript/ai-edit")
async def ai_edit_source_transcript_endpoint(notebook_id: str, source_id: str, request: TranscriptAiEditRequest):
    try:
        result = ai_edit_source_transcript(
            notebook_id,
            source_id,
            request.transcript_text,
            request.operation,
            request.llm_provider,
        )
        if result.get("status") != "success":
            raise HTTPException(status_code=400, detail=result.get("message", "AI 編輯失敗"))
        return result
    except SourceNotFoundError:
        raise HTTPException(status_code=404, detail="找不到指定來源")
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"AI 編輯失敗：{error}")


@app.get("/api/notebooks/{notebook_id}/sources/{source_id}/audio")
async def get_source_audio(notebook_id: str, source_id: str):
    audio = get_audio_file_path(notebook_id, source_id)
    if not audio:
        raise HTTPException(status_code=404, detail="找不到來源音檔")
    return FileResponse(audio["path"], filename=audio["filename"])


@app.delete("/api/notebooks/{notebook_id}/sources/{source_id}")
async def delete_source(notebook_id: str, source_id: str):
    result = delete_source_data(notebook_id, source_id)
    if result.get("status") != "success":
        raise HTTPException(status_code=404, detail=result.get("message", "找不到指定來源"))
    return result


@app.put("/api/notebooks/{notebook_id}/sources/{source_id}/filename")
async def update_source_filename(notebook_id: str, source_id: str, request: SourceFilenameUpdate):
    result = rename_source_filename(notebook_id, source_id, request.filename)
    if result.get("status") != "success":
        raise HTTPException(status_code=404, detail=result.get("message", "找不到指定來源"))
    return result


@app.post("/api/notebooks/{notebook_id}/sources/{source_id}/filename/suggest")
async def suggest_source_filename_endpoint(notebook_id: str, source_id: str, llm_provider: str | None = None):
    result = suggest_source_filename(notebook_id, source_id, llm_provider)
    if result.get("status") != "success":
        message = result.get("message", "AI 取檔名失敗")
        status_code = 404 if "找不到" in message else 500
        raise HTTPException(status_code=status_code, detail=message)
    return result


@app.post("/api/notebooks/{notebook_id}/sources/{source_id}/reanalyze")
async def reanalyze_source(notebook_id: str, source_id: str, llm_provider: str | None = None):
    result = reanalyze_source_data(notebook_id, source_id, llm_provider)
    if result.get("status") != "success":
        message = result.get("message", "重新分析失敗")
        status_code = 404 if "找不到" in message else 500
        raise HTTPException(status_code=status_code, detail=message)
    return result


@app.get("/api/data-consistency")
async def get_data_consistency():
    return check_data_consistency()


@app.post("/api/maintenance/clear-orphan-chroma")
async def clear_orphan_chroma_data_endpoint():
    try:
        return clear_orphan_chroma_data()
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"清除 Chroma 孤兒索引失敗：{error}")


@app.delete("/api/notebooks/{notebook_id}")
async def delete_notebook(notebook_id: str):
    result = delete_notebook_data(notebook_id)
    if result.get("status") != "success":
        raise HTTPException(status_code=404, detail=result.get("message", "Notebook not found"))
    return result


@app.post("/upload-audio/")
async def upload_audio(
    notebook_id: str = Form(...),
    file: UploadFile = File(...),
    auto_filename: bool = Form(False),
    fallback_filename: str | None = Form(None),
    llm_provider: str | None = Form(None)
):
    return process_audio_upload(notebook_id, file, auto_filename, fallback_filename, llm_provider)


@app.post("/ask-question/")
async def ask_question(request: QuestionRequest):
    return answer_question(request.notebook_id, request.question, request.llm_provider)


@app.post("/api/stop-answer/")
async def stop_answer(llm_provider: str | None = None):
    result = stop_llm_generation(llm_provider)
    if result.get("status") != "success":
        raise HTTPException(status_code=500, detail=result.get("message", "停止回答失敗"))
    return result
