from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from db import (
    create_notebook_record,
    delete_notebook_record,
    get_notebook_details_data,
    get_notebooks_data,
    get_source_transcript_data,
    init_db,
    mark_suggested_question_used_record,
    update_notebook_name,
)
from schemas import NotebookUpdate, QuestionRequest, TranscriptUpdateRequest


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")


init_db()

from audio_service import process_audio_upload
from rag_service import (
    SourceNotFoundError,
    answer_question,
    delete_notebook_collection,
    get_notebook_collection,
    update_source_transcript,
)


@app.get("/api/notebooks/")
async def get_notebooks():
    return {"status": "success", "notebooks": get_notebooks_data()}


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
        return update_source_transcript(notebook_id, source_id, request.transcript_text)
    except SourceNotFoundError:
        raise HTTPException(status_code=404, detail="找不到指定來源")


@app.delete("/api/notebooks/{notebook_id}")
async def delete_notebook(notebook_id: str):
    delete_notebook_record(notebook_id)
    delete_notebook_collection(notebook_id)
    return {"status": "success"}


@app.post("/upload-audio/")
async def upload_audio(notebook_id: str = Form(...), file: UploadFile = File(...)):
    return process_audio_upload(notebook_id, file)


@app.post("/ask-question/")
async def ask_question(request: QuestionRequest):
    return answer_question(request.notebook_id, request.question)
