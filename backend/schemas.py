from pydantic import BaseModel


class NotebookUpdate(BaseModel):
    name: str


class TranscriptUpdateRequest(BaseModel):
    transcript_text: str


class SourceFilenameUpdate(BaseModel):
    filename: str


class QuestionRequest(BaseModel):
    notebook_id: str
    question: str
