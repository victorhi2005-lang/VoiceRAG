from typing import Literal

from pydantic import BaseModel


class NotebookUpdate(BaseModel):
    name: str


class TranscriptUpdateRequest(BaseModel):
    transcript_text: str
    llm_provider: str | None = None


class TranscriptAiEditRequest(BaseModel):
    transcript_text: str
    operation: Literal["correct", "punctuate"]
    llm_provider: str | None = None


class SourceFilenameUpdate(BaseModel):
    filename: str


class QuestionRequest(BaseModel):
    notebook_id: str
    question: str
    llm_provider: str | None = None
