import json
import mimetypes
import os
import re
import uuid
from pathlib import Path
from typing import Any, Iterable

from db import (
    get_source_file_info,
    get_source_filenames,
    insert_source_record,
)
from rag_service import delete_source_from_index, index_source_document, should_use_deep_analysis


BASE_DIR = Path(__file__).resolve().parent
DOCUMENT_UPLOAD_DIR = BASE_DIR / "document_uploads"
DOCUMENT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_DOCUMENT_BYTES = 50 * 1024 * 1024
SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
EXPECTED_MIME_TYPES = {
    ".pdf": {"application/pdf", "application/octet-stream"},
    ".docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/zip",
        "application/octet-stream",
    },
    ".txt": {"text/plain", "application/octet-stream"},
    ".md": {"text/markdown", "text/plain", "application/octet-stream"},
}


class DocumentProcessingError(ValueError):
    pass


def is_supported_document(filename: str | None) -> bool:
    return Path(filename or "").suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS


def get_document_delivery_options(
    filename: str,
    mime_type: str | None,
    path: Path | None = None,
) -> tuple[str | None, str]:
    extension = Path(filename).suffix.lower()
    if extension in {".txt", ".md"}:
        charset = detect_text_document_encoding(path) if path else "utf-8"
        return f"text/plain; charset={charset}", "inline"
    if extension == ".pdf":
        return mime_type or "application/pdf", "inline"
    return mime_type, "attachment"


def sanitize_document_filename(filename: str) -> str:
    original = Path(filename or "").name
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", Path(original).stem).strip(" ._")
    stem = re.sub(r"\s+", " ", stem)[:80].rstrip(" ._") or "document"
    extension = Path(original).suffix.lower()
    if extension not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise DocumentProcessingError("不支援此文件格式。")
    return f"{stem}{extension}"


def ensure_unique_document_filename(notebook_id: str, filename: str, ignore_filename: str | None = None) -> str:
    existing = set(get_source_filenames(notebook_id))
    stem = Path(filename).stem
    extension = Path(filename).suffix
    candidate = filename
    counter = 2
    while (
        (candidate in existing and candidate != ignore_filename)
        or ((DOCUMENT_UPLOAD_DIR / candidate).exists() and candidate != ignore_filename)
    ):
        candidate = f"{stem}-{counter}{extension}"
        counter += 1
    return candidate


def save_document_upload(file: Any, filename: str) -> Path:
    destination = DOCUMENT_UPLOAD_DIR / filename
    total = 0
    try:
        with destination.open("wb") as output:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_DOCUMENT_BYTES:
                    raise DocumentProcessingError("文件大小超過 50 MB 上限。")
                output.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    if total == 0:
        destination.unlink(missing_ok=True)
        raise DocumentProcessingError("文件內容為空。")
    return destination


def validate_document_upload(path: Path, content_type: str | None) -> None:
    extension = path.suffix.lower()
    normalized_mime = (content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream").lower()
    if normalized_mime not in EXPECTED_MIME_TYPES[extension]:
        raise DocumentProcessingError(f"檔案類型與副檔名不符：{normalized_mime}")

    with path.open("rb") as input_file:
        prefix = input_file.read(4096)
    if extension == ".pdf" and not prefix.startswith(b"%PDF-"):
        raise DocumentProcessingError("檔案不是有效的 PDF。")
    if extension == ".docx" and not prefix.startswith(b"PK"):
        raise DocumentProcessingError("檔案不是有效的 DOCX。")
    if extension in {".txt", ".md"} and b"\x00" in prefix:
        raise DocumentProcessingError("文字文件包含無法解析的二進位內容。")


def make_segment(
    segment_id: str,
    text: str,
    location_type: str,
    location_label: str,
    **location: Any,
) -> dict[str, Any]:
    segment = {
        "segment_id": segment_id,
        "text": text.strip(),
        "location_type": location_type,
        "location_label": location_label,
    }
    segment.update(location)
    return segment


def table_to_text(rows: Iterable[Iterable[Any]]) -> str:
    rendered_rows = []
    for row in rows:
        cells = [re.sub(r"\s+", " ", str(cell or "")).strip() for cell in row]
        if any(cells):
            rendered_rows.append(" | ".join(cells))
    return "\n".join(rendered_rows)


def extract_pdf_segments(path: Path) -> list[dict[str, Any]]:
    try:
        import pdfplumber
    except ImportError as error:
        raise RuntimeError("缺少 pdfplumber，請重新安裝 requirements.txt。") from error

    segments: list[dict[str, Any]] = []
    try:
        with pdfplumber.open(str(path)) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text = (page.extract_text() or "").strip()
                table_texts = []
                for table in page.extract_tables() or []:
                    rendered = table_to_text(table)
                    if rendered and rendered not in text:
                        table_texts.append(rendered)
                combined = "\n\n".join(part for part in [text, *table_texts] if part).strip()
                if combined:
                    segments.append(make_segment(
                        f"page-{page_number}",
                        combined,
                        "page",
                        f"第 {page_number} 頁",
                        page_number=page_number,
                    ))
    except Exception as error:
        message = str(error).lower()
        if "password" in message or "encrypt" in message:
            raise DocumentProcessingError("此 PDF 已加密，請先解除密碼後再上傳。") from error
        raise DocumentProcessingError(f"PDF 解析失敗：{error}") from error

    if not segments:
        raise DocumentProcessingError("PDF 沒有可抽取文字，可能是掃描 PDF；目前不支援 OCR。")
    return segments


def iter_docx_blocks(document: Any) -> Iterable[Any]:
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    for child in document.element.body.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, document)
        elif child.tag.endswith("}tbl"):
            yield Table(child, document)


def extract_docx_segments(path: Path) -> list[dict[str, Any]]:
    try:
        from docx import Document
        from docx.table import Table
    except ImportError as error:
        raise RuntimeError("缺少 python-docx，請重新安裝 requirements.txt。") from error

    try:
        document = Document(str(path))
    except Exception as error:
        raise DocumentProcessingError(f"DOCX 解析失敗：{error}") from error

    segments: list[dict[str, Any]] = []
    current_heading = ""
    current_parts: list[str] = []
    start_paragraph = 1
    paragraph_number = 0

    def flush(end_at: int | None = None) -> None:
        nonlocal current_parts, start_paragraph
        text = "\n".join(current_parts).strip()
        if not text:
            current_parts = []
            return
        end_paragraph = max(start_paragraph, end_at if end_at is not None else paragraph_number)
        label = current_heading or f"段落 {start_paragraph}-{end_paragraph}"
        segments.append(make_segment(
            f"section-{len(segments) + 1}",
            text,
            "section",
            label,
            section_title=current_heading,
            paragraph_start=start_paragraph,
            paragraph_end=end_paragraph,
        ))
        current_parts = []
        start_paragraph = paragraph_number + 1

    for block in iter_docx_blocks(document):
        if isinstance(block, Table):
            table_text = table_to_text([[cell.text for cell in row.cells] for row in block.rows])
            if table_text:
                current_parts.append(table_text)
            continue

        paragraph_number += 1
        text = block.text.strip()
        style_name = str(getattr(block.style, "name", "") or "")
        is_heading = style_name.lower().startswith("heading") or style_name.startswith("標題")
        if is_heading and text:
            flush(paragraph_number - 1)
            current_heading = text
            start_paragraph = paragraph_number
            current_parts.append(text)
        elif text:
            current_parts.append(text)
    flush(paragraph_number)

    if not segments:
        raise DocumentProcessingError("DOCX 沒有可抽取的段落或表格文字。")
    return segments


def decode_text_document(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp950", "big5"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    try:
        from charset_normalizer import from_bytes
        best = from_bytes(raw).best()
        if best is not None and best.encoding:
            return str(best)
    except Exception:
        pass
    raise DocumentProcessingError("文字文件編碼無法辨識，請轉存為 UTF-8 後再上傳。")


def detect_text_document_encoding(path: Path) -> str:
    raw = path.read_bytes()
    for encoding, charset in (
        ("utf-8-sig", "utf-8"),
        ("utf-8", "utf-8"),
        ("cp950", "big5"),
        ("big5", "big5"),
    ):
        try:
            raw.decode(encoding)
            return charset
        except UnicodeDecodeError:
            continue
    try:
        from charset_normalizer import from_bytes

        best = from_bytes(raw).best()
        if best is not None and best.encoding:
            normalized = best.encoding.lower().replace("_", "-")
            return "big5" if normalized in {"cp950", "big5"} else normalized
    except Exception:
        pass
    return "utf-8"


def extract_text_segments(path: Path) -> list[dict[str, Any]]:
    text = decode_text_document(path).replace("\r\n", "\n").replace("\r", "\n")
    lines = text.splitlines()
    if not any(line.strip() for line in lines):
        raise DocumentProcessingError("文字文件沒有可建立索引的內容。")

    segments: list[dict[str, Any]] = []
    current_heading = ""
    block_start = 1
    block_lines: list[str] = []

    def flush(end_line: int) -> None:
        nonlocal block_lines, block_start
        block_text = "\n".join(block_lines).strip()
        if block_text:
            label = current_heading or f"第 {block_start}-{end_line} 行"
            segments.append(make_segment(
                f"lines-{block_start}-{end_line}",
                block_text,
                "lines",
                label,
                section_title=current_heading,
                line_start=block_start,
                line_end=end_line,
            ))
        block_lines = []

    for line_number, line in enumerate(lines, start=1):
        heading_match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", line) if path.suffix.lower() == ".md" else None
        if heading_match:
            flush(line_number - 1)
            current_heading = heading_match.group(1).strip()
            block_start = line_number
            block_lines = [line]
            continue
        if not block_lines:
            block_start = line_number
        block_lines.append(line)
        if len(block_lines) >= 40:
            flush(line_number)
            block_start = line_number + 1
    flush(len(lines))
    return segments


def extract_document_segments(path: Path) -> list[dict[str, Any]]:
    extension = path.suffix.lower()
    if extension == ".pdf":
        return extract_pdf_segments(path)
    if extension == ".docx":
        return extract_docx_segments(path)
    if extension in {".txt", ".md"}:
        return extract_text_segments(path)
    raise DocumentProcessingError("不支援此文件格式。")


def join_segment_text(segments: list[dict[str, Any]]) -> str:
    return "\n\n".join(str(segment.get("text") or "").strip() for segment in segments if str(segment.get("text") or "").strip())


def process_document_upload(notebook_id: str, file: Any, llm_provider: str | None = None) -> dict[str, Any]:
    filename = ensure_unique_document_filename(notebook_id, sanitize_document_filename(file.filename or "document"))
    path = save_document_upload(file, filename)
    source_id = str(uuid.uuid4())
    try:
        validate_document_upload(path, getattr(file, "content_type", None))
        segments = extract_document_segments(path)
        content_text = join_segment_text(segments)
        analysis_mode = "deep" if should_use_deep_analysis(content_text, []) else "quick"
        chunk_count = index_source_document(
            notebook_id, source_id, filename, segments, llm_provider
        )
        saved_at = insert_source_record(
            notebook_id,
            source_id,
            filename,
            content_text,
            "[]",
            analysis_mode,
            "pending",
            None,
            source_type="document",
            mime_type=getattr(file, "content_type", None) or mimetypes.guess_type(filename)[0],
            content_segments_json=json.dumps(segments, ensure_ascii=False),
        )
    except Exception:
        path.unlink(missing_ok=True)
        try:
            delete_source_from_index(notebook_id, source_id)
        except Exception:
            pass
        raise

    return {
        "status": "success",
        "filename": filename,
        "message": f"文件已上傳並建立 {chunk_count} 個索引片段。",
        "analysis_mode": analysis_mode,
        "analysis_status": "pending",
        "source": {
            "id": source_id,
            "filename": filename,
            "source_type": "document",
            "mime_type": getattr(file, "content_type", None) or "",
            "added_at": saved_at,
            "has_transcript": True,
            "has_content": True,
            "analysis_mode": analysis_mode,
            "analysis_status": "pending",
            "indexed_at": saved_at,
            "content_updated_at": saved_at,
        },
    }


def get_document_file_path(notebook_id: str, source_id: str) -> dict[str, Any] | None:
    source = get_source_file_info(notebook_id, source_id)
    if not source or source["source_type"] != "document":
        return None
    path = (DOCUMENT_UPLOAD_DIR / Path(source["filename"]).name).resolve()
    if path.parent != DOCUMENT_UPLOAD_DIR.resolve() or not path.is_file():
        return None
    return {**source, "path": str(path)}


def delete_document_file(filename: str) -> dict[str, Any]:
    path = (DOCUMENT_UPLOAD_DIR / Path(filename or "").name).resolve()
    if path.parent != DOCUMENT_UPLOAD_DIR.resolve() or not path.exists():
        return {"deleted": False, "missing": True, "path": str(path)}
    if not path.is_file():
        return {"deleted": False, "missing": False, "path": str(path)}
    path.unlink()
    return {"deleted": True, "missing": False, "path": str(path)}


def rename_document_file(notebook_id: str, old_filename: str, requested_filename: str) -> str:
    extension = Path(old_filename).suffix.lower()
    requested_stem = Path(requested_filename or "").stem
    safe = sanitize_document_filename(f"{requested_stem}{extension}")
    new_filename = ensure_unique_document_filename(notebook_id, safe, ignore_filename=old_filename)
    old_path = DOCUMENT_UPLOAD_DIR / Path(old_filename).name
    new_path = DOCUMENT_UPLOAD_DIR / new_filename
    if old_path.exists() and old_path.resolve() != new_path.resolve():
        os.replace(old_path, new_path)
    return new_filename
