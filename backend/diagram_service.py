import re
from typing import Any, Literal, cast

from db import insert_diagram_record
from llm_service import generate_text, is_ollama_provider, normalize_llm_provider, stop_llm_generation
from models import clear_cuda_memory
from rag_service import build_recent_history_text, extract_json_object, retrieve_rag_context


DiagramType = Literal["flowchart", "fishbone"]

DIAGRAM_TYPE_LABELS = {
    "flowchart": "流程圖",
    "fishbone": "魚骨圖",
}

DANGEROUS_MERMAID_PATTERNS = (
    "<script",
    "</script",
    "javascript:",
    "onerror=",
    "onload=",
    "click ",
    "href=",
)


def normalize_diagram_type(diagram_type: str | None) -> DiagramType:
    value = (diagram_type or "").strip().lower()
    if value in {"fishbone", "魚骨圖", "鱼骨图", "石川圖", "石川图", "ishikawa", "cause_effect"}:
        return "fishbone"
    return "flowchart"


def detect_requested_diagram_type(text: str) -> DiagramType | None:
    normalized = (text or "").strip().lower()
    if not normalized:
        return None
    fishbone_keywords = ("魚骨圖", "鱼骨图", "石川圖", "石川图", "根因分析", "因果圖", "ishikawa", "fishbone")
    if any(keyword in normalized for keyword in fishbone_keywords):
        return "fishbone"
    flowchart_keywords = ("流程圖", "流程图", "步驟圖", "步骤图", "flowchart")
    if any(keyword in normalized for keyword in flowchart_keywords):
        return "flowchart"
    return None


def select_diagram_references(reference_candidates: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    seen_source_ids: set[str] = set()
    for candidate in reference_candidates:
        reference = cast(dict[str, Any], candidate.get("reference") or {})
        source_id = str(reference.get("source_id") or "")
        if not source_id or source_id in seen_source_ids:
            continue
        seen_source_ids.add(source_id)
        references.append(reference)
        if len(references) >= limit:
            break
    return references


def build_diagram_prompt(
    question: str,
    diagram_type: DiagramType,
    retrieved_context: str,
    history_text: str = "",
) -> str:
    diagram_label = DIAGRAM_TYPE_LABELS[diagram_type]
    if diagram_type == "fishbone":
        diagram_rules = """
- 使用 Mermaid `flowchart LR` 模擬魚骨圖，因 Mermaid 沒有原生 fishbone 語法。
- 右側放核心問題或主題節點，左側放 4 到 6 個主要原因分類。
- 每個主要原因分類下方最多放 1 到 3 個次原因，必須來自參考資料。
- 方向要像「原因匯入結果」，不要做成一般清單。
"""
    else:
        diagram_rules = """
- 使用 Mermaid `flowchart LR`，讓流程由左到右橫向呈現。
- 依照參考資料中的事件、步驟、處理流程或資料流排序。
- 節點最多 10 個；優先保留 6 到 10 個主要節點，避免橫式流程過長。
- 使用箭頭呈現先後、條件或依賴關係。
"""

    history_section = f"\n近期對話：\n{history_text}\n" if history_text else ""
    return f"""你是 VoiceRAG 的知識視覺化助理，任務是根據 RAG 參考資料產生 Mermaid {diagram_label}。

請只輸出 JSON，不要輸出 Markdown，不要輸出說明文字。JSON 格式如下：
{{
  "title": "圖表標題，20 字以內",
  "summary": "用一句話說明這張圖在整理什麼",
  "mermaid_code": "Mermaid 原始碼"
}}

共同規則：
- 使用繁體中文。
- Mermaid 原始碼只能放在 mermaid_code，不要使用 ``` code fence。
- Mermaid 第一行必須是 `flowchart TD`、`flowchart LR`、`graph TD` 或 `graph LR`。
- 節點文字要短，避免過長句子。
- 只能根據參考資料生成，不要補充資料外的新原因、新步驟或新結論。
- 禁止使用 HTML、script、javascript、click、href 等互動或外部連結語法。
{diagram_rules}
{history_section}
使用者需求：
{question}

RAG 參考資料：
{retrieved_context}
"""


def clean_mermaid_code(raw_code: str) -> str:
    code = (raw_code or "").strip()
    if code.startswith("```"):
        code = re.sub(r"^```(?:mermaid)?\s*", "", code, flags=re.IGNORECASE)
        code = re.sub(r"\s*```$", "", code).strip()
    if code.lower().startswith("mermaid\n"):
        code = code.split("\n", 1)[1].strip()

    code = code.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not code:
        raise ValueError("AI 沒有產生 Mermaid 原始碼。")
    if len(code) > 6000:
        raise ValueError("AI 產生的 Mermaid 原始碼過長，請縮小問題範圍後再試。")

    lowered = code.lower()
    if any(pattern in lowered for pattern in DANGEROUS_MERMAID_PATTERNS):
        raise ValueError("Mermaid 原始碼包含不安全語法，已拒絕保存。")

    first_line = next((line.strip().lower() for line in code.split("\n") if line.strip()), "")
    if not (first_line.startswith("flowchart ") or first_line.startswith("graph ")):
        raise ValueError("Mermaid 第一行必須是 flowchart 或 graph。")
    return code


def force_flowchart_lr(code: str) -> str:
    return re.sub(r"^(\s*(?:flowchart|graph)\s+)(TD|TB)\b", r"\1LR", code, count=1, flags=re.IGNORECASE)


def parse_diagram_response(response_text: str, diagram_type: DiagramType) -> dict[str, str]:
    parsed = extract_json_object(response_text)
    if parsed:
        title = str(parsed.get("title") or DIAGRAM_TYPE_LABELS[diagram_type]).strip()
        summary = str(parsed.get("summary") or "").strip()
        mermaid_code = clean_mermaid_code(str(parsed.get("mermaid_code") or ""))
        if diagram_type == "flowchart":
            mermaid_code = force_flowchart_lr(mermaid_code)
        return {
            "title": title[:60] or DIAGRAM_TYPE_LABELS[diagram_type],
            "summary": summary,
            "mermaid_code": mermaid_code,
        }

    mermaid_code = clean_mermaid_code(response_text)
    if diagram_type == "flowchart":
        mermaid_code = force_flowchart_lr(mermaid_code)
    return {
        "title": DIAGRAM_TYPE_LABELS[diagram_type],
        "summary": "",
        "mermaid_code": mermaid_code,
    }


def generate_notebook_diagram(
    notebook_id: str,
    question: str,
    diagram_type: str | None = None,
    llm_provider: str | None = None,
) -> dict[str, Any]:
    provider = normalize_llm_provider(llm_provider)
    question = (question or "").strip()
    if not question:
        return {"status": "error", "message": "請先輸入要整理成圖表的問題或主題。"}
    selected_type = normalize_diagram_type(diagram_type) if diagram_type else detect_requested_diagram_type(question) or "flowchart"
    llm_was_used = False
    try:
        try:
            retrieval = retrieve_rag_context(notebook_id, question, provider)
        except ValueError as error:
            return {"status": "error", "message": str(error)}

        retrieved_context = str(retrieval["retrieved_context"])
        reference_candidates = cast(list[dict[str, Any]], retrieval["reference_candidates"])
        references = select_diagram_references(reference_candidates)
        history_text = build_recent_history_text(notebook_id, limit=4)
        prompt = build_diagram_prompt(question, selected_type, retrieved_context, history_text)

        llm_was_used = True
        response_text = generate_text(prompt, provider)
        parsed = parse_diagram_response(response_text, selected_type)
        diagram = insert_diagram_record(
            notebook_id=notebook_id,
            title=parsed["title"],
            diagram_type=selected_type,
            prompt=question,
            mermaid_code=parsed["mermaid_code"],
            references=references,
        )
        diagram["summary"] = parsed["summary"]

        return {
            "status": "success",
            "response_type": "diagram",
            "question": question,
            "answer": f"已產生「{diagram['title']}」{DIAGRAM_TYPE_LABELS[selected_type]}，可在右側圖表工作區查看。",
            "diagram": diagram,
            "references": references,
        }
    except Exception as error:
        return {"status": "error", "message": f"圖表生成失敗：{str(error)}"}
    finally:
        if llm_was_used and is_ollama_provider(provider):
            stop_llm_generation(provider)
        if is_ollama_provider(provider):
            clear_cuda_memory()
