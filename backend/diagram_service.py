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


class DiagramGenerationError(ValueError):
    def __init__(self, message: str, error_type: str = "diagram_plan_invalid"):
        super().__init__(message)
        self.error_type = error_type


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
- 只輸出 `fishbone_plan`，不要輸出 Mermaid；Mermaid 會由後端模板穩定產生。
- `effect` 是右側魚頭，只能是使用者問題或參考資料可支持的核心問題。
- `categories` 是主要原因分類，必須 3 到 6 類；資料足夠時優先 4 到 6 類，資料不足時可合併成 3 類，但不能憑空補原因。
- 每個主要原因分類底下放 1 到 3 個次原因，次原因與 evidence 都必須來自 RAG 參考資料。
- 分類名稱要短，例如「人員」「方法」「設備」「環境」「資料」「流程」。
- 禁止產生時間線、一般流程圖、心智圖或普通條列關係圖。
"""
        json_schema = """{
  "title": "圖表標題，20 字以內",
  "summary": "用一句話說明這張圖在整理什麼",
  "fishbone_plan": {
    "effect": "右側核心問題",
    "categories": [
      {
        "name": "主要原因分類，例如人員、方法、設備、環境、資料、流程",
        "causes": [
          {
            "label": "次原因短句",
            "evidence": "支持此原因的參考資料依據"
          }
        ]
      }
    ],
    "causes": ["所有次原因摘要"],
    "evidence": ["使用到的參考資料依據摘要"]
  }
}"""
    else:
        diagram_rules = """
- 只輸出 `flow_plan`，不要輸出 Mermaid；Mermaid 會由後端模板穩定產生。
- 若資料是一串事件或時間線，請整理成「開始 -> 主要階段 -> 判斷或處理 -> 結果 -> 結束」，不要只做水平時間軸。
- 節點建議 6 到 12 個；資訊太多時要合併為階段，不要把每個細節都畫成節點。
- 節點 ID 必須使用簡短英數字，例如 `A`、`B1`、`Decision1`；中文只放在節點顯示文字。
- 每個 node 必須包含 `id`、`label`、`type`。
- node.type 只能使用：`start`、`end`、`process`、`decision`、`input_output`、`document`、`subprocess`、`database`。
- 每個 edge 必須包含 `from`、`to`，可選 `label`。
- 如果使用 `decision`，該節點後方必須至少有兩條 edge，且 edge.label 必須標示條件，例如「是」「否」。
- 沒有明確分支條件時，不要使用 `decision`，請使用 `process`。
"""
        json_schema = """{
  "title": "圖表標題，20 字以內",
  "summary": "用一句話說明這張圖在整理什麼",
  "flow_plan": {
    "goal": "這張流程圖要說明的流程目標",
    "nodes": [
      {"id": "Start", "label": "開始", "type": "start"},
      {"id": "Step1", "label": "主要步驟", "type": "process"},
      {"id": "End", "label": "結束", "type": "end"}
    ],
    "edges": [
      {"from": "Start", "to": "Step1", "label": ""},
      {"from": "Step1", "to": "End", "label": ""}
    ]
  }
}"""

    history_section = f"\n近期對話：\n{history_text}\n" if history_text else ""
    return f"""你是 VoiceRAG 的知識視覺化助理，任務是根據 RAG 參考資料產生 Mermaid {diagram_label}。

請只輸出 JSON，不要輸出 Markdown，不要輸出說明文字。JSON 格式如下：
{json_schema}

共同規則：
- 使用繁體中文。
- 不要輸出 Mermaid 原始碼，不要輸出 ``` code fence。
- 節點與原因文字要短，避免過長句子。
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
        raise DiagramGenerationError("AI 沒有產生 Mermaid 原始碼。", "diagram_render_invalid")
    if len(code) > 6000:
        raise DiagramGenerationError("AI 產生的 Mermaid 原始碼過長，請縮小問題範圍後再試。", "diagram_render_invalid")

    lowered = code.lower()
    if any(pattern in lowered for pattern in DANGEROUS_MERMAID_PATTERNS):
        raise DiagramGenerationError("Mermaid 原始碼包含不安全語法，已拒絕保存。", "unsafe_mermaid")

    first_line = next((line.strip().lower() for line in code.split("\n") if line.strip()), "")
    if not (first_line.startswith("flowchart ") or first_line.startswith("graph ")):
        raise DiagramGenerationError("Mermaid 第一行必須是 flowchart 或 graph。", "diagram_render_invalid")
    return code


def force_flowchart_lr(code: str) -> str:
    return re.sub(
        r"^\s*(?:flowchart|graph)\s+(TD|TB|LR|RL)\b",
        "flowchart LR",
        code,
        count=1,
        flags=re.IGNORECASE,
    )


def build_diagram_repair_prompt(
    question: str,
    diagram_type: DiagramType,
    retrieved_context: str,
    previous_response: str,
    error_message: str,
) -> str:
    base_prompt = build_diagram_prompt(question, diagram_type, retrieved_context)
    return f"""{base_prompt}

上一次輸出沒有通過後端驗證，請只修正 JSON plan，不要輸出 Mermaid，不要解釋。

驗證錯誤：
{error_message}

上一次輸出：
{previous_response[:4000]}
"""


def sanitize_mermaid_label(value: Any, fallback: str, max_length: int = 32) -> str:
    label = str(value or "").strip() or fallback
    label = re.sub(r"\s+", " ", label)
    replacements = {
        "[": "【",
        "]": "】",
        "{": "｛",
        "}": "｝",
        "|": "｜",
        "<": "＜",
        ">": "＞",
    }
    for source, target in replacements.items():
        label = label.replace(source, target)
    for pattern in DANGEROUS_MERMAID_PATTERNS:
        label = label.replace(pattern, pattern.replace(":", "：").replace("=", "＝"))
    if len(label) > max_length:
        label = f"{label[:max_length - 1]}…"
    return label or fallback


def normalize_node_type(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "begin": "start",
        "terminal": "start",
        "finish": "end",
        "stop": "end",
        "action": "process",
        "step": "process",
        "task": "process",
        "data": "input_output",
        "input": "input_output",
        "output": "input_output",
        "io": "input_output",
        "report": "document",
        "file": "document",
        "sub_process": "subprocess",
        "phase": "subprocess",
        "db": "database",
        "knowledge_base": "database",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in {"start", "end", "process", "decision", "input_output", "document", "subprocess", "database"}:
        return normalized
    return "process"


def make_flow_node(source_id: str, label: Any, node_type: Any) -> dict[str, str]:
    return {
        "source_id": source_id,
        "label": sanitize_mermaid_label(label, source_id, max_length=30),
        "type": normalize_node_type(node_type),
    }


def build_legacy_flow_nodes(flow_plan: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    nodes: list[dict[str, str]] = []
    edges: list[dict[str, str]] = []
    start_label = str(flow_plan.get("start") or "開始").strip()
    end_label = str(flow_plan.get("end") or "結束").strip()
    main_steps = flow_plan.get("main_steps") or []
    if not isinstance(main_steps, list):
        main_steps = []

    nodes.append(make_flow_node("Start", start_label, "start"))
    for index, step in enumerate(main_steps[:10], start=1):
        nodes.append(make_flow_node(f"Step{index}", step, "process"))
    nodes.append(make_flow_node("End", end_label, "end"))

    for index in range(len(nodes) - 1):
        edges.append({"from": nodes[index]["source_id"], "to": nodes[index + 1]["source_id"], "label": ""})
    return nodes, edges


def normalize_flow_diagram_plan(parsed: dict[str, Any], allow_decision_downgrade: bool = False) -> dict[str, Any]:
    flow_plan = parsed.get("flow_plan")
    if not isinstance(flow_plan, dict):
        raise DiagramGenerationError("AI 輸出格式需要重試：缺少 flow_plan。")

    raw_nodes = flow_plan.get("nodes")
    raw_edges = flow_plan.get("edges")
    if not isinstance(raw_nodes, list):
        raw_nodes, raw_edges = build_legacy_flow_nodes(flow_plan)

    nodes: list[dict[str, str]] = []
    seen_source_ids: set[str] = set()
    for index, raw_node in enumerate(raw_nodes, start=1):
        if not isinstance(raw_node, dict):
            continue
        source_id = str(raw_node.get("id") or raw_node.get("source_id") or f"N{index}").strip() or f"N{index}"
        if source_id in seen_source_ids:
            source_id = f"{source_id}_{index}"
        seen_source_ids.add(source_id)
        nodes.append(make_flow_node(source_id, raw_node.get("label") or raw_node.get("name") or source_id, raw_node.get("type")))

    if len(nodes) < 3:
        raise DiagramGenerationError("目前筆記本資料不足以生成標準流程圖。", "insufficient_context")

    if not any(node["type"] == "start" for node in nodes):
        nodes.insert(0, make_flow_node("Start", flow_plan.get("start") or "開始", "start"))
    if not any(node["type"] == "end" for node in nodes):
        nodes.append(make_flow_node("End", flow_plan.get("end") or "結束", "end"))

    known_source_ids = {node["source_id"] for node in nodes}
    edges: list[dict[str, str]] = []
    if isinstance(raw_edges, list):
        for raw_edge in raw_edges:
            if not isinstance(raw_edge, dict):
                continue
            source = str(raw_edge.get("from") or "").strip()
            target = str(raw_edge.get("to") or "").strip()
            if source in known_source_ids and target in known_source_ids and source != target:
                edges.append({
                    "from": source,
                    "to": target,
                    "label": sanitize_mermaid_label(raw_edge.get("label") or "", "", max_length=14),
                })

    if not edges:
        for index in range(len(nodes) - 1):
            edges.append({"from": nodes[index]["source_id"], "to": nodes[index + 1]["source_id"], "label": ""})

    outgoing_by_source: dict[str, list[dict[str, str]]] = {}
    for edge in edges:
        outgoing_by_source.setdefault(edge["from"], []).append(edge)

    for node in nodes:
        if node["type"] != "decision":
            continue
        outgoing = outgoing_by_source.get(node["source_id"], [])
        labeled = [edge for edge in outgoing if edge["label"]]
        if len(outgoing) < 2 or len(labeled) < 2:
            if allow_decision_downgrade:
                node["type"] = "process"
                for edge in outgoing:
                    edge["label"] = ""
            else:
                raise DiagramGenerationError("判斷節點缺少兩條已標註條件的後續分支。")

    return {
        "title": str(parsed.get("title") or "流程圖").strip()[:60] or "流程圖",
        "summary": str(parsed.get("summary") or "").strip(),
        "nodes": nodes[:14],
        "edges": edges,
    }


def render_flowchart_mermaid(plan: dict[str, Any]) -> str:
    nodes = cast(list[dict[str, str]], plan["nodes"])
    edges = cast(list[dict[str, str]], plan["edges"])
    id_map = {node["source_id"]: f"N{index}" for index, node in enumerate(nodes, start=1)}
    shape_templates = {
        "start": "{id}([{label}])",
        "end": "{id}([{label}])",
        "process": "{id}[{label}]",
        "decision": "{id}{{{label}}}",
        "input_output": "{id}[/{label}/]",
        "document": "{id}[/{label}/]",
        "subprocess": "{id}[[{label}]]",
        "database": "{id}[({label})]",
    }

    lines = ["flowchart LR"]
    for node in nodes:
        node_id = id_map[node["source_id"]]
        template = shape_templates.get(node["type"], shape_templates["process"])
        lines.append(f"  {template.format(id=node_id, label=node['label'])}")

    known_ids = set(id_map)
    for edge in edges:
        source = edge["from"]
        target = edge["to"]
        if source not in known_ids or target not in known_ids:
            continue
        label = edge.get("label", "")
        if label:
            lines.append(f"  {id_map[source]} -->|{label}| {id_map[target]}")
        else:
            lines.append(f"  {id_map[source]} --> {id_map[target]}")

    lines.extend([
        "  classDef terminal fill:#2f3b52,stroke:#8fb4ff,color:#f8fafc,stroke-width:1px;",
        "  classDef decision fill:#3b314d,stroke:#c4a2ff,color:#f8fafc,stroke-width:1px;",
        "  classDef data fill:#243f3a,stroke:#7dd3c7,color:#f8fafc,stroke-width:1px;",
    ])
    for node in nodes:
        node_id = id_map[node["source_id"]]
        if node["type"] in {"start", "end"}:
            lines.append(f"  class {node_id} terminal;")
        elif node["type"] == "decision":
            lines.append(f"  class {node_id} decision;")
        elif node["type"] in {"input_output", "document", "database"}:
            lines.append(f"  class {node_id} data;")
    return "\n".join(lines)


def normalize_cause(raw_cause: Any, fallback: str) -> dict[str, str]:
    if isinstance(raw_cause, dict):
        label = raw_cause.get("label") or raw_cause.get("name") or raw_cause.get("cause") or fallback
        evidence = raw_cause.get("evidence") or ""
    else:
        label = raw_cause or fallback
        evidence = ""
    return {
        "label": sanitize_mermaid_label(label, fallback, max_length=28),
        "evidence": str(evidence or "").strip(),
    }


def normalize_fishbone_diagram_plan(parsed: dict[str, Any]) -> dict[str, Any]:
    fishbone_plan = parsed.get("fishbone_plan")
    if not isinstance(fishbone_plan, dict):
        raise DiagramGenerationError("AI 輸出格式需要重試：缺少 fishbone_plan。")

    effect = sanitize_mermaid_label(fishbone_plan.get("effect"), "核心問題", max_length=34)
    raw_categories = fishbone_plan.get("categories") or []
    categories: list[dict[str, Any]] = []
    if isinstance(raw_categories, list):
        for index, raw_category in enumerate(raw_categories, start=1):
            if not isinstance(raw_category, dict):
                continue
            raw_causes = raw_category.get("causes") or []
            if not isinstance(raw_causes, list):
                raw_causes = []
            causes = [
                normalize_cause(raw_cause, f"原因 {cause_index}")
                for cause_index, raw_cause in enumerate(raw_causes, start=1)
                if str(raw_cause or "").strip()
            ]
            if causes:
                categories.append({
                    "name": sanitize_mermaid_label(raw_category.get("name"), f"分類 {index}", max_length=12),
                    "causes": causes[:3],
                })

    if len(categories) < 3:
        raw_causes = fishbone_plan.get("causes") or []
        flattened_causes: list[dict[str, str]] = []
        if isinstance(raw_causes, list):
            flattened_causes = [
                normalize_cause(raw_cause, f"原因 {index}")
                for index, raw_cause in enumerate(raw_causes, start=1)
                if str(raw_cause or "").strip()
            ]
        if not flattened_causes:
            for category in categories:
                flattened_causes.extend(cast(list[dict[str, str]], category["causes"]))
        if len(flattened_causes) >= 3:
            fallback_names = ["人員", "方法", "流程", "資料", "環境", "設備"]
            categories = [
                {"name": fallback_names[index], "causes": [flattened_causes[index]]}
                for index in range(3)
            ]

    if len(categories) < 3:
        raise DiagramGenerationError("目前筆記本資料不足以生成標準魚骨圖。", "insufficient_context")

    categories = categories[:6]
    return {
        "title": str(parsed.get("title") or "魚骨圖").strip()[:60] or "魚骨圖",
        "summary": str(parsed.get("summary") or "").strip(),
        "effect": effect,
        "categories": categories,
    }


def render_fishbone_mermaid(plan: dict[str, Any]) -> str:
    categories = cast(list[dict[str, Any]], plan["categories"])
    spine_count = max(3, min(6, len(categories)))
    lines = ["flowchart LR"]
    lines.append(f"  Effect([{plan['effect']}])")
    for index in range(1, spine_count + 1):
        lines.append(f"  S{index}((主骨))")
    for index in range(1, spine_count):
        lines.append(f"  S{index} --- S{index + 1}")
    lines.append(f"  S{spine_count} --> Effect")

    for category_index, category in enumerate(categories, start=1):
        category_id = f"C{category_index}"
        spine_id = f"S{min(category_index, spine_count)}"
        lines.append(f"  {category_id}[{category['name']}] --> {spine_id}")
        causes = cast(list[dict[str, str]], category["causes"])
        for cause_index, cause in enumerate(causes, start=1):
            cause_id = f"C{category_index}{chr(96 + cause_index)}"
            lines.append(f"  {cause_id}[{cause['label']}] --> {category_id}")

    lines.extend([
        "  classDef effect fill:#3f4358,stroke:#9ca3ff,color:#f8fafc,stroke-width:1.5px;",
        "  classDef spine fill:#2a2d34,stroke:#94a3b8,color:#cbd5e1,stroke-width:1px;",
        "  classDef category fill:#344153,stroke:#93c5fd,color:#f8fafc,stroke-width:1px;",
        "  classDef cause fill:#23272f,stroke:#64748b,color:#e5e7eb,stroke-width:1px;",
        "  class Effect effect;",
        f"  class {','.join(f'S{index}' for index in range(1, spine_count + 1))} spine;",
        f"  class {','.join(f'C{index}' for index in range(1, len(categories) + 1))} category;",
    ])
    cause_ids = [
        f"C{category_index}{chr(96 + cause_index)}"
        for category_index, category in enumerate(categories, start=1)
        for cause_index, _cause in enumerate(cast(list[dict[str, str]], category["causes"]), start=1)
    ]
    if cause_ids:
        lines.append(f"  class {','.join(cause_ids)} cause;")
    return "\n".join(lines)


def build_diagram_from_plan_response(
    response_text: str,
    diagram_type: DiagramType,
    allow_decision_downgrade: bool = False,
) -> dict[str, Any]:
    parsed = extract_json_object(response_text)
    if not isinstance(parsed, dict):
        raise DiagramGenerationError("AI 輸出格式需要重試：沒有回傳有效 JSON。")

    if diagram_type == "flowchart":
        plan = normalize_flow_diagram_plan(parsed, allow_decision_downgrade)
        mermaid_code = render_flowchart_mermaid(plan)
    else:
        plan = normalize_fishbone_diagram_plan(parsed)
        mermaid_code = render_fishbone_mermaid(plan)

    mermaid_code = force_flowchart_lr(clean_mermaid_code(mermaid_code))
    return {
        "title": str(plan["title"]),
        "summary": str(plan.get("summary") or ""),
        "mermaid_code": mermaid_code,
        "diagram_data": {
            "diagram_type": diagram_type,
            "title": str(plan["title"]),
            "summary": str(plan.get("summary") or ""),
            "plan": plan,
        },
    }


def parse_diagram_response(response_text: str, diagram_type: DiagramType) -> dict[str, Any]:
    return build_diagram_from_plan_response(response_text, diagram_type)


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
            return {
                "status": "error",
                "error_type": "insufficient_context",
                "message": f"目前筆記本資料不足以生成標準{DIAGRAM_TYPE_LABELS[selected_type]}。{str(error)}",
            }

        retrieved_context = str(retrieval["retrieved_context"])
        reference_candidates = cast(list[dict[str, Any]], retrieval["reference_candidates"])
        references = select_diagram_references(reference_candidates)
        history_text = build_recent_history_text(notebook_id, limit=4)
        prompt = build_diagram_prompt(question, selected_type, retrieved_context, history_text)

        llm_was_used = True
        response_text = generate_text(prompt, provider)
        try:
            parsed = build_diagram_from_plan_response(response_text, selected_type)
        except DiagramGenerationError as first_error:
            if first_error.error_type != "diagram_plan_invalid":
                raise
            repair_prompt = build_diagram_repair_prompt(
                question,
                selected_type,
                retrieved_context,
                response_text,
                str(first_error),
            )
            repaired_response_text = generate_text(repair_prompt, provider)
            parsed = build_diagram_from_plan_response(
                repaired_response_text,
                selected_type,
                allow_decision_downgrade=True,
            )
        diagram = insert_diagram_record(
            notebook_id=notebook_id,
            title=parsed["title"],
            diagram_type=selected_type,
            prompt=question,
            mermaid_code=parsed["mermaid_code"],
            references=references,
            diagram_data=cast(dict[str, Any] | None, parsed.get("diagram_data")),
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
    except DiagramGenerationError as error:
        return {
            "status": "error",
            "error_type": error.error_type,
            "message": str(error),
        }
    except Exception as error:
        return {
            "status": "error",
            "error_type": "diagram_render_invalid",
            "message": f"圖表生成失敗：{str(error)}",
        }
    finally:
        if llm_was_used and is_ollama_provider(provider):
            stop_llm_generation(provider)
        if is_ollama_provider(provider):
            clear_cuda_memory()
