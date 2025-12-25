import json
import re
from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel


app = FastAPI()

TOOL_SEARCH_TAG = "tool_search"
TOOL_SEARCH_RESULT_TAG = "tool_search_result"
TOOL_CALL_TAG = "tool_call"
TOOL_CALL_RESULT_TAG = "tool_call_result"


class TraceRequest(BaseModel):
    trace: str


def _strip_tags(text: str) -> str:
    cleaned = re.sub(
        r"<(tool_search|tool_search_result|tool_call|tool_call_result)>.*?</\1>",
        "",
        text,
        flags=re.DOTALL,
    )
    cleaned = cleaned.replace("<fold_thought>", "")
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _extract_plan_lines(text: str) -> str:
    plan_lines = []
    for line in text.splitlines():
        if re.match(r"^\s*(Plan|规划|计划|步骤)[:：]", line, flags=re.IGNORECASE):
            plan_lines.append(line.strip())
    return "\n".join(plan_lines).strip()


def _truncate_label(label: str, max_len: int = 60) -> str:
    label = re.sub(r"\s+", " ", label).strip()
    if len(label) <= max_len:
        return label
    return f"{label[: max_len - 1]}…"


def _safe_label(label: str) -> str:
    return _truncate_label(label).replace('"', "'")


def _parse_tool_search_results(content: str) -> List[str]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        names = []
        for item in data:
            if isinstance(item, dict):
                name = item.get("name") or item.get("function", {}).get("name")
                if name:
                    names.append(str(name))
        return names
    return []


def _extract_steps(trace: str) -> List[Dict[str, Any]]:
    steps = []
    for tag in [
        TOOL_SEARCH_TAG,
        TOOL_SEARCH_RESULT_TAG,
        TOOL_CALL_TAG,
        TOOL_CALL_RESULT_TAG,
    ]:
        pattern = rf"<{tag}>(.*?)</{tag}>"
        for match in re.finditer(pattern, trace, flags=re.DOTALL):
            content = match.group(1).strip()
            steps.append(
                {
                    "type": tag,
                    "content": content,
                    "start": match.start(),
                }
            )
    return sorted(steps, key=lambda item: item["start"])


def _build_mermaid(steps: List[Dict[str, Any]]) -> str:
    if not steps:
        return "graph LR\n  A[\"暂无工具路径\"]"
    nodes = []
    edges = []
    for idx, step in enumerate(steps, start=1):
        node_id = f"step{idx}"
        label = step["content"]
        if step["type"] == TOOL_SEARCH_TAG:
            label = f"Tool Search: {label}"
        elif step["type"] == TOOL_SEARCH_RESULT_TAG:
            tool_names = _parse_tool_search_results(step["content"])
            if tool_names:
                label = f"Tool Gen: {', '.join(tool_names)}"
            else:
                label = "Tool Gen: 搜索结果"
        elif step["type"] == TOOL_CALL_TAG:
            try:
                payload = json.loads(step["content"])
                label = f"Tool Call: {payload.get('name', 'tool')}"
            except json.JSONDecodeError:
                label = "Tool Call"
        elif step["type"] == TOOL_CALL_RESULT_TAG:
            label = "Tool Result"
        nodes.append(f'  {node_id}["{_safe_label(label)}"]')
        if idx > 1:
            edges.append(f"  step{idx - 1} --> {node_id}")
    return "graph LR\n" + "\n".join(nodes + edges)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(
        """
<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <title>Agent 轨迹可视化</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
      body { font-family: "Segoe UI", Arial, sans-serif; margin: 0; background: #f7f7fb; color: #222; }
      header { padding: 24px 32px; background: #1f3a8a; color: #fff; }
      main { padding: 24px 32px; display: grid; gap: 20px; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
      .card { background: #fff; border-radius: 12px; padding: 20px; box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08); }
      textarea { width: 100%; height: 160px; border-radius: 8px; border: 1px solid #d9d9e3; padding: 12px; font-family: "JetBrains Mono", monospace; }
      button { background: #1d4ed8; color: #fff; border: none; border-radius: 8px; padding: 10px 16px; cursor: pointer; }
      button:hover { background: #1e40af; }
      .section-title { font-size: 16px; font-weight: 600; margin-bottom: 12px; }
      .pill { display: inline-block; padding: 4px 10px; border-radius: 999px; background: #e0e7ff; color: #1e3a8a; margin: 4px 6px 0 0; }
      .output { white-space: pre-wrap; background: #f3f4f6; padding: 12px; border-radius: 8px; min-height: 140px; }
      .modal { display: none; position: fixed; z-index: 999; inset: 0; background: rgba(15, 23, 42, 0.6); }
      .modal-content { background: #fff; margin: 6% auto; padding: 24px; border-radius: 12px; width: min(900px, 90%); }
      .modal-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
      .close { font-size: 24px; cursor: pointer; }
      .mermaid { background: #f8fafc; padding: 12px; border-radius: 10px; }
    </style>
  </head>
  <body>
    <header>
      <h1>多工具 Agent 思考与路径可视化</h1>
      <p>仅展示思考 / 规划摘要、工具生成列表与调用路径图。</p>
    </header>
    <main>
      <section class="card">
        <div class="section-title">Agent 轨迹输入</div>
        <textarea id="traceInput" placeholder="粘贴包含 <tool_search> / <tool_call> 等标签的 agent 输出..."></textarea>
        <div style="margin-top: 12px; display: flex; gap: 12px;">
          <button id="renderBtn">解析并渲染</button>
          <button id="openGraphBtn">查看工具调用路径图</button>
        </div>
      </section>
      <section class="card">
        <div class="section-title">思考与规划</div>
        <div id="thoughtOutput" class="output">暂无内容</div>
      </section>
      <section class="card">
        <div class="section-title">工具生成列表</div>
        <div id="toolList">暂无工具</div>
      </section>
    </main>

    <div id="graphModal" class="modal">
      <div class="modal-content">
        <div class="modal-header">
          <div class="section-title">工具调用路径图</div>
          <span class="close" id="closeModal">&times;</span>
        </div>
        <div id="graphContainer"></div>
      </div>
    </div>

    <script>
      mermaid.initialize({ startOnLoad: false });
      const renderBtn = document.getElementById("renderBtn");
      const openGraphBtn = document.getElementById("openGraphBtn");
      const modal = document.getElementById("graphModal");
      const closeModal = document.getElementById("closeModal");
      const thoughtOutput = document.getElementById("thoughtOutput");
      const toolList = document.getElementById("toolList");
      const graphContainer = document.getElementById("graphContainer");

      function renderMermaid(code) {
        graphContainer.innerHTML = '<div class="mermaid">' + code + '</div>';
        mermaid.init(undefined, graphContainer.querySelectorAll(".mermaid"));
      }

      renderBtn.addEventListener("click", async () => {
        const trace = document.getElementById("traceInput").value;
        const response = await fetch("/parse", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ trace })
        });
        const data = await response.json();
        thoughtOutput.textContent = data.thoughts || "暂无内容";
        toolList.innerHTML = "";
        if (data.tools && data.tools.length) {
          data.tools.forEach((tool) => {
            const pill = document.createElement("span");
            pill.className = "pill";
            pill.textContent = tool;
            toolList.appendChild(pill);
          });
        } else {
          toolList.textContent = "暂无工具";
        }
        renderMermaid(data.mermaid || "graph LR\\n  A[\"暂无工具路径\"]");
      });

      openGraphBtn.addEventListener("click", () => {
        modal.style.display = "block";
      });

      closeModal.addEventListener("click", () => {
        modal.style.display = "none";
      });

      window.addEventListener("click", (event) => {
        if (event.target === modal) {
          modal.style.display = "none";
        }
      });
    </script>
  </body>
</html>
        """
    )


@app.post("/parse")
def parse_trace(payload: TraceRequest) -> JSONResponse:
    trace = payload.trace or ""
    steps = _extract_steps(trace)
    tool_names = []
    for step in steps:
        if step["type"] == TOOL_SEARCH_RESULT_TAG:
            tool_names.extend(_parse_tool_search_results(step["content"]))
        if step["type"] == TOOL_CALL_TAG:
            try:
                payload = json.loads(step["content"])
                name = payload.get("name")
                if name:
                    tool_names.append(str(name))
            except json.JSONDecodeError:
                continue

    cleaned_text = _strip_tags(trace)
    plan_text = _extract_plan_lines(cleaned_text)
    thoughts = cleaned_text if cleaned_text else "暂无内容"
    if plan_text:
        thoughts = f"{thoughts}\n\n规划摘要:\n{plan_text}"

    return JSONResponse(
        {
            "thoughts": thoughts,
            "tools": tool_names,
            "mermaid": _build_mermaid(steps),
        }
    )
