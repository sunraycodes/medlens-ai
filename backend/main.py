import os
import json
import io
import uuid
from datetime import datetime
import requests
import pdfplumber
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from prompts import EXTRACTION_PROMPT, ANALYSIS_PROMPT, DOCTOR_SUMMARY_PROMPT, RAG_QA_PROMPT
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY not found in environment variables")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS_TO_TRY = [
    "openrouter/free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek/deepseek-r1:free",
    "qwen/qwen-2.5-72b-instruct:free",
]

chroma_client = chromadb.Client()
embedding_fn = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection(
    name="patient_reports",
    embedding_function=embedding_fn
)


def call_ai(prompt: str) -> str:
    last_error = None
    for m in MODELS_TO_TRY:
        try:
            response = requests.post(
                url=OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title": "MedLens AI",
                },
                data=json.dumps({
                    "model": m,
                    "messages": [{"role": "user", "content": prompt}]
                }),
                timeout=60
            )
            result = response.json()
            if "choices" not in result:
                last_error = result
                continue
            text = result["choices"][0]["message"]["content"].strip()
            if text.startswith("```"):
                text = text.strip("`")
                if text.startswith("json"):
                    text = text[4:]
            return text.strip()
        except Exception as e:
            last_error = e
            continue
    raise Exception(f"All models failed: {last_error}")


def extract_text_from_file(file: UploadFile, content: bytes) -> str:
    filename = file.filename.lower()
    if filename.endswith(".pdf"):
        text = ""
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    else:
        return content.decode("utf-8", errors="ignore")


def parse_date_safe(date_str):
    if not date_str:
        return datetime.min

    formats = [
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y/%m/%d"
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    return datetime.min

def compute_trends(valid_reports: list) -> list:
    """Algorithmically detect trends in lab values across reports (no AI)."""
    test_history = {}

    for report in valid_reports:
        date = report.get("date", "")
        for lab in report.get("lab_values", []):
            test_name = lab.get("test", "").strip()
            value_str = lab.get("value", "")
            if not test_name or not value_str:
                continue

            if "/" in value_str:
                continue

            num = ""
            unit = ""
            for ch in value_str:
                if ch.isdigit() or ch == ".":
                    num += ch
                elif num:
                    unit += ch

            try:
                num_val = float(num)
            except ValueError:
                continue

            test_history.setdefault(test_name, []).append({
                "date": date,
                "value": num_val,
                "unit": unit.strip()
            })

    trends = []
    for test_name, history in test_history.items():
        if len(history) < 2:
            continue
        history.sort(key=lambda x: x["date"])
        first = history[0]
        last = history[-1]
        delta = round(last["value"] - first["value"], 2)

        if delta > 0:
            direction = "increasing"
        elif delta < 0:
            direction = "decreasing"
        else:
            direction = "stable"

        trends.append({
            "test": test_name,
            "history": history,
            "direction": direction,
            "change": delta,
            "unit": last["unit"],
            "first_value": first["value"],
            "first_date": first["date"],
            "last_value": last["value"],
            "last_date": last["date"]
        })

    return trends


def build_knowledge_graph(patient_summary: dict) -> dict:
    """Build a simple patient -> conditions/medications/allergies graph (no AI)."""
    nodes = []
    edges = []

    nodes.append({"id": "patient", "label": "Patient", "type": "patient"})

    for condition in patient_summary.get("conditions", []):
        node_id = f"condition_{condition}"
        nodes.append({"id": node_id, "label": condition, "type": "condition"})
        edges.append({"source": "patient", "target": node_id, "relation": "diagnosed_with"})

    for med in patient_summary.get("current_medications", []):
        node_id = f"medication_{med}"
        nodes.append({"id": node_id, "label": med, "type": "medication"})
        edges.append({"source": "patient", "target": node_id, "relation": "prescribed"})

    for allergy in patient_summary.get("allergies", []):
        node_id = f"allergy_{allergy}"
        nodes.append({"id": node_id, "label": allergy, "type": "allergy"})
        edges.append({"source": "patient", "target": node_id, "relation": "allergic_to"})

    return {"nodes": nodes, "edges": edges}


@app.post("/process")
async def process_reports(files: list[UploadFile] = File(...)):
    extracted_reports = []

    try:
        existing = collection.get()
        if existing and existing.get("ids"):
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    for f in files:
        content = await f.read()
        text = extract_text_from_file(f, content)

        if not text.strip():
            extracted_reports.append({"error": "empty_or_unreadable", "filename": f.filename})
            continue

        try:
            collection.add(
                documents=[text],
                metadatas=[{"filename": f.filename}],
                ids=[f.filename]
            )
        except Exception:
            pass

        prompt = EXTRACTION_PROMPT.replace("<<report_text>>", text)
        parsed = None
        for attempt in range(2):
            result = call_ai(prompt)
            try:
                parsed = json.loads(result)
                break
            except json.JSONDecodeError:
                continue

        if parsed:
            parsed["source_file"] = f.filename
            extracted_reports.append(parsed)
        else:
            extracted_reports.append({"error": "parse_failed", "raw": result, "filename": f.filename})

    valid_reports = [r for r in extracted_reports if "error" not in r]
    valid_reports.sort(
    key=lambda x: parse_date_safe(x.get("date", ""))
    )

    analysis_prompt = ANALYSIS_PROMPT.replace("<<n>>", str(len(valid_reports)))
    analysis_prompt = analysis_prompt.replace("<<json_data>>", json.dumps(valid_reports, indent=2))
    analysis_result = call_ai(analysis_prompt)
    analysis = json.loads(analysis_result)
    trends = compute_trends(valid_reports)
    knowledge_graph = build_knowledge_graph(analysis.get("patient_summary", {}))

    summary_prompt = DOCTOR_SUMMARY_PROMPT.replace("<<json_data>>", json.dumps(analysis, indent=2))
    doctor_summary = call_ai(summary_prompt)

    return {
        "extracted_reports": extracted_reports,
        "analysis": analysis,
        "doctor_summary": doctor_summary,
        "trends": trends,
        "knowledge_graph": knowledge_graph
    }


@app.post("/ask")
async def ask_question(payload: dict):
    question = payload.get("question", "")
    if not question.strip():
        return {"answer": "Please provide a question.", "sources": []}

    try:
        results = collection.query(query_texts=[question], n_results=3)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
    except Exception:
        docs, metas = [], []

    if not docs:
        return {"answer": "No patient documents have been processed yet. Please upload reports first.", "sources": []}

    context = ""
    for doc, meta in zip(docs, metas):
        context += f"\n[From {meta['filename']}]:\n{doc}\n"

    prompt = RAG_QA_PROMPT.replace("<<context>>", context)
    prompt = prompt.replace("<<question>>", question)
    answer = call_ai(prompt)

    return {"answer": answer, "sources": [m["filename"] for m in metas]}


@app.post("/export-pdf")
async def export_pdf(payload: dict):
    """Takes {'doctor_summary': '...'} and returns a downloadable PDF"""
    text = payload.get("doctor_summary", "")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("MedLens AI — Doctor Handoff Summary", styles["Title"]))
    story.append(Spacer(1, 12))

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            story.append(Spacer(1, 8))
            continue
        if line.isupper() and len(line) < 60:
            story.append(Paragraph(f"<b>{line}</b>", styles["Heading3"]))
        elif line.startswith("-"):
            story.append(Paragraph(f"• {line[1:].strip()}", styles["Normal"]))
        else:
            story.append(Paragraph(line, styles["Normal"]))

    doc.build(story)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=doctor_summary.pdf"}
    )


@app.get("/")
def root():
    return {"status": "MedLens AI backend running"}
