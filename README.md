# MedLens AI

**An AI-powered second opinion and patient history reconstruction system.**

MedLens AI helps doctors quickly understand a patient's medical history by analyzing multiple reports (blood tests, prescriptions, discharge summaries, scan reports) and generating a timeline, risk flags, and a one-page handoff summary — all in under a minute.

## Problem

Patients accumulate years of medical reports scattered across visits and formats. Doctors often have only a few minutes per consultation and may miss patterns hidden across documents like gradually worsening lab values or missed follow-ups.

## What It Does

Given multiple patient reports (PDF or text), MedLens AI:

1. **Extracts structured data** from each report (diagnoses, lab values, medications, allergies) using LLM-based extraction
2. **Builds a Health Timeline** showing how the patient's condition evolved across visits
3. **Generates a Patient Summary** (conditions, current medications, allergies)
4. **Detects Risk Flags** — cross-document inconsistencies, missed follow-ups, worsening trends, medication concerns
5. **Computes Lab Trends algorithmically** — e.g., HbA1c rising from 5.9% → 6.8% → 7.4% across visits, detected via direct numeric comparison (no AI)
6. **Builds a Knowledge Graph** linking the patient to conditions, medications, and allergies
7. **Answers follow-up questions** via RAG (Retrieval-Augmented Generation) — semantically retrieves relevant report excerpts from a vector database and cites sources
8. **Generates a one-page Doctor Handoff Summary**, downloadable as a PDF

## Tech Stack

- **Backend**: FastAPI (Python)
- **AI/LLM**: OpenRouter (multi-model fallback for reliability)
- **Vector DB / RAG**: ChromaDB + Sentence-Transformers embeddings
- **PDF Processing**: pdfplumber (extraction), ReportLab (export)
- **Frontend**: React / Next.js + Tailwind

## API Endpoints

### `POST /process`
Upload multiple reports (`.txt` or `.pdf`) as multipart form-data (field name: `files`).

Returns:
```json
{
  "extracted_reports": [...],
  "analysis": {
    "patient_summary": { "age": ..., "conditions": [...], "current_medications": [...], "allergies": [...] },
    "timeline": [{ "date": "YYYY-MM", "event": "..." }],
    "risk_flags": [{ "severity": "high/medium/low", "flag": "...", "explanation": "..." }]
  },
  "doctor_summary": "plain text one-page summary",
  "trends": [{ "test": "...", "history": [...], "direction": "increasing/decreasing/stable", "change": ... }],
  "knowledge_graph": { "nodes": [...], "edges": [...] }
}
```

### `POST /ask`
Body: `{ "question": "..." }`

Returns: `{ "answer": "...", "sources": ["filename1", ...] }`

RAG-based Q&A — retrieves relevant excerpts from processed reports via ChromaDB and answers with source citations.

### `POST /export-pdf`
Body: `{ "doctor_summary": "..." }`

Returns a downloadable PDF of the doctor handoff summary.

## Setup & Running Locally

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file in `backend/`:
Get a free key at [openrouter.ai](https://openrouter.ai)

Run the server:
```bash
uvicorn main:app --reload --port 8000
```

### Test the API

```bash
curl -X POST http://localhost:8000/process \
  -F "files=@sample_data/report_2022.txt" \
  -F "files=@sample_data/report_2023.txt" \
  -F "files=@sample_data/report_2024.txt" \
  -F "files=@sample_data/report_2025.txt"
```

## Sample Data

`backend/sample_data/` contains 4 synthetic patient reports (2022–2025) demonstrating a hypertension → prediabetes → diabetes → kidney function decline progression, used to showcase timeline generation and risk detection.
