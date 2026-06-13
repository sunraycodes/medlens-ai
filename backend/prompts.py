EXTRACTION_PROMPT = """You are a medical data extraction system. Extract structured information from the report below.

Return ONLY valid JSON, no markdown, no explanation, in this exact format:
{
  "date": "YYYY-MM",
  "age": <number or null>,
  "diagnoses": ["list of diagnosed conditions"],
  "lab_values": [{"test": "name", "value": "value with unit", "flag": "normal/elevated/low/diabetic_range/etc"}],
  "medications": ["list of medications with dosage"],
  "allergies": ["list of allergies, empty if none mentioned"],
  "notes": "any clinically relevant note, especially about follow-ups or missing evaluations"
}

REPORT TEXT:
<<report_text>>
"""

ANALYSIS_PROMPT = """You are a clinical decision support AI helping a doctor quickly understand a patient's history across multiple reports.

Below is structured data extracted from <<n>> medical reports for one patient, in chronological order.

DATA:
<<json_data>>

Based on this data, return ONLY valid JSON in this exact format:
{
  "patient_summary": {
    "age": <latest age>,
    "conditions": ["list of all current/ongoing conditions"],
    "current_medications": ["list of current medications"],
    "allergies": ["list of all allergies"]
  },
  "timeline": [
    {"date": "YYYY-MM", "event": "short description of key event/change"}
  ],
  "risk_flags": [
    {"severity": "high/medium/low", "flag": "short title", "explanation": "1-2 sentence clinical reasoning, citing which reports/dates support this"}
  ]
}

Look specifically for:
- Trends across reports (e.g. gradually worsening lab values)
- Missed follow-ups (a concerning value in one report with no evaluation/action in later reports)
- Potential medication interactions or concerns given the conditions
- Any contradictions between reports
"""

DOCTOR_SUMMARY_PROMPT = """Based on the following patient analysis JSON, write a concise one-page doctor handoff summary in plain text (not JSON). 
It should be readable in under 30 seconds. Use clear section headers: PATIENT OVERVIEW, KEY TIMELINE, ACTIVE RISK FLAGS, CURRENT MEDICATIONS, ALLERGIES.
Be direct and clinical. No fluff.

DATA:
<<json_data>>
"""

RAG_QA_PROMPT = """You are a clinical assistant. Use the retrieved report excerpts below to answer the doctor's question. Be concise and cite which file supports your answer.

RETRIEVED EXCERPTS:
<<context>>

DOCTOR'S QUESTION: <<question>>

Answer in 2-4 sentences, clinical tone, citing source filenames where relevant."""