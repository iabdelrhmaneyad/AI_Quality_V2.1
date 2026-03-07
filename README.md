# AI Quality Model — V2.1

AI-powered session quality review engine using Retrieval-Augmented Generation (RAG).  
Analyses recorded teaching sessions, detects quality issues, and produces structured JSON reports with scoring.

---

## What It Does

- Processes session recordings (MP4) + transcripts (TXT)
- Detects issues across 5 categories: **Teaching (T)**, **Curriculum (C)**, **Preparation (P)**, **Attitude (A)**, **Setup (S)**
- Classifies each finding as a regular comment or flag (Yellow / Red)
- Produces a weighted quality score and areas-for-improvement list
- Cross-validates findings against a human-audited baseline (audit_calibration.json)

---

## Repository Structure

```
AI_Quality_V2.1/
├── rag_video_analysis.py          # Core RAG analysis engine
├── batch_runner.py                # Batch processing script — add your sessions here
├── vector_db_simple/
│   ├── rules.json                 # Quality assessment rules (loaded on every run)
│   └── audit_calibration.json    # Calibration data from 524 audited sessions
├── comments_bank.json             # 1,929 real reviewer comment examples
├── quality_guide.json             # Quality review guide & scoring logic
├── flag_examples.json             # Flag comment templates (Yellow / Red)
└── quality_comments.json          # Quality comments reference bank
```

---

## Requirements

```
google-generativeai
python-dotenv
```

Install:
```bash
pip install google-generativeai python-dotenv
```

Set your Gemini API key:
```bash
# Windows
set GEMINI_API_KEY=your_api_key_here

# Linux / macOS
export GEMINI_API_KEY=your_api_key_here
```

---

## Single Session Usage

```bash
python rag_video_analysis.py \
  --input       "Sessions/T-1004/T-1004_Jan_12_2026_Slot 5.mp4" \
  --output_report "Sessions/T-1004/Quality_Report_T-1004.txt" \
  --transcript  "Sessions/T-1004/T-1004_Jan_12_2026_Slot 5.txt"
```

**Output files** (auto-generated alongside the `.txt` report):
- `Quality_Report_<TID>.json` — full structured data for dashboards
- `Quality_Report_<TID>.txt`  — human-readable summary

---

## Batch Processing

Edit `batch_runner.py` to list your sessions, then run:

```bash
python batch_runner.py
```

Each entry in the `runs` list:
```python
("T-1004",                                          # Session ID
 "Sessions/T-1004/T-1004_Jan_12_2026_Slot 5.mp4",  # MP4 path
 "Sessions/T-1004/T-1004_Jan_12_2026_Slot 5.txt",  # Transcript path
 "Sessions/T-1004/Quality_Report_T-1004.txt"),      # Output path
```

---

## JSON Output Format (per session)

```json
{
  "session_id": "T-1004",
  "scoring": {
    "final_weighted_score": 96.8,
    "deductions": { ... }
  },
  "areas_for_improvement": [
    {
      "category": "Teaching",
      "subcategory": "Student Engagement",
      "text": "...",
      "timestamp": "00:12:34",
      "is_advice_only": false
    }
  ],
  "flags": [ ... ],
  "summary": "..."
}
```

---

## Configuration Files

| File | Purpose | Loaded At |
|---|---|---|
| `vector_db_simple/rules.json` | Scoring rules and category weights | Every run |
| `vector_db_simple/audit_calibration.json` | Accuracy calibration from 524 audited sessions | Every run |
| `comments_bank.json` | RAG retrieval — real reviewer wording examples | Every run |
| `quality_guide.json` | Grading guide injected into model prompt | Every run |
| `flag_examples.json` | Flag comment templates | Every run |
| `quality_comments.json` | Extended comments reference | Every run |

> ⚠️ All JSON files in `vector_db_simple/` and the root must be present before running — the model will fail to load otherwise.

---

## Accuracy Baseline

Cross-validated against 524 human-audited sessions:

| Metric | Value |
|---|---|
| Valid Detection Rate (MATCH + ACCURATE) | **91.3%** |
| MATCH (exact/close finding) | 71.7% |
| ACCURATE (correct category, related angle) | 19.6% |
| INACCURATE (not in baseline) | 8.7% |

---

## Version

**V2.1** — March 2026  
Model: Gemini 2.5 Flash (RAG)  
Calibration: 524 sessions, 1,894 findings checked
