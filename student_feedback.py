"""
student_feedback.py — AI Student Feedback Generator
======================================================
Generates a structured, child-friendly feedback report for iSchool students
based on a session transcript.

INPUT:
  --transcript   Path to the session transcript .txt file
  --session_id   Student / session identifier (e.g. T-1004)
  --output       Output HTML file path (default: StudentFeedback_<session_id>.html)

OUTPUT:
  HTML report with iSchool branding containing:
  - 5-star ratings: Observation, Creativity, Problem Solving, Communication, Homework
  - Constructive feedback (≤3 lines) in session language (Arabic / English)
  - Session summary: what was learned + key reminders
  - Selected template tags from the iSchool Feedback Templates list

USAGE:
  python student_feedback.py --transcript "Sessions/T-1004/T-1004_Jan_12_2026.txt" --session_id T-1004

NOTES:
  - Does NOT require a video file — transcript only
  - Detects session language automatically (Arabic / English)
  - Instructs the model to clean up Zoom auto-transcription artifacts
    (especially for Arabic: mis-joined words, broken diacritics, repeated phrases)
  - Uses Gemini 2.5 Flash via google.genai SDK
"""

import os
import sys
import io
import json
import re
import argparse
from datetime import datetime

from google import genai
from google.genai import types

# Fix Windows console encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Configuration ─────────────────────────────────────────────────────────
API_KEY = "AIzaSyDE3cwtYlOkRuUIdO-TXeSeYqyFjIoHdkY"
os.environ["GEMINI_API_KEY"] = API_KEY

BASE_DIR = r"c:\Users\omarhemied26\Downloads\TestVideo_Archive\TestVideo 22122025"
MODEL_NAME = "gemini-2.5-flash-preview-04-17"
MODEL_TEMPERATURE = 1.0

client = genai.Client(api_key=API_KEY)

# ── iSchool logo (base64 PNG, embedded for offline HTML) ──────────────────
LOGO_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAaMAAAB4CAMAAABoxW2eAAABPlBMVEX///8Fb+z/1wD/fxz/1AAAbew"
    "AZusAZOsAa+wAYusAaesAZesAYOsAauz/2AD/2gDv9P3a5fvC1PnQ3vrH2Pmyyff4+/7m7vyXt/Xf6fzz"
    "9/6Rs/RZkfBnmfF4o/K5zvipw/b/3UlBhe5Miu+gvfV1ofL/7Kf/8sX/31T/cwB/qPM3gO6IrfNhlvD/"
    "8b7/5oX/4m3/9tj/+eT//vn/++4ddu0AZfcmee3/66H/2if/5Hr/6ZT/egD/dx3/7rFnjcL/4WT/9dD/"
    "7eT/hzH/vg3/rxL/3k//2zn/8er/uJH/j0T/waH/qHb/37X/ykb/jRjBtXC9tXtah8ubpJ2qq4/qyyw3"
    "e9z/4dL/gxv/pxT/1b//ywj/xGT/7tWuv9fPvmHaxFTkyUSGmq99lrSOnqd1kr2VqsdLgtKwrojoyzzO"
    "3qaiAAAU2klEQVR4nO1daXvjyHEWNCDOpsBbPEWKIkcakrpGos4ZzWjHu7Id2/Ge8cZJ1rtex8n//wMB"
    "QBDsrqrGwQElPRHfD/vsUGig0dV1Vxc2NpLi7PTo6vjkq4drFw8HJ8ev998mHrvG6vH27uZaVdVNHu6/"
    "1YOr06ee2hoe9m82AXl4Sm3e7D/1BF869k/k9AnJdLyWek+G89vNOAIFZPrq/qnn+jJxf5CMQDMqPayp"
    "9OjYf0hBIZ9KH8+fes4vC6kp5FPp6qmn/YJwuiuhkBpARqSHNSs9Ds5OCBq4dNk9uX19d7S/f3R3dfwR"
    "uUvBZUdPPfsXgddo7T1f9R5xyOnVAUEm9fgp5vyycA4VkXp9K48lHH3EBD14xNm+SFypkEAx7unZLWKm"
    "h8eZ6gvFmWgrqCeJfJ5bSKTrVc/zBeOeX2x18/Ys4ThkZKyJtCrw/KBupvJ17tfi7lHAR37U27SjgfGw"
    "NhxWgPNrjkI3SaUcB1ErrU3w7HG6WGH1YblMw5FIpLUzmzHeciR6vexN7kUircNC2eJgwUSfsbQikdZ2"
    "Q7ZY3lYQsC8Y7593rzUAvgrW9XMrSIRYn7rOoGcJX0qpD0uYcwA3PJF2M5jZGiH2N9Xk5vIXX/xW+reH"
    "tW23OiQUTL/93ac3Hn7/LzTTnQt2Q4bzWyMp/vDm3asZ3r35HXkFr5KWt+PXWBZ/fDenkE+lT6TIe3hS"
    "RmpUKo1Hf+jzwds/vXkl4h1FpNOnY6TCUDdt22TtyqM+9vng6q/vX0F8oi78yPGR4/57u1IdtCaj0aTe"
    "6xabK5xhiRmKD8PcW+Fjni3OH5x/RSR69Y7SSWFgyXG2/twaM9PWLMsyDMvSNJ1Z7Xp5NVOsMCWEdbma"
    "ZzxnvFa3foPZ6NWrN5S08yNLW47z9TcXej6nABiWbndK2U+xpvFPsYbZP2EFKPbq9WoR/94sDeq9dHv5"
    "QN2k2EiCSJ5H7Gx+a5gGpE9IJ9uabC/3VlLUBRopLOv7rwBVxXRFjGbmBuLvxTYzmQurlfhWpx5bqBQb"
    "uaAGXG8532ka4iABFutku4rw/oP4IU+MjjmfrDYtcL+PmK4ftuqjS6YbBI9R8D0ehxR1EmF3+/0PZjSF"
    "fGZi9SzeNMA2A3c/zPDmK8HI5maml8LQ37+nxhGj2bJTJRZwUlzpcSGv0bMaTCpFJOgDnK4l1nKImiTs"
    "mNs3r3SlAUNtVCf050beHhTfQEMvssy6HLaPTu3zGYOtjSYjDuNq8rosdp9Nwtuz apSERedqNyHuVRi0"
    "Sb6rFUaMUI3OFo9+AS+SDQacV2FIGJGnMqxJnMHSE1Z4lYJPBiWlqsD7EF3XiMCB7g0HT1jQEDsqN4M"
    "9JBfuYNjjfhBmqgHD+vQrjsZzgvKTa3AvD0uTk3FGbDCmWi9Xjqv59JNxfX5MoxQfyDCoJi2msMvZcq"
    "pIjh0pMXfpxBQW2CX785YGFbVK1n+4YJCBqRLtJaH6V9j7VCKpNn9BqsE2OuDQrAISTsIRcKiJDjVbz"
    "DJYS70RaCp5IB8mpXcEm2u4UzKKEOXbcqRBB8Kk9gHSQWI6GHRXSI9uy0CaTMCIPbBvuUJqrU7H4EIbX"
    "T2j1oO5VGVzarFObMaIboUNl7kGaWuANjCy0PVJGEbI1hm8RkABONkQbbXFkIUOkS0eZbKqoRThLRc+W"
    "KxqmCnMR5FbBbDvlTv6o+rNnv52KoQV3WuEfaKvhJ2RkUZRStYIkUUPMJNf8QJEsaihSMgaGp5YPi0hS"
    "cU+SkJU2WshXq2QjVBqAqiJRdJA8RjZp7GWY9omUY5e6gLMYXVBrgS9Q5IqyIWzBvO7c+oxY8uMqCeOc"
    "A5JC3NsAtqvhBBmZHHJHY0rl8yRhG6kJT3mxdz/lz36TqS7E/t8dFDYPcXoCSjxXpMEsELYK0JBvqrLm"
    "C7CnLtlTcqXmhiRs4dMmMXQr1XBEpSheLjbN3lgJDTOJZSRQAbPHxnZLsqM7BDRODSXGpnkVj7bz8WYV"
    "X7VBYFpJa5vCaT9kxjZ64gJJAbIFCGKmwjl5I5CiiViPJJYYsK83t9gCRjPYKKrdwfCiNr5m6v3+1pSD"
    "3KcN9kJRQWD9lqHbWV8tKtWGiHxMNfVULy9RcXfBfJ4Lk0NjvmlaaDaLQAv6wj4Ap3jZJHCF7IHQJbm8"
    "dQKnhRIlBMeHuQ=="
)

# ── iSchool Feedback Templates (from Feedback Templates.pdf) ──────────────
FEEDBACK_TEMPLATES = """
FEEDBACK TEMPLATE CATEGORIES (use these as reference for tone and wording):

HIGH PERFORMANCE:
- All-Round Superstar – General Excellence
- Excellent problem-solving abilities
- Discussed in a brilliant way
- Seeking additional challenges (Growth Mindset)
- Consistent Progress & Great Final Effort
- Always Ready – A Joy to Teach!
- Sharp Mind – Needs More Structure

HOMEWORK:
- Excellent study appears in the homework
- Creative homework
- Excellent code organization in the homework
- Excellent homework details
- Excellent understanding of the homework
- Excellent homework effort
- Brave in seeking help with homework challenges

PROJECT:
- Excellent project implementation
- Creative project solution
- Well-structured project code
- Successful concept application
- Excellent debugging skills in project
- Project demonstrates attention to detail
- Good project effort
- Project shows improvement

ENGAGEMENT:
- Shows enthusiasm during exercise
- Positive attitude
- Solid Understanding – Let's Boost Confidence
- Shy but Smart

GROWTH/MIXED:
- Engaged Learner – Needs More Homework Practice
- Good Understanding – Needs to Speak Up More
- Big Effort – Needs More Preparation Time
- Great Problem-Solving Skills – Needs More Patience
- Creative Thinker – Let's Build Communication Too
- Great Start – Time to Refocus & Finish Strong
- Took Time to Improve – Great Comeback!

IMPROVEMENT:
- More engagement is required
- Needs to focus
- Need to study more
- Need to improve code quality
- Need to enhance coding engagement
- Improving Responsiveness
- Improving Assignment Consistency
- Needs to improve Code Clarity
- Need to test the code for homework assignment
- Need to focus in doing the homework
- Quiet Sessions – Needs Review and Practice
- Needs Stronger Focus – You've Got What It Takes
- Missed Homework – Strong in Class
- Needs More Review – You're Almost There

ARABIC POSITIVE:
- حريص على التعلم (Keen on learning)
- سلوكه إيجابي (Positive attitude)
- متحمس ونشيط (Enthusiastic and active)
- مهتم بتعلم المزيد (Interested in learning more)
- مجتهد ومثابر (Diligent and persistent)
- دائماً مستعد (Always ready)
- متعاون (Cooperative)
- يدير وقته بفاعلية (Manages time effectively)

ARABIC NEUTRAL:
- يطلب المساعدة (Asks for help)
- يقبل أخطاءه ويسعى لإصلاحها (Accepts mistakes/seeks fix)

ARABIC NEEDS IMPROVEMENT:
- يحتاج مزيداً من الانتباه (Needs more attention)
- انتبه للتفاصيل (Pay attention to details)
- يحتاج للمراجعة (Needs review)
- يجب الاهتمام بالجزء النظري (Focus on theory)
- مزيد من الاعتماد على النفس (More self-reliance)
- الاهتمام بإدارة الوقت (Focus on time management)
- مزيداً من التفاعل والمشاركة (More interaction)
- اقبل أخطاءك (Accept your mistakes)
- كن مستعداً (Be prepared)
- يرجى مراجعة الدرس وإتمام المهام (Review and complete tasks)

SUPPORT/ADMIN:
- Feel Free to Ask / لا تتردد في طلب المساعدة
- Advice to solve problem
"""

# ── JSON schema for Gemini response ───────────────────────────────────────
FEEDBACK_SCHEMA = {
    "type": "object",
    "required": [
        "session_language",
        "student_name_or_id",
        "ratings",
        "overall_feedback",
        "session_summary",
        "key_reminders",
        "template_tags",
    ],
    "properties": {
        "session_language": {"type": "string"},
        "student_name_or_id": {"type": "string"},
        "ratings": {
            "type": "object",
            "required": ["observation", "creativity", "problem_solving", "communication", "homework"],
            "properties": {
                "observation":      {"type": "integer", "minimum": 1, "maximum": 5},
                "creativity":       {"type": "integer", "minimum": 1, "maximum": 5},
                "problem_solving":  {"type": "integer", "minimum": 1, "maximum": 5},
                "communication":    {"type": "integer", "minimum": 1, "maximum": 5},
                "homework":         {"type": "integer", "minimum": 1, "maximum": 5},
            },
        },
        "rating_justifications": {
            "type": "object",
            "properties": {
                "observation":      {"type": "string"},
                "creativity":       {"type": "string"},
                "problem_solving":  {"type": "string"},
                "communication":    {"type": "string"},
                "homework":         {"type": "string"},
            },
        },
        "overall_feedback": {
            "type": "string",
            "description": "Max 3 lines. Constructive, warm, encourages the student. Written in the session language.",
        },
        "session_summary": {
            "type": "string",
            "description": "Brief summary of what was covered/learned in the session. In session language.",
        },
        "key_reminders": {
            "type": "array",
            "items": {"type": "string"},
            "description": "2-4 key points or action items for the student to remember. In session language.",
        },
        "template_tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "1-3 template labels from the feedback template list that best match this student.",
        },
    },
}


# ── Helpers ────────────────────────────────────────────────────────────────
def load_transcript(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def build_prompt(transcript: str, session_id: str) -> str:
    ischool_description = """
iSchool is an online coding and technology academy for school-age students (ages 8-18).
Students attend live one-on-one or small-group sessions via Zoom with a tutor.
Sessions cover programming concepts in Python, Scratch, web development, and related tech skills.
The environment is supportive, student-centered, and encourages learning through making and problem-solving.
"""

    return f"""
You are an expert AI feedback assistant for iSchool, an online coding academy for kids and teenagers.

## YOUR TASK
Analyze the session transcript below and generate a structured student feedback report in JSON format.

## iSchool CONTEXT
{ischool_description}

## FEEDBACK TEMPLATE REFERENCE
Use these templates for tone, wording, and tag selection:
{FEEDBACK_TEMPLATES}

## TRANSCRIPT CLEANING INSTRUCTIONS (CRITICAL)
The transcript was auto-generated by Zoom, which often introduces errors, especially in Arabic:
- Fix broken/mis-joined Arabic words (e.g. "ال طالب" → "الطالب")
- Fix mis-transcribed numbers and technical terms
- Remove repeated filler phrases from Zoom noise
- Fix broken diacritics and punctuation in Arabic
- For English: fix obvious autocorrect errors and sentence breaks
- DO NOT change the meaning — only fix transcription artifacts

## SESSION ID
{session_id}

## TRANSCRIPT
{transcript}

## OUTPUT REQUIREMENTS
Return a single JSON object matching the schema exactly. Follow these rules carefully:

### LANGUAGE
- Detect the primary session language from the transcript (Arabic or English)
- Write overall_feedback, session_summary, and key_reminders IN THE SAME LANGUAGE as the session
- Use Arabic script for Arabic sessions, English for English sessions

### RATINGS (1–5 stars)
Rate each category based only on evidence from the transcript:
- **Observation**: Did the student pay attention, notice details, follow what the tutor was explaining?
- **Creativity**: Did the student show creative thinking, suggest ideas, approach problems originally?
- **Problem Solving**: Did the student attempt to debug, figure things out, reason through challenges?
- **Communication**: Was the student clear, engaged in dialogue, asked/answered questions?
- **Homework**: Was there evidence of completed/attempted homework? Quality of that work?

If there is no evidence in the transcript for a category (e.g. homework was not discussed), give a neutral 3.

### OVERALL FEEDBACK
- Maximum 3 lines total
- Warm, encouraging, constructive — written for a child/teen
- Specifically address what the student did well AND one area to work on
- Must be in the session language

### SESSION SUMMARY
- 2-4 sentences describing what was covered/learned
- In the session language

### KEY REMINDERS
- 2-4 short bullet points the student should remember or action
- In the session language

### TEMPLATE TAGS
- Pick 1-3 labels from the FEEDBACK TEMPLATE REFERENCE that best match this student
- Use the exact label text from the reference list

Return ONLY the JSON — no markdown fencing, no extra text.
"""


def call_gemini(prompt: str) -> dict:
    """Call Gemini and return parsed feedback JSON."""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=MODEL_TEMPERATURE,
            response_mime_type="application/json",
            response_schema=FEEDBACK_SCHEMA,
            max_output_tokens=4096,
        ),
    )
    text = response.text or ""
    # Strip markdown fences if any
    text = re.sub(r"^```[a-z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text.strip())
    return json.loads(text)


# ── HTML Report Generator ─────────────────────────────────────────────────
def stars_html(rating: int, max_stars: int = 5) -> str:
    filled = "★" * rating
    empty  = "☆" * (max_stars - rating)
    return (
        f"<span style='color:#f5a623;font-size:1.4rem;letter-spacing:2px;'>{filled}</span>"
        f"<span style='color:#d1d5db;font-size:1.4rem;letter-spacing:2px;'>{empty}</span>"
        f"<span style='font-size:.78rem;font-weight:700;color:#4b5563;margin-left:.4rem;'>{rating}/5</span>"
    )


def color_for_rating(r: int) -> str:
    return {1: "#dc2626", 2: "#f97316", 3: "#f59e0b", 4: "#16a34a", 5: "#15803d"}.get(r, "#6b7280")


def generate_html(data: dict, session_id: str, transcript_path: str) -> str:
    now     = datetime.now().strftime("%d %b %Y, %H:%M")
    lang    = data.get("session_language", "English")
    student = data.get("student_name_or_id", session_id)
    ratings = data.get("ratings", {})
    justif  = data.get("rating_justifications", {})
    feedback = data.get("overall_feedback", "")
    summary  = data.get("session_summary", "")
    reminders = data.get("key_reminders", [])
    tags    = data.get("template_tags", [])

    is_arabic = "arabic" in lang.lower() or "ar" == lang.lower()
    dir_attr  = 'dir="rtl"' if is_arabic else 'dir="ltr"'
    text_align = "right" if is_arabic else "left"

    # Rating categories config
    rating_cats = [
        ("observation",    "👁️ Observation",    "الملاحظة"   if is_arabic else "Observation"),
        ("creativity",     "💡 Creativity",     "الإبداع"    if is_arabic else "Creativity"),
        ("problem_solving","🔧 Problem Solving","حل المشكلات" if is_arabic else "Problem Solving"),
        ("communication",  "💬 Communication",  "التواصل"    if is_arabic else "Communication"),
        ("homework",       "📚 Homework",       "الواجب"     if is_arabic else "Homework"),
    ]

    # Build rating rows
    rating_rows = ""
    for key, en_label, local_label in rating_cats:
        r   = ratings.get(key, 3)
        col = color_for_rating(r)
        j   = justif.get(key, "")
        rating_rows += f"""
        <div style='display:flex;align-items:flex-start;gap:1rem;padding:.85rem 1rem;
                    border-bottom:1px solid #f1f5f9;flex-wrap:wrap;'>
          <div style='min-width:160px;'>
            <div style='font-size:.82rem;font-weight:700;color:#1e293b;margin-bottom:.3rem;
                        text-align:{text_align};'>{local_label}</div>
            <div>{stars_html(r)}</div>
          </div>
          <div style='flex:1;font-size:.76rem;color:#64748b;line-height:1.55;
                      border-left:3px solid {col}20;padding-left:.75rem;
                      text-align:{text_align};'>{j}</div>
        </div>"""

    # Key reminders list
    reminder_items = "".join(
        f"<li style='margin-bottom:.35rem;line-height:1.55;'>{r}</li>"
        for r in reminders
    )

    # Template tags
    tag_badges = "".join(
        f"<span style='display:inline-block;background:#eff6ff;color:#1d4ed8;"
        f"border:1px solid #bfdbfe;border-radius:20px;padding:.2rem .7rem;"
        f"font-size:.72rem;font-weight:600;margin:.2rem;'>{t}</span>"
        for t in tags
    )

    # Overall average
    all_ratings = [ratings.get(k, 3) for k, _, __ in rating_cats]
    avg = round(sum(all_ratings) / len(all_ratings), 1) if all_ratings else 3
    avg_col = color_for_rating(round(avg))
    avg_label_en  = {1:"Needs Work",2:"Developing",3:"Good",4:"Great",5:"Outstanding!"}
    avg_label_ar  = {1:"يحتاج جهد",2:"في تطور",3:"جيد",4:"رائع",5:"متميز!"}
    avg_labels = avg_label_ar if is_arabic else avg_label_en
    avg_label = avg_labels.get(round(avg), "")

    html = f"""<!DOCTYPE html>
<html lang="{'ar' if is_arabic else 'en'}" {dir_attr}>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Student Feedback — {student}</title>
<style>
  *, *::before, *::after {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{
    font-family: {'Segoe UI, Tahoma, Arial, sans-serif' if is_arabic else "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"};
    background: #f1f5f9;
    color: #0f172a;
    padding: 1.5rem;
    direction: {'rtl' if is_arabic else 'ltr'};
  }}
  @media print {{
    body {{ background:#fff; padding:.5cm; }}
    .no-print {{ display:none; }}
    @page {{ margin: 1.5cm; size: A4; }}
  }}
</style>
</head>
<body>
<div style="max-width:820px;margin:0 auto;">

  <!-- Header / Branding -->
  <div style="background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 60%,#1e40af 100%);
              border-radius:12px;padding:1.5rem 2rem;margin-bottom:1.25rem;
              display:flex;align-items:center;justify-content:space-between;
              flex-wrap:wrap;gap:1rem;">
    <div style="display:flex;align-items:center;gap:1rem;">
      <img src="data:image/png;base64,{LOGO_B64}"
           alt="iSchool Logo"
           style="height:52px;width:auto;border-radius:6px;background:#fff;padding:4px;">
      <div>
        <div style="font-size:.65rem;text-transform:uppercase;letter-spacing:.12em;
                    color:#93c5fd;margin-bottom:.2rem;">Student Feedback Report</div>
        <div style="font-size:1.25rem;font-weight:800;color:#f1f5f9;">
          {'تقرير متابعة الطالب' if is_arabic else 'Session Feedback'}
        </div>
      </div>
    </div>
    <div style="text-align:{'left' if is_arabic else 'right'};">
      <div style="font-size:1.8rem;font-weight:900;color:#22d3ee;line-height:1;">{avg}</div>
      <div style="font-size:.7rem;color:#94a3b8;">{avg_label}</div>
      <div style="font-size:.65rem;color:#64748b;margin-top:.2rem;">{now}</div>
    </div>
  </div>

  <!-- Student Info Bar -->
  <div style="background:#fff;border:1px solid #e2e8f0;border-radius:8px;
              padding:.85rem 1.25rem;margin-bottom:1.25rem;
              display:flex;flex-wrap:wrap;gap:1.5rem;align-items:center;">
    <div>
      <div style="font-size:.65rem;text-transform:uppercase;letter-spacing:.08em;color:#94a3b8;">
        {'الطالب / المجموعة' if is_arabic else 'Student / Session'}
      </div>
      <div style="font-size:.95rem;font-weight:700;color:#1e293b;">{student}</div>
    </div>
    <div>
      <div style="font-size:.65rem;text-transform:uppercase;letter-spacing:.08em;color:#94a3b8;">
        {'لغة الجلسة' if is_arabic else 'Session Language'}
      </div>
      <div style="font-size:.9rem;font-weight:600;color:#1e293b;">{lang}</div>
    </div>
    <div>
      <div style="font-size:.65rem;text-transform:uppercase;letter-spacing:.08em;color:#94a3b8;">
        {'تاريخ التقرير' if is_arabic else 'Report Date'}
      </div>
      <div style="font-size:.9rem;font-weight:600;color:#1e293b;">{now}</div>
    </div>
    {'<div>' + tag_badges + '</div>' if tags else ''}
  </div>

  <!-- Star Ratings -->
  <div style="background:#fff;border:1px solid #e2e8f0;border-radius:8px;
              padding:1.25rem;margin-bottom:1.25rem;overflow:hidden;">
    <div style="font-size:.8rem;font-weight:700;text-transform:uppercase;
                letter-spacing:.08em;color:#1e293b;margin-bottom:.75rem;
                text-align:{text_align};">
      {'التقييم' if is_arabic else '⭐ Performance Ratings'}
    </div>
    {rating_rows}
  </div>

  <!-- Overall Feedback -->
  <div style="background:#fff;border:1px solid #e2e8f0;border-top:4px solid #1d4ed8;
              border-radius:8px;padding:1.25rem;margin-bottom:1.25rem;">
    <div style="font-size:.8rem;font-weight:700;text-transform:uppercase;
                letter-spacing:.08em;color:#1d4ed8;margin-bottom:.75rem;
                text-align:{text_align};">
      {'💬 التقييم العام' if is_arabic else '💬 Overall Feedback'}
    </div>
    <div style="font-size:.9rem;line-height:1.75;color:#1e293b;
                text-align:{text_align};background:#eff6ff;
                border-radius:6px;padding:1rem 1.25rem;
                border-left:{'none' if is_arabic else '4px solid #3b82f6'};
                border-right:{'4px solid #3b82f6' if is_arabic else 'none'};">
      {feedback.replace(chr(10), '<br>')}
    </div>
  </div>

  <!-- Session Summary -->
  <div style="background:#fff;border:1px solid #e2e8f0;border-top:4px solid #16a34a;
              border-radius:8px;padding:1.25rem;margin-bottom:1.25rem;">
    <div style="font-size:.8rem;font-weight:700;text-transform:uppercase;
                letter-spacing:.08em;color:#16a34a;margin-bottom:.75rem;
                text-align:{text_align};">
      {'📖 ماذا تعلمنا اليوم؟' if is_arabic else '📖 What We Learned Today'}
    </div>
    <div style="font-size:.88rem;line-height:1.7;color:#374151;text-align:{text_align};">
      {summary.replace(chr(10), '<br>')}
    </div>
  </div>

  <!-- Key Reminders -->
  <div style="background:#fff;border:1px solid #e2e8f0;border-top:4px solid #f59e0b;
              border-radius:8px;padding:1.25rem;margin-bottom:1.25rem;">
    <div style="font-size:.8rem;font-weight:700;text-transform:uppercase;
                letter-spacing:.08em;color:#f59e0b;margin-bottom:.75rem;
                text-align:{text_align};">
      {'🔔 تذكّر دائماً' if is_arabic else '🔔 Key Reminders'}
    </div>
    <ul style="padding-{'right' if is_arabic else 'left'}:1.25rem;
               font-size:.88rem;color:#374151;text-align:{text_align};">
      {reminder_items}
    </ul>
  </div>

  <!-- Footer -->
  <div style="margin-top:1.5rem;padding-top:1rem;border-top:1px solid #e2e8f0;
              font-size:.68rem;color:#94a3b8;text-align:center;">
    <img src="data:image/png;base64,{LOGO_B64}" alt="iSchool"
         style="height:22px;width:auto;vertical-align:middle;margin-{'left' if is_arabic else 'right'}:.5rem;opacity:.6;">
    iSchool &nbsp;|&nbsp; Student Feedback Report &nbsp;|&nbsp; {now}
  </div>

</div>
</body>
</html>"""
    return html


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="AI Student Feedback Generator — iSchool")
    parser.add_argument("--transcript",  required=True, help="Path to session transcript .txt file")
    parser.add_argument("--session_id",  default="STU-001", help="Student or session ID")
    parser.add_argument("--output",      default=None, help="Output HTML path (default: StudentFeedback_<id>.html)")
    args = parser.parse_args()

    out_path = args.output or os.path.join(
        os.path.dirname(args.transcript),
        f"StudentFeedback_{args.session_id}.html"
    )

    print(f"\n{'='*55}")
    print(f"  iSchool Student Feedback Generator")
    print(f"  Session:    {args.session_id}")
    print(f"  Transcript: {args.transcript}")
    print(f"  Output:     {out_path}")
    print(f"{'='*55}\n")

    # Load transcript
    print("[1/3] Loading transcript...")
    transcript = load_transcript(args.transcript)
    print(f"      {len(transcript):,} characters loaded")

    # Build prompt & call Gemini
    print("[2/3] Calling Gemini (analyzing session)...")
    prompt = build_prompt(transcript, args.session_id)
    data   = call_gemini(prompt)
    lang   = data.get("session_language", "?")
    ratings = data.get("ratings", {})
    print(f"      Language detected: {lang}")
    print(f"      Ratings — Obs:{ratings.get('observation','?')} "
          f"Cre:{ratings.get('creativity','?')} "
          f"PS:{ratings.get('problem_solving','?')} "
          f"Com:{ratings.get('communication','?')} "
          f"HW:{ratings.get('homework','?')}")

    # Generate HTML
    print("[3/3] Generating HTML report...")
    html = generate_html(data, args.session_id, args.transcript)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    # Also save raw JSON alongside for inspection
    json_path = out_path.replace(".html", "_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] HTML report: {out_path} ({len(html)//1024} KB)")
    print(f"[OK] JSON data:   {json_path}")
    print(f"\n{'='*55}\n")


if __name__ == "__main__":
    main()
