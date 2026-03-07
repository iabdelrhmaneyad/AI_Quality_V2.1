"""
RAG-BASED VIDEO ANALYSIS TOOL — V38
=============================
V38 Changes (calibrated from full V37 audit of 71 findings):
- REMOVED Zoom annotation tools as deduction (100% FP rate, 6/6 inaccurate in V37)
- Student Engagement: 5-min threshold; exempt checkup/concept explanation phases
- Project Implementation: don't flag timestamps <00:25:00 as step-by-step (likely HW review)
- Inappropriate language by student: only flag if tutor HEARD IT and didn't address it
- REMOVED ice-breaker/first-session rule (was causing false positives)
- Frame naming: Frame-N-HH-MM-SS format (e.g. Frame-5-00-21-32.jpg)
- Late joining detection: strengthened — start_time <00:02:00 = flag without extra evidence
=============================

This script uses a "Chat-based Retrieval" approach to strictly enforce Quality Guidelines.
Instead of a single prompt, it establishes a "Knowledge Base" from the provided PDFs and then
analyzes the session by cross-referencing every observation against these rules.

WORKFLOW:
1. Extract Resources (Audio/Frames) - Optimized
2. Upload Resources (Parallel)
3. Initialize Chat Session
4. Step 1: Ingest Rules (Send PDFs)
5. Step 2: Ingest Session Data (Send Frames + Transcript)
6. Step 3: Perform "Chain-of-Thought" Analysis
7. Generate Report
"""

import os
from google import genai
from google.genai import types
import json
import argparse
import time
import subprocess
import shutil
import glob
import random
import re
import concurrent.futures
import sys
import io

# Fix Windows console encoding to support Unicode characters
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ============================================================================
# CONFIGURATION
# ============================================================================
# Force hardcoded key to resolve persistent env var issues
API_KEY = "AIzaSyDE3cwtYlOkRuUIdO-TXeSeYqyFjIoHdkY"
os.environ["GEMINI_API_KEY"] = API_KEY  # Ensure SDK sees it too if it checks env
print(f"DEBUG: Active API Key starts with {API_KEY[:5]}...")


BASE_DIR = r"c:\Users\omarhemied26\Downloads\TestVideo_Archive\TestVideo 22122025"

# File Paths
VIDEO_FILE_PATH = os.path.join(BASE_DIR, "Sessions/T-4053/T-4053_Jan_5_2026_Slot 5.mp4")
TRANSCRIPT_PATH = os.path.join(BASE_DIR, "Sessions/T-4053/T-4053_Jan_5_2026_Slot 5.txt")

# V38: JSON knowledge base (converted from PDFs for cleaner model comprehension)
# Run convert_pdfs_to_json.py once to regenerate these from the source PDFs.
REFERENCE_JSON_FILES = [
    os.path.join(BASE_DIR, "quality_guide.json"),        # Quality Guide for Reviewers
    os.path.join(BASE_DIR, "quality_comments.json"),     # Quality Comments V1062025
    os.path.join(BASE_DIR, "flag_examples.json"),        # Examples of Flag Comments
    os.path.join(BASE_DIR, "comments_bank.json"),        # Comments Bank (real examples)
]

# Legacy PDF paths kept for reference (no longer uploaded to Gemini)
PDF_REFERENCE_FILES = [
    os.path.join(BASE_DIR, "Quality Guide for Reviewers.pdf"),
    os.path.join(BASE_DIR, "Quality Comments V1062025.pdf"),
    os.path.join(BASE_DIR, "Examples of Flag comments.pdf"),
    os.path.join(BASE_DIR, "Comments Bank.pdf")
]

# Output
OUTPUT_REPORT_TXT = os.path.join(BASE_DIR, "Sessions/T-4053/Quality_Report_RAG_T-4053.txt")

# Video Processing
# Video Processing (Optimized for Quality)
DEFAULT_START_TIME = "00:15:00"
FRAME_EXTRACTION_INTERVAL = 120  # Extract 1 frame every 120 seconds (30 frames/hr — optimal token/quality tradeoff)
FRAME_WIDTH = 1024               # Reduced from 1280 to save tokens (still readable)
FRAME_QUALITY = 2                # -q:v 2 (Near lossless)
TARGET_FRAME_COUNT = 30          # 30 frames max — halves token cost vs 60

MODEL_NAME = "gemini-3-flash-preview"

# ============================================================================
# DETERMINISTIC CONFIGURATION (Gemini 3.0 Flash)
# ============================================================================
# Gemini 3.0 Flash Preview  (V19 — aligned with Gemini 3 docs)
# - Latest generation Flash model
# - optimized for speed and multimodal understanding
# - Per Gemini 3 docs: temperature MUST be 1.0
#   "Setting it below 1.0 may lead to unexpected behavior such as
#    looping or degraded performance."
# ============================================================================

MODEL_TEMPERATURE = 1.0   # Gemini 3 REQUIRES 1.0 — lower values cause looping/degraded output
# NOTE: With temp=1.0 the seed parameter still provides partial reproducibility.

# Gemini 3.0 Thinking Control (V19: thinking_level replaces thinking_budget):
# thinking_level controls internal reasoning depth. Options:
# - "minimal": Fastest, least accurate
# - "low": Light reasoning
# - "medium": Balanced
# - "high": Deep reasoning (DEFAULT for Gemini 3 Flash)
# IMPORTANT: Cannot use thinking_budget AND thinking_level in the same request.
DEFAULT_THINKING_LEVEL = "medium"  # Balanced reasoning depth

# Media resolution for multimodal inputs (audio + frames)
# V19: Frames re-enabled. MEDIUM gives 70 tokens/frame for video frames,
# 560 tokens/image for uploaded images — good balance of detail vs cost.
# Options: MEDIA_RESOLUTION_LOW, MEDIA_RESOLUTION_MEDIUM, MEDIA_RESOLUTION_HIGH
# V33: Switched to HIGH — docs recommend HIGH for image-based analysis (frames are uploaded as images)
# HIGH = 1120 tokens/image vs MEDIUM = 560 tokens/image → 2x detail for reading slide text/screen content
DEFAULT_MEDIA_RESOLUTION = "MEDIA_RESOLUTION_HIGH"

# Reproducibility controls
# The seed parameter provides PARTIAL reproducibility (not guaranteed deterministic)
DEFAULT_SEED = 42

# ============================================================================
# SINGLE-RUN MODE (COST-EFFECTIVE)
# ============================================================================
# With temperature=1.0 + seed, there is some variance between runs.
# The seed parameter provides PARTIAL reproducibility.
# For most consistent results, use a fixed seed.
# ============================================================================
DEFAULT_CONSISTENCY_RUNS = 1  # Single run (cost-effective)

# Output control (leave unset by default; can be overridden via CLI)
DEFAULT_MAX_OUTPUT_TOKENS = 20480  # Increased to reduce JSON truncation on first attempt

# Retry control for API failures / JSON parse errors
MAX_JSON_RETRIES = 3
MAX_EMPTY_JSON_RETRIES = 3  # Retries specifically for empty/truncated JSON responses
RETRY_TOKEN_BOOST = 4096  # Extra output tokens per retry attempt to prevent truncation

# Score variance threshold - if runs differ by more than this, flag as unreliable
SCORE_VARIANCE_THRESHOLD = 5.0  # Points

# Costs (Gemini 3.0 Flash - Paid Tier pricing, per 1M tokens)
# Input: $0.50/M (text/image/video), $1.00/M (audio)
# Output (including thinking tokens): $3.00/M
COST_PER_MILLION_INPUT_TOKENS = 0.50        # text / image / video
COST_PER_MILLION_AUDIO_INPUT_TOKENS = 1.00  # audio input is 2x
COST_PER_MILLION_OUTPUT_TOKENS = 3.00       # includes thinking tokens
AUDIO_TOKENS_PER_SECOND = 32                # Gemini audio tokenisation rate

# Batch API: 50% discount
BATCH_COST_PER_MILLION_INPUT_TOKENS = 0.25
BATCH_COST_PER_MILLION_AUDIO_INPUT_TOKENS = 0.50
BATCH_COST_PER_MILLION_OUTPUT_TOKENS = 1.50

# Media integrity checks
MEDIA_PROBE_TIMEOUT_SEC = 20
AUDIO_EXTRACTION_TIMEOUT_SEC = 180
MIN_VALID_MEDIA_DURATION_SEC = 60
MIN_VALID_AUDIO_DURATION_SEC = 30
MIN_VALID_AUDIO_FILE_BYTES = 64 * 1024

# Focused issue-matching targets (high-miss subcategories)
PRIORITY_SUBCATEGORY_TARGETS = [
    "Session study",
    "Knowledge About Subject",
    "Tools used and Methodology",
    "Class Management",
    "Student Engagement",
    "Session Synchronization",
]

# V33: JSON Schema for Steps 2 & 3 — enforces structure, guarantees integer ratings, eliminates retries
# response_schema ensures: all required keys present, rating always int 0-5, no markdown fencing
QUALITY_REPORT_SCHEMA = {
    "type": "object",
    "required": ["_reasoning_trace", "meta", "positive_feedback", "areas_for_improvement", "flags", "scoring", "action_plan"],
    "properties": {
        "_reasoning_trace": {"type": "array", "items": {"type": "string"}},
        "meta": {
            "type": "object",
            "required": ["tutor_id", "group_id", "session_date", "session_summary"],
            "properties": {
                "tutor_id": {"type": "string"},
                "group_id": {"type": "string"},
                "session_date": {"type": "string"},
                "session_summary": {"type": "string"}
            }
        },
        "positive_feedback": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["category", "subcategory", "text", "cite", "timestamp"],
                "properties": {
                    "category": {"type": "string"},
                    "subcategory": {"type": "string"},
                    "text": {"type": "string"},
                    "cite": {"type": "string"},
                    "timestamp": {"type": "string"}
                }
            }
        },
        "areas_for_improvement": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["category", "subcategory", "text", "cite", "timestamp"],
                "properties": {
                    "category": {"type": "string"},
                    "subcategory": {"type": "string"},
                    "text": {"type": "string"},
                    "cite": {"type": "string"},
                    "timestamp": {"type": "string"}
                }
            }
        },
        "flags": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["level", "subcategory", "reason", "cite", "timestamp"],
                "properties": {
                    "level": {"type": "string"},
                    "subcategory": {"type": "string"},
                    "reason": {"type": "string"},
                    "cite": {"type": "string"},
                    "timestamp": {"type": "string"}
                }
            }
        },
        "scoring": {
            "type": "object",
            "required": ["setup", "attitude", "preparation", "curriculum", "teaching", "averages", "final_weighted_score"],
            "properties": {
                "setup": {"type": "array", "items": {"type": "object", "required": ["subcategory", "rating", "reason"], "properties": {"subcategory": {"type": "string"}, "rating": {"type": "integer", "minimum": 0, "maximum": 5}, "reason": {"type": "string"}}}},
                "attitude": {"type": "array", "items": {"type": "object", "required": ["subcategory", "rating", "reason"], "properties": {"subcategory": {"type": "string"}, "rating": {"type": "integer", "minimum": 0, "maximum": 5}, "reason": {"type": "string"}}}},
                "preparation": {"type": "array", "items": {"type": "object", "required": ["subcategory", "rating", "reason"], "properties": {"subcategory": {"type": "string"}, "rating": {"type": "integer", "minimum": 0, "maximum": 5}, "reason": {"type": "string"}}}},
                "curriculum": {"type": "array", "items": {"type": "object", "required": ["subcategory", "rating", "reason"], "properties": {"subcategory": {"type": "string"}, "rating": {"type": "integer", "minimum": 0, "maximum": 5}, "reason": {"type": "string"}}}},
                "teaching": {"type": "array", "items": {"type": "object", "required": ["subcategory", "rating", "reason"], "properties": {"subcategory": {"type": "string"}, "rating": {"type": "integer", "minimum": 0, "maximum": 5}, "reason": {"type": "string"}}}},
                "averages": {
                    "type": "object",
                    "properties": {
                        "setup": {"type": "number"},
                        "attitude": {"type": "number"},
                        "preparation": {"type": "number"},
                        "curriculum": {"type": "number"},
                        "teaching": {"type": "number"}
                    }
                },
                "final_weighted_score": {"type": "number"}
            }
        },
        "action_plan": {"type": "array", "items": {"type": "string"}}
    }
}

# File Search Store for PDF knowledge base (persistent — survives across sessions)
# Disabled per request: always upload all 4 PDFs each run.
FILE_SEARCH_STORE_DISPLAY_NAME = ""

# Temp Files
TEMP_AUDIO_FILENAME = "temp_audio.mp3"
TEMP_FRAMES_DIRNAME = "frames"

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

# Initialize the new google.genai client
client = genai.Client(api_key=API_KEY)


def load_reference_knowledge_base(json_paths=None):
    """
    V38: Load the 4 quality reference JSON files and return them as a single
    structured text block to inject into the prompt.
    This replaces uploading PDFs — the model gets clean, structured JSON text
    instead of raw PDF bytes, which improves comprehension and reduces errors.
    """
    if json_paths is None:
        json_paths = REFERENCE_JSON_FILES

    FILE_LABELS = {
        "quality_guide.json":    "QUALITY GUIDE FOR REVIEWERS",
        "quality_comments.json": "QUALITY COMMENTS — STANDARD TEMPLATES BY CATEGORY",
        "flag_examples.json":    "EXAMPLES OF FLAG COMMENTS (YELLOW / RED FLAG TEMPLATES)",
        "comments_bank.json":    "COMMENTS BANK — REAL REVIEWER EXAMPLES",
    }

    blocks = []
    for path in json_paths:
        fname = os.path.basename(path)
        label = FILE_LABELS.get(fname, fname.replace(".json", "").upper())
        if not os.path.exists(path):
            print(f"[KB] WARNING: JSON not found: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # For comments_bank (large file), include only a condensed subset
        if fname == "comments_bank.json":
            condensed = {
                "title": data.get("title"),
                "purpose": data.get("purpose"),
                "total_unique_comments": data.get("total_unique_comments"),
                "categories": {}
            }
            # Include up to 15 most representative comments per subcategory
            for cat, subcats in data.get("categories", {}).items():
                condensed["categories"][cat] = {}
                for sc, comments in subcats.items():
                    condensed["categories"][cat][sc] = comments[:15]
            data = condensed

        text_block = json.dumps(data, ensure_ascii=False, indent=2)
        blocks.append(f"\n\n{'='*60}\n## KNOWLEDGE BASE: {label}\n{'='*60}\n{text_block}")

    if blocks:
        header = (
            "\n\n# ===== QUALITY REFERENCE KNOWLEDGE BASE (AUTHORITATIVE) =====\n"
            "The following JSON files contain iSchool's OFFICIAL quality standards.\n"
            "You MUST consult these before making ANY finding or score decision.\n"
            "When the knowledge base and your own inference conflict, the knowledge base WINS.\n"
        )
        kb_text = header + "".join(blocks)
        loaded_kb = [b for b in json_paths if os.path.exists(b)]
        print(f"[KB] Loaded {len(loaded_kb)}/4 reference JSON files into prompt")
        return kb_text
    else:
        print("[KB] WARNING: No reference JSON files found — knowledge base empty")
        return ""


def get_or_create_file_search_store(pdf_paths):
    """Create or reuse a File Search Store with the quality reference PDFs.
    
    The store persists across runs — PDFs are indexed once, then reused.
    Returns the store name (e.g. 'fileSearchStores/abc123') or None on failure.
    """
    if not FILE_SEARCH_STORE_DISPLAY_NAME:
        return None
    
    # Check if store already exists
    try:
        existing_stores = list(client.file_search_stores.list())
        for store in existing_stores:
            if getattr(store, 'display_name', '') == FILE_SEARCH_STORE_DISPLAY_NAME:
                # Verify it has documents
                docs = list(client.file_search_stores.documents.list(parent=store.name))
                if len(docs) >= len(pdf_paths):
                    print(f"[FILE_SEARCH] Reusing existing store: {store.name} ({len(docs)} docs)")
                    return store.name
                else:
                    print(f"[FILE_SEARCH] Store exists but only has {len(docs)}/{len(pdf_paths)} docs — rebuilding")
                    try:
                        client.file_search_stores.delete(name=store.name, config={'force': True})
                    except Exception as e:
                        print(f"[FILE_SEARCH] Warning: Could not delete old store: {e}")
    except Exception as e:
        print(f"[FILE_SEARCH] Could not list stores: {e}")
    
    # Create new store
    try:
        print(f"[FILE_SEARCH] Creating new store: {FILE_SEARCH_STORE_DISPLAY_NAME}")
        store = client.file_search_stores.create(
            config={'display_name': FILE_SEARCH_STORE_DISPLAY_NAME}
        )
        print(f"[FILE_SEARCH] Store created: {store.name}")
        
        # Upload each PDF to the store
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                print(f"[FILE_SEARCH] WARNING: PDF not found: {pdf_path}")
                continue
            
            display_name = os.path.basename(pdf_path).replace('.pdf', '')
            print(f"[FILE_SEARCH] Uploading: {display_name}...")
            
            operation = client.file_search_stores.upload_to_file_search_store(
                file=pdf_path,
                file_search_store_name=store.name,
                config={'display_name': display_name}
            )
            
            # Wait for indexing to complete
            while not operation.done:
                time.sleep(3)
                operation = client.operations.get(operation)
            
            print(f"[FILE_SEARCH] Indexed: {display_name}")
        
        # Verify documents
        docs = list(client.file_search_stores.documents.list(parent=store.name))
        print(f"[FILE_SEARCH] Store ready with {len(docs)} documents")
        return store.name
        
    except Exception as e:
        print(f"[FILE_SEARCH] ERROR creating store: {e}")
        print(f"[FILE_SEARCH] Falling back to direct PDF upload")
        return None


def _is_quota_exhausted_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    msg = str(exc)
    return ("RESOURCE_EXHAUSTED" in msg) or (" 429" in msg) or ("quota" in msg.lower())


def _call_genai_with_backoff(fn, *, what: str = "genai", max_attempts: int | None = None):
    """Call a google.genai SDK function with light backoff; exit 42 on persistent quota exhaustion.

    Exit code 42 is intentional: the Node server detects quota exhaustion and delays retries.
    """
    if max_attempts is None:
        max_attempts = max(1, int(os.environ.get("GEMINI_MAX_ATTEMPTS", "3")))

    base = float(os.environ.get("GEMINI_BACKOFF_BASE_SEC", "5"))
    cap = float(os.environ.get("GEMINI_BACKOFF_MAX_SEC", "300"))
    quota_attempts = max(1, int(os.environ.get("GEMINI_429_MAX_ATTEMPTS", "2")))

    quota_seen = 0
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:
            if _is_quota_exhausted_error(e):
                quota_seen += 1
                if quota_seen >= quota_attempts:
                    print(f"[QUOTA_EXHAUSTED] {what}: {e}")
                    sys.exit(42)

            if attempt >= max_attempts:
                raise

            sleep_s = min(cap, base * (2 ** (attempt - 1)))
            sleep_s = sleep_s * (0.7 + random.random() * 0.6)  # jitter 0.7x..1.3x
            print(f"[RETRY] {what} attempt {attempt}/{max_attempts} failed: {e} (sleep {sleep_s:.1f}s)")
            time.sleep(sleep_s)


def delete_uploaded_gemini_files(uploaded_files):
    """Best-effort cleanup for Gemini Files API to avoid accumulating storage.

    Gemini file uploads persist server-side until deleted; leaving them will eventually
    hit per-project storage limits.
    """
    enabled = str(os.environ.get("GEMINI_DELETE_UPLOADED_FILES", "true")).lower() != "false"
    if not enabled:
        return

    if not uploaded_files:
        return

    deleted = 0
    for f, original_path in uploaded_files:
        try:
            name = getattr(f, "name", None)
            if not name:
                continue
            client.files.delete(name=name)
            deleted += 1
        except Exception as e:
            # Don't fail the whole run due to cleanup
            try:
                base = os.path.basename(original_path or "")
            except Exception:
                base = ""
            print(f"[CLEANUP WARNING] Could not delete uploaded file {base}: {e}")

    if deleted:
        print(f"[CLEANUP] Deleted {deleted} uploaded Gemini file(s)")

def _resolve_ffmpeg_exe():
    """Return path to ffmpeg binary, preferring system ffmpeg then imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:
        return None
    return None

def upload_to_gemini(path, mime_type=None, index=None, total=None):
    """Uploads the given file to Gemini sequentially with progress tracking."""
    if index is not None and total is not None:
        prefix = f"Counter: [{index}/{total}]"
    else:
        prefix = "Uploading"
    
    try:
        # Small jitter to reduce request bursts under parallel uploads
        time.sleep(float(os.environ.get("GEMINI_UPLOAD_JITTER_SEC", "0.2")) * random.random())
        file = _call_genai_with_backoff(
            lambda: client.files.upload(file=path, config={"mime_type": mime_type} if mime_type else None),
            what=f"files.upload({os.path.basename(path)})",
        )
        print(f"{prefix} [OK] {file.uri}")
        return (file, path)  # Return tuple for sorting by original path
    except Exception as e:
        print(f"{prefix} [FAIL] Failed to upload {path}: {e}")
        raise

def upload_files_sequentially(files_to_upload):
    """Uploads multiple files sequentially (One after another)."""
    uploaded_files = []
    total_files = len(files_to_upload)
    print(f"Starting sequential upload for {total_files} files...")
    
    for i, (path, mime_type) in enumerate(files_to_upload, 1):
        try:
            file = upload_to_gemini(path, mime_type, index=i, total=total_files)
            uploaded_files.append(file)
        except Exception as e:
            # Error is already printed in upload_to_gemini
            pass
                
    return uploaded_files

def upload_files_parallel(files_to_upload):
    """Uploads multiple files in parallel using ThreadPoolExecutor."""
    uploaded_files = []
    total_files = len(files_to_upload)
    print(f"Starting parallel upload for {total_files} files...")

    max_workers = int(os.environ.get("GEMINI_UPLOAD_MAX_WORKERS", "2"))
    max_workers = max(1, min(8, max_workers))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(upload_to_gemini, path, mime_type, index=i+1, total=total_files): (path, mime_type)
            for i, (path, mime_type) in enumerate(files_to_upload)
        }
        
        for future in concurrent.futures.as_completed(future_to_file):
            try:
                file = future.result()
                uploaded_files.append(file)
            except Exception as e:
                print(f"Upload failed: {e}")
                
    return uploaded_files

def wait_for_files_active(files):
    """Waits for files to be active. Expects (file, path) tuples."""
    print("Waiting for file processing...")
    for f, _ in files:
        file = _call_genai_with_backoff(lambda: client.files.get(name=f.name), what=f"files.get({f.name})")
        while file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(2)
            file = _call_genai_with_backoff(lambda: client.files.get(name=f.name), what=f"files.get({f.name})")
        if file.state.name != "ACTIVE":
            raise Exception(f"File {file.name} failed to process")
    print("...all files ready")

def get_start_time_from_transcript(transcript_path):
    """Parses transcript for first timestamp."""
    print(f"Parsing transcript for start time: {transcript_path}")
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Try VTT format first: 00:13:17.540 --> 00:13:18.659
            match = re.search(r'(\d{2}:\d{2}:\d{2})', content)
            if match:
                start_time = match.group(1)
                print(f"Found start time: {start_time}")
                return start_time
    except Exception as e:
        print(f"Error parsing transcript: {e}")
    return DEFAULT_START_TIME

def _resolve_ffprobe_exe():
    """Return path to ffprobe binary, deriving from ffmpeg location if needed."""
    # Try system ffprobe first
    exe = shutil.which("ffprobe")
    if exe:
        return exe
    # Derive from ffmpeg location (ffprobe is usually next to ffmpeg)
    ffmpeg_exe = _resolve_ffmpeg_exe()
    if ffmpeg_exe:
        ffprobe_candidate = os.path.join(os.path.dirname(ffmpeg_exe), "ffprobe" + (".exe" if os.name == "nt" else ""))
        if os.path.exists(ffprobe_candidate):
            return ffprobe_candidate
    return None

def get_video_duration(video_path):
    """Gets video duration in seconds using ffprobe, falling back to ffmpeg stderr parsing."""
    # Method 1: Try ffprobe (most accurate)
    ffprobe_exe = _resolve_ffprobe_exe()
    if ffprobe_exe:
        cmd = [
            ffprobe_exe,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            dur = float(result.stdout.strip())
            if dur > 0:
                return dur
        except Exception as e:
            print(f"[WARNING] ffprobe failed: {e}")
    
    # Method 2: Fall back to ffmpeg stderr parsing (metadata-only; do NOT decode full video)
    ffmpeg_exe = _resolve_ffmpeg_exe()
    if ffmpeg_exe:
        cmd = [ffmpeg_exe, "-hide_banner", "-i", video_path]
        try:
            # ffmpeg returns non-zero here because no output is specified; stderr still contains metadata.
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=20)
            import re as _re
            m = _re.search(r"Duration:\s*(\d+):(\d+):(\d+)\.(\d+)", result.stderr)
            if m:
                h, mn, s, cs = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
                dur = h * 3600 + mn * 60 + s + cs / 100.0
                if dur > 0:
                    print(f"[INFO] Got duration from ffmpeg: {dur:.2f}s")
                    return dur
        except subprocess.TimeoutExpired:
            print("[WARNING] ffmpeg duration fallback timed out")
        except Exception as e:
            print(f"[WARNING] ffmpeg duration fallback failed: {e}")
    
    print(f"[WARNING] Could not determine video duration (no ffprobe/ffmpeg found)")
    return 0


def probe_media_metadata(media_path):
    """Return ffprobe metadata JSON for a media file, or None if unavailable."""
    ffprobe_exe = _resolve_ffprobe_exe()
    if not ffprobe_exe:
        return None
    cmd = [
        ffprobe_exe,
        "-v", "error",
        "-show_streams",
        "-show_format",
        "-print_format", "json",
        media_path,
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=MEDIA_PROBE_TIMEOUT_SEC,
        )
        return json.loads(result.stdout or "{}")
    except Exception as e:
        print(f"[WARNING] ffprobe metadata failed for {os.path.basename(media_path)}: {e}")
        return None


def validate_video_input(video_path, duration_hint_sec=0):
    """Validate that the input recording exists, has valid duration, and includes audio."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")

    duration = duration_hint_sec or get_video_duration(video_path)
    if duration <= 0:
        raise ValueError("Input video duration is invalid (0 seconds). The recording may be corrupt.")
    if duration < MIN_VALID_MEDIA_DURATION_SEC:
        raise ValueError(
            f"Input video duration is too short ({duration:.1f}s). Expected at least {MIN_VALID_MEDIA_DURATION_SEC}s."
        )

    metadata = probe_media_metadata(video_path)
    has_audio_stream = None
    if metadata:
        streams = metadata.get("streams", [])
        has_audio_stream = any((s or {}).get("codec_type") == "audio" for s in streams)
        if not has_audio_stream:
            raise ValueError("Input video has no audio stream. Recording appears invalid/corrupt for audio analysis.")

    print(f"[MEDIA-OK] Video validated: {os.path.basename(video_path)} ({duration:.1f}s)")
    return {
        "duration_sec": duration,
        "audio_stream_present": has_audio_stream,
        "probe_used": bool(metadata),
    }


def validate_audio_file(audio_path):
    """Validate extracted MP3 is usable and non-corrupt."""
    if not audio_path or not os.path.exists(audio_path):
        raise ValueError("Extracted audio file is missing.")

    size_bytes = os.path.getsize(audio_path)
    if size_bytes < MIN_VALID_AUDIO_FILE_BYTES:
        raise ValueError(
            f"Extracted audio file too small ({size_bytes} bytes). Likely corrupt extraction."
        )

    metadata = probe_media_metadata(audio_path)
    if metadata:
        streams = metadata.get("streams", [])
        has_audio_stream = any((s or {}).get("codec_type") == "audio" for s in streams)
        if not has_audio_stream:
            raise ValueError("Extracted audio has no audio stream. Corrupt audio output.")

        duration_str = ((metadata.get("format") or {}).get("duration") or "0").strip() if isinstance(metadata, dict) else "0"
        try:
            duration = float(duration_str)
        except Exception:
            duration = 0
        if duration <= 0:
            raise ValueError("Extracted audio duration is invalid (0 seconds). Corrupt audio output.")
        if duration < MIN_VALID_AUDIO_DURATION_SEC:
            raise ValueError(
                f"Extracted audio too short ({duration:.1f}s). Expected at least {MIN_VALID_AUDIO_DURATION_SEC}s."
            )
        print(f"[AUDIO-OK] Audio validated: {os.path.basename(audio_path)} ({duration:.1f}s, {size_bytes:,} bytes)")
        return {
            "duration_sec": duration,
            "size_bytes": size_bytes,
            "probe_mode": "ffprobe",
        }
    else:
        # Fallback when ffprobe is unavailable: validate duration via ffmpeg parser + size.
        duration = get_video_duration(audio_path)
        if duration <= 0:
            raise ValueError("Could not verify extracted audio duration (ffprobe unavailable and ffmpeg parse failed).")
        if duration < MIN_VALID_AUDIO_DURATION_SEC:
            raise ValueError(
                f"Extracted audio too short ({duration:.1f}s). Expected at least {MIN_VALID_AUDIO_DURATION_SEC}s."
            )
        print(
            f"[AUDIO-OK] Audio validated via ffmpeg fallback: "
            f"{os.path.basename(audio_path)} ({duration:.1f}s, {size_bytes:,} bytes)"
        )
        return {
            "duration_sec": duration,
            "size_bytes": size_bytes,
            "probe_mode": "ffmpeg_fallback",
        }

def time_str_to_seconds(time_str):
    """Converts HH:MM:SS to seconds."""
    try:
        h, m, s = map(int, time_str.split(':'))
        return h * 3600 + m * 60 + s
    except ValueError:
        return 0

def extract_frame_at_time(video_path, time_sec, output_path):
    """Extracts a single frame at a specific time."""
    ffmpeg_exe = _resolve_ffmpeg_exe()
    if not ffmpeg_exe:
        print("[WARNING] ffmpeg not found. Skipping frame extraction.")
        return
    cmd = [
        ffmpeg_exe,
        "-ss", str(time_sec),
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", str(FRAME_QUALITY),
        "-vf", f"scale={FRAME_WIDTH}:-1",
        "-y",
        output_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"[WARNING] Frame extraction failed at {time_sec}s (exit={result.returncode})")

def extract_audio(video_path, output_path):
    """Extracts audio from video."""
    print(f"Extracting audio to {output_path}...")
    ffmpeg_exe = _resolve_ffmpeg_exe()
    if not ffmpeg_exe:
        print(f"[WARNING] ffmpeg not found. Skipping audio extraction.")
        return None
    cmd = [
        ffmpeg_exe,
        "-i", video_path,
        "-vn",
        "-acodec", "libmp3lame",
        "-q:a", "2",
        "-y",
        output_path
    ]
    try:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
            timeout=AUDIO_EXTRACTION_TIMEOUT_SEC,
            text=True,
        )
        return output_path
    except subprocess.TimeoutExpired:
        print(f"[WARNING] Audio extraction timed out after {AUDIO_EXTRACTION_TIMEOUT_SEC}s")
        return None
    except subprocess.CalledProcessError as e:
        err = (e.stderr or "")[:300]
        print(f"[WARNING] Audio extraction failed: {e}. ffmpeg stderr: {err}")
        return None

def extract_resources(video_path, start_time):
    """Extracts frames using parallel ffmpeg seeking (Super Fast)."""
    print("--- Extracting Resources (Super Fast Parallel) ---")
    
    base_dir = os.path.dirname(video_path)
    frames_dir = os.path.join(base_dir, TEMP_FRAMES_DIRNAME)
    
    # Check if frames already exist
    if os.path.exists(frames_dir):
        existing_frames = glob.glob(os.path.join(frames_dir, "*.jpg"))
        if len(existing_frames) >= 15:  # Accept if we have at least 15 frames (enough for analysis)
            print(f"Frames already exist in {frames_dir} ({len(existing_frames)} frames). Skipping extraction.")
            return frames_dir
        else:
            print("Found partial frames, re-extracting...")
            shutil.rmtree(frames_dir)
            
    os.makedirs(frames_dir)
    
    start_seconds = time_str_to_seconds(start_time)
    duration = get_video_duration(video_path)
    
    if duration == 0:
        print("Could not determine duration, using default range...")
        duration = start_seconds + 3600 # Default to 1 hour if unknown

    print(f"Video Duration: {duration:.2f}s, Start Time: {start_seconds}s")
    
    timestamps = []
    current_time = start_seconds
    while current_time < duration:
        timestamps.append(current_time)
        current_time += FRAME_EXTRACTION_INTERVAL
        
    print(f"Extracting {len(timestamps)} frames in parallel...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        for i, ts in enumerate(timestamps):
            ts_int = int(ts)
            hh = ts_int // 3600
            mm = (ts_int % 3600) // 60
            ss = ts_int % 60
            frame_name = f"Frame-{i+1}-{hh:02d}-{mm:02d}-{ss:02d}.jpg"
            output_path = os.path.join(frames_dir, frame_name)
            futures.append(executor.submit(extract_frame_at_time, video_path, ts, output_path))
            
        # Wait for all to complete
        concurrent.futures.wait(futures)
    
    # Clean up: remove any 0-byte failed frames and count successes
    actual_frames = []
    for f in sorted(glob.glob(os.path.join(frames_dir, "*.jpg"))):
        if os.path.getsize(f) > 0:
            actual_frames.append(f)
        else:
            os.remove(f)  # Remove empty/failed frame files
    
    if len(actual_frames) < len(timestamps):
        print(f"[INFO] Extracted {len(actual_frames)}/{len(timestamps)} frames successfully (rest were beyond video end)")
            
    return frames_dir

def should_rerun_analysis(data):
    """
    Checks if analysis should be re-run due to quality concerns:
    1. Any subcategory rating being 0 (indicates incomplete evaluation)
    2. Final weighted score being less than 68% (below acceptable threshold)
    
    This acts as a quality gate to ensure all sessions are properly evaluated.
    If either condition is triggered, the analysis is re-run to get a more accurate assessment.
    
    Args:
        data (dict): Parsed JSON data from the initial analysis
        
    Returns:
        tuple: (should_rerun: bool, reason: str)
        
    Example:
        >>> data = {"scoring": {"final_weighted_score": 65, "setup": [{"rating": 0}]}}
        >>> should_rerun, reason = should_rerun_analysis(data)
        >>> should_rerun
        True
        >>> reason
        "Category 'setup' subcategory 'Unknown' has rating 0"
    """
    try:
        scoring = data.get("scoring", {})
        final_score = scoring.get("final_weighted_score", 0)
        
        # Check if score is below 68%
        if final_score < 68:
            return True, f"Score {final_score} is below 68% threshold"
        
        # Check if any subcategory has rating 0
        categories = ["setup", "attitude", "preparation", "curriculum", "teaching"]
        for cat in categories:
            if cat in scoring and isinstance(scoring[cat], list):
                for item in scoring[cat]:
                    rating = item.get("rating", 0)
                    if rating == 0:
                        subcategory = item.get("subcategory", "Unknown")
                        return True, f"Category '{cat}' subcategory '{subcategory}' has rating 0"
        
        return False, "Analysis meets quality threshold"
    except Exception as e:
        return False, f"Error checking analysis: {e}"

def repair_truncated_json(json_text):
    """
    Attempts to repair truncated JSON output from the model.
    Common issues: unclosed brackets/braces, trailing commas, incomplete strings.
    Returns the repaired JSON string (may still be invalid).
    """
    if not json_text or not json_text.strip():
        return json_text
    
    text = json_text.strip()
    
    # Remove markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json) and last line (```)
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = "\n".join(lines).strip()
    
    # Try parsing as-is first
    try:
        json.loads(text)
        return text  # Already valid
    except json.JSONDecodeError:
        pass
    
    # Remove any trailing incomplete string (ends with unclosed quote)
    # Find the last complete key-value pair
    lines = text.split("\n")
    while lines:
        last = lines[-1].strip()
        # Remove empty lines, lines with only partial content
        if not last or last in (',', ''):
            lines.pop()
            continue
        break
    text = "\n".join(lines)
    
    # Remove trailing commas before closing brackets
    import re
    text = re.sub(r',\s*([}\]])', r'\1', text)
    
    # Count unclosed brackets and braces
    open_braces = text.count('{') - text.count('}')
    open_brackets = text.count('[') - text.count(']')
    
    # If there are unclosed structures, try to close them
    if open_braces > 0 or open_brackets > 0:
        # Remove any trailing incomplete value (partial string, number, etc.)
        # Find last complete element
        text = text.rstrip()
        if text and text[-1] not in ('}', ']', '"', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'e', 'l', 'u'):
            # Likely truncated mid-value, try to remove the partial part
            last_comma = text.rfind(',')
            last_brace = max(text.rfind('{'), text.rfind('['))
            cutoff = max(last_comma, last_brace)
            if cutoff > 0:
                text = text[:cutoff + 1]
        
        # Remove trailing comma
        text = text.rstrip().rstrip(',')
        
        # Close unclosed structures
        open_braces = text.count('{') - text.count('}')
        open_brackets = text.count('[') - text.count(']')
        text += ']' * max(0, open_brackets)
        text += '}' * max(0, open_braces)
    
    return text


def validate_json_response(json_text):
    """
    Validates that the JSON response is complete and properly structured.
    Returns (is_valid, data_or_error_msg)
    
    Attempts repair of truncated JSON before validation.
    A valid response must have:
    1. Valid JSON syntax
    2. Non-empty object (not {})
    3. Required top-level keys: meta, scoring
    4. Scoring section with at least one category
    """
    try:
        # Check for empty or whitespace-only response
        if not json_text or not json_text.strip():
            return False, "Empty response received"
        
        # Attempt repair before parsing
        repaired = repair_truncated_json(json_text)
        
        # Try repaired version first, fall back to original
        data = None
        for attempt_text in [repaired, json_text]:
            try:
                data = json.loads(attempt_text)
                if data:
                    break
            except json.JSONDecodeError:
                continue
        
        if data is None:
            return False, f"Invalid JSON syntax (repair also failed)"
        
        # Unwrap single-element arrays: model sometimes returns [{...}] instead of {...}
        if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
            print("[AUTO-FIX] Unwrapped single-element JSON array to object")
            data = data[0]
        
        # Check for empty object
        if not data or data == {}:
            return False, "Empty JSON object received ({})"
        
        # AUTO-FIX: Remap common alternative key names to expected keys
        key_mapping = {
            # meta alternatives
            "metadata": "meta",
            "session_info": "meta",
            "session_metadata": "meta",
            "session_meta": "meta",
            # scoring alternatives
            "scores": "scoring",
            "subcategory_scores": "scoring",
            "category_scores": "scoring",
            "ratings": "scoring",
            "score_breakdown": "scoring",
            "score": "scoring",
            # positive_feedback alternatives
            "positive_highlights": "positive_feedback",
            "positives": "positive_feedback",
            "strengths": "positive_feedback",
            # areas_for_improvement alternatives
            "improvements": "areas_for_improvement",
            "areas_of_improvement": "areas_for_improvement",
            "improvement_areas": "areas_for_improvement",
            "weaknesses": "areas_for_improvement",
        }
        remapped = False
        for wrong_key, correct_key in key_mapping.items():
            if wrong_key in data and correct_key not in data:
                data[correct_key] = data.pop(wrong_key)
                print(f"[AUTO-FIX] Remapped key '{wrong_key}' -> '{correct_key}'")
                remapped = True
        
        # If scoring is a flat dict of numbers (e.g. {"S": 5, "A": 4.5, ...}), try to restructure
        scoring_val = data.get("scoring")
        if isinstance(scoring_val, dict) and "final_weighted_score" not in scoring_val:
            # Check if it has single-letter keys like S, A, P, C, T
            letter_keys = {"S", "A", "P", "C", "T"}
            if letter_keys.issubset(set(scoring_val.keys())):
                print("[AUTO-FIX] Restructuring flat scoring dict with letter keys")
                cat_map = {"S": "setup", "A": "attitude", "P": "preparation", "C": "curriculum", "T": "teaching"}
                new_scoring = {}
                for letter, full_name in cat_map.items():
                    val = scoring_val.get(letter, 5)
                    new_scoring[full_name] = [{"subcategory": full_name.title(), "rating": val, "reason": "Auto-mapped from flat structure"}]
                new_scoring["averages"] = {full_name: scoring_val.get(letter, 5) for letter, full_name in cat_map.items()}
                # Calculate weighted score
                weights = {"setup": 0.25, "attitude": 0.20, "preparation": 0.15, "curriculum": 0.15, "teaching": 0.25}
                total = sum((new_scoring["averages"][cat] / 5) * 100 * w for cat, w in weights.items())
                new_scoring["final_weighted_score"] = round(total, 1)
                data["scoring"] = new_scoring
                remapped = True
        
        # Check required top-level keys
        required_keys = ["meta", "scoring"]
        missing_keys = [k for k in required_keys if k not in data]
        if missing_keys:
            return False, f"Missing required keys: {missing_keys}"
        
        # Check scoring has content
        scoring = data.get("scoring", {})
        if not scoring:
            return False, "Scoring section is empty"
        
        # Check at least one category has ratings
        categories = ["setup", "attitude", "preparation", "curriculum", "teaching"]
        has_ratings = False
        for cat in categories:
            if cat in scoring and isinstance(scoring[cat], list) and len(scoring[cat]) > 0:
                has_ratings = True
                break
        
        if not has_ratings:
            return False, "No category ratings found in scoring"
        
        return True, data
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON syntax: {e}"
    except Exception as e:
        return False, f"Validation error: {e}"

def compute_median_score(scores):
    """
    Computes the median score from a list of scores.
    The median is more robust to outliers than the mean.
    """
    if not scores:
        return 0
    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    if n % 2 == 1:
        return sorted_scores[n // 2]
    else:
        return (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2

def select_best_analysis_by_median(analyses):
    """
    Given multiple analysis results, selects the one closest to the median score.
    This ensures we pick a representative analysis, not an outlier.
    
    Args:
        analyses: List of (json_text, data_dict, score) tuples
        
    Returns:
        tuple: (best_json_text, best_data, median_score, all_scores, variance)
    """
    if not analyses:
        return None, {}, 0, [], 0
    
    scores = [a[2] for a in analyses]
    median = compute_median_score(scores)
    
    # Calculate variance
    variance = max(scores) - min(scores) if len(scores) > 1 else 0
    
    # Find analysis closest to median
    best_idx = 0
    min_diff = abs(scores[0] - median)
    for i, score in enumerate(scores):
        diff = abs(score - median)
        if diff < min_diff:
            min_diff = diff
            best_idx = i
    
    best_json, best_data, best_score = analyses[best_idx]
    
    # Update the score in data to be the median for consistency
    if "scoring" in best_data:
        best_data["scoring"]["final_weighted_score"] = round(median, 1)
        best_data["scoring"]["_consistency_info"] = {
            "individual_scores": scores,
            "median_score": round(median, 1),
            "score_variance": round(variance, 1),
            "runs": len(scores),
            "reliable": variance <= SCORE_VARIANCE_THRESHOLD
        }
        best_json = json.dumps(best_data, indent=2)
    
    return best_json, best_data, median, scores, variance

def compare_and_keep_best(data1, data2):
    """
    Compares two analysis JSON objects and returns the one with the higher score.
    This function is used when analysis is re-run to select the best result.
    
    Args:
        data1 (dict): First analysis result
        data2 (dict): Second (retry) analysis result
        
    Returns:
        tuple: (best_data: dict, score1: float, score2: float, selected: str)
               where selected is either "First" or "Second"
               
    Example:
        >>> data1 = {"scoring": {"final_weighted_score": 72}}
        >>> data2 = {"scoring": {"final_weighted_score": 75}}
        >>> best, s1, s2, sel = compare_and_keep_best(data1, data2)
        >>> sel
        "Second"
        >>> best == data2
        True
    """
    try:
        score1 = data1.get("scoring", {}).get("final_weighted_score", 0)
        score2 = data2.get("scoring", {}).get("final_weighted_score", 0)
        
        if score1 >= score2:
            return data1, score1, score2, "First"
        else:
            return data2, score1, score2, "Second"
    except Exception as e:
        print(f"Error comparing analyses: {e}")
        return data1, 0, 0, "First (default)"

def generate_html_report_from_json(json_path):
    """Generates a premium, fixed-style professional HTML report from JSON data."""
    html_path = os.path.splitext(json_path)[0] + ".html"
    print(f"\n--- Generating Premium HTML Report: {html_path} ---")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract data from JSON
        final_score = data.get('scoring', {}).get('final_weighted_score', 0)
        cat_avg = data.get('scoring', {}).get('averages', {})
        cat_scores = {
            'Setup': cat_avg.get('setup', 0),
            'Attitude': cat_avg.get('attitude', 0),
            'Preparation': cat_avg.get('preparation', 0),
            'Curriculum': cat_avg.get('curriculum', 0),
            'Teaching': cat_avg.get('teaching', 0)
        }

        # Score-based styling
        score_color = "#10b981" if final_score >= 90 else "#f59e0b" if final_score >= 70 else "#ef4444"
        perf_label = "EXCELLENT" if final_score >= 90 else "GOOD" if final_score >= 70 else "NEEDS IMPROVEMENT"
        
        # Progress circle math
        dash_offset = 283 - (final_score / 100 * 283)

        # Helper to render lists
        def render_feedback(items, css_class):
            html = ""
            for item in items:
                cat = item.get('category', '?')
                sub = item.get('subcategory', 'General')
                text = item.get('text', '')
                cite = item.get('cite', '')
                time = item.get('timestamp', '')
                html += f'<div class="feedback-box {css_class}"><strong>[{cat}] {sub}:</strong><p>{text}</p>'
                if cite:
                    html += f'<small class="cite">{cite}</small>'
                if time:
                    html += f'<small class="timestamp">⏱️ {time}</small>'
                html += '</div>'
            return html

        def render_flags(items):
            html = ""
            for item in items:
                level = item.get('level', 'Yellow')
                # Determine class based on level
                css_class = "f-box-red" if "Red" in level else "f-box-yellow"
                sub = item.get('subcategory', '')
                reason = item.get('reason', '')
                cite = item.get('cite', '')
                time = item.get('timestamp', '')
                html += f'<div class="feedback-box {css_class}">🚩 <strong>{level} Flag: {sub}</strong><p>{reason}</p>'
                if cite:
                    html += f'<small class="cite">{cite}</small>'
                if time:
                    html += f'<small class="timestamp">⏱️ {time}</small>'
                html += '</div>'
            return html

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>iSchool | Quality Audit Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --primary: #4f46e5;
            --primary-dark: #3730a3;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --text-main: #1f2937;
            --text-muted: #6b7280;
            --bg-body: #f9fafb;
            --bg-card: #ffffff;
            --score-color: {score_color};
        }}

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Outfit', sans-serif; 
            background-color: var(--bg-body); 
            color: var(--text-main); 
            line-height: 1.6;
            overflow-x: hidden;
        }}

        /* Dynamic Background */
        body::before {{
            content: '';
            position: fixed;
            top: 0; left: 0; width: 100%; height: 350px;
            background: linear-gradient(135deg, var(--primary-dark) 0%, #7c3aed 100%);
            z-index: -1;
            clip-path: polygon(0 0, 100% 0, 100% 80%, 0 100%);
        }}

        .wrapper {{ max-width: 1100px; margin: 40px auto; padding: 0 20px; }}

        /* Glassmorphic Header */
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 30px;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: white;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}

        .logo-box img {{ height: 100px; filter: brightness(0) invert(1); }}
        .header-info h1 {{ font-size: 1.8rem; font-weight: 700; letter-spacing: -0.5px; }}
        .header-info p {{ opacity: 0.8; font-weight: 300; }}

        /* Main Dashboard Layout */
        .dashboard-grid {{
            display: block;
            margin-bottom: 30px;
        }}

        .card {{
            background: var(--bg-card);
            border-radius: 24px;
            padding: 30px;
            box-shadow: 0 4px 25px rgba(0,0,0,0.05);
            border: 1px solid #f1f5f9;
        }}

        /* Score Card Styling */
        .score-card {{ 
            display: flex; 
            flex-direction: row; 
            align-items: center; 
            justify-content: space-around; 
            margin-bottom: 30px;
        }}
        .score-circle {{ position: relative; width: 180px; height: 180px; margin-bottom: 0; }}
        .score-circle svg {{ transform: rotate(-90deg); width: 100%; height: 100%; }}
        .score-circle .bg {{ fill: none; stroke: #f1f5f9; stroke-width: 8; }}
        .score-circle .progress {{ fill: none; stroke: var(--score-color); stroke-width: 10; stroke-linecap: round; transition: 1s ease-out; }}
        .score-val {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 3rem; font-weight: 800; color: var(--score-color); }}
        .perf-badge {{ padding: 8px 20px; border-radius: 50px; background: var(--score-color); color: white; font-weight: 700; font-size: 0.8rem; letter-spacing: 1px; margin-top: 10px; }}
        
        .score-left-pane {{ display: flex; flex-direction: column; align-items: center; }}

        /* Category Breakdown */
        .cat-list {{ width: 50%; margin-top: 0; }}
        .cat-item {{ margin-bottom: 18px; }}
        .cat-head {{ display: flex; justify-content: space-between; margin-bottom: 6px; font-weight: 600; font-size: 0.85rem; color: var(--text-muted); }}
        .bar-bg {{ height: 8px; background: #f1f5f9; border-radius: 10px; overflow: hidden; }}
        .bar-fill {{ height: 100%; background: linear-gradient(90deg, var(--primary) 0%, #9333ea 100%); border-radius: 10px; }}

        /* Report Content Styling */
        .report-section {{ background: white; border-radius: 24px; padding: 40px; box-shadow: 0 4px 25px rgba(0,0,0,0.05); border: 1px solid #f1f5f9; }}
        .report-content {{ font-size: 1.05rem; color: var(--text-main); }}
        
        h2 {{ color: var(--primary); font-size: 1.4rem; margin: 35px 0 20px; display: flex; align-items: center; gap: 12px; border-bottom: 2px solid #f1f5f9; padding-bottom: 12px; }}
        h2:first-child {{ margin-top: 0; }}
        
        .feedback-box {{ padding: 18px 22px; border-radius: 16px; margin-bottom: 15px; border-left: 6px solid; transition: transform 0.2s; }}
        .feedback-box:hover {{ transform: translateX(5px); }}
        .feedback-box p {{ margin: 8px 0; line-height: 1.5; }}
        .feedback-box .cite {{ display: block; margin-top: 8px; color: #9ca3af; font-size: 0.85rem; }}
        .feedback-box .timestamp {{ display: block; margin-top: 4px; color: #9ca3af; font-size: 0.85rem; }}
        .p-box {{ background: #f0fdf4; border-color: var(--success); color: #065f46; }}
        .i-box {{ background: #fffbeb; border-color: var(--warning); color: #92400e; }}
        
        /* Flag Colors */
        .f-box-yellow {{ background: #FEFCE8; border-color: #EAB308; color: #713F12; font-weight: 600; }}
        .f-box-red {{ background: #FEF2F2; border-color: #EF4444; color: #B91C1C; font-weight: 600; }}

        /* Professional Tables */
        .table-container {{ overflow-x: auto; margin: 25px 0; border-radius: 16px; border: 1px solid #f1f5f9; }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; }}
        th {{ background: #f8fafc; padding: 16px 20px; color: var(--text-muted); font-weight: 700; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 1px; border-bottom: 2px solid #f1f5f9; }}
        td {{ padding: 16px 20px; border-bottom: 1px solid #f1f5f9; font-size: 0.95rem; }}
        tr:last-child td {{ border-bottom: none; }}
        
        pre {{ white-space: pre-wrap; font-family: 'Outfit', sans-serif; }}

        @media (max-width: 900px) {{
            .dashboard-grid {{ display: block; }}
            .score-card {{ flex-direction: column; }}
            .cat-list {{ width: 100%; margin-top: 20px; }}
            header {{ flex-direction: column; text-align: center; gap: 20px; }}
        }}
    </style>
</head>
<body>
    <div class="wrapper">
        <header>
            <div class="logo-box">
                <img src="https://www.webit.network/img/portfolio/ischool/logo.png" alt="iSchool">
            </div>
            <div class="header-info">
                <h1>QUALITY AUDIT REPORT</h1>
                <p>Forensic Session Analysis Dashboard</p>
            </div>
        </header>

        <div class="dashboard-grid">
            <aside class="card score-card">
                <div class="score-left-pane">
                    <div class="score-circle">
                        <svg viewBox="0 0 100 100">
                            <circle class="bg" cx="50" cy="50" r="45"></circle>
                            <circle class="progress" cx="50" cy="50" r="45" style="stroke-dasharray: 283; stroke-dashoffset: {dash_offset};"></circle>
                        </svg>
                        <div class="score-val">{final_score:.0f}</div>
                    </div>
                    <div class="perf-badge">{perf_label}</div>
                </div>
                
                <div class="cat-list">
                    <div class="cat-item">
                        <div class="cat-head"><span>Setup</span><span>{cat_scores['Setup']}/5</span></div>
                        <div class="bar-bg"><div class="bar-fill" style="width: {cat_scores['Setup']/5*100}%"></div></div>
                    </div>
                    <div class="cat-item">
                        <div class="cat-head"><span>Attitude</span><span>{cat_scores['Attitude']}/5</span></div>
                        <div class="bar-bg"><div class="bar-fill" style="width: {cat_scores['Attitude']/5*100}%"></div></div>
                    </div>
                    <div class="cat-item">
                        <div class="cat-head"><span>Preparation</span><span>{cat_scores['Preparation']}/5</span></div>
                        <div class="bar-bg"><div class="bar-fill" style="width: {cat_scores['Preparation']/5*100}%"></div></div>
                    </div>
                    <div class="cat-item">
                        <div class="cat-head"><span>Curriculum</span><span>{cat_scores['Curriculum']}/5</span></div>
                        <div class="bar-bg"><div class="bar-fill" style="width: {cat_scores['Curriculum']/5*100}%"></div></div>
                    </div>
                    <div class="cat-item">
                        <div class="cat-head"><span>Teaching</span><span>{cat_scores['Teaching']}/5</span></div>
                        <div class="bar-bg"><div class="bar-fill" style="width: {cat_scores['Teaching']/5*100}%"></div></div>
                    </div>
                </div>
            </aside>

            <main class="report-section">
                <div class="report-content">
                    <h2>📋 Session Information</h2>
                    <div class="feedback-box p-box">
                        <strong>Tutor:</strong> {data.get('meta', {}).get('tutor_id', 'N/A')}<br>
                        <strong>Date:</strong> {data.get('meta', {}).get('session_date', 'N/A')}<br>
                        <strong>Summary:</strong> {data.get('meta', {}).get('session_summary', 'N/A')}
                    </div>

                    <h2> Audit Summary</h2>
                    <p>{data.get('meta', {}).get('session_summary', 'Analysis completed successfully.')}</p>

                    <h2>✅ Positive Highlights</h2>
                    {render_feedback(data.get('positive_feedback', []), 'p-box')}

                    <h2>⚠️ Improvement Points</h2>
                    {render_feedback(data.get('areas_for_improvement', []), 'i-box')}

                    <h2>🚩 Compliance Violations</h2>
                    {render_flags(data.get('flags', []))}

                    <h2>🎯 Recommended Actions</h2>
                    <ul>
                        {"".join([f"<li>{x}</li>" for x in data.get('action_plan', [])])}
                    </ul>
                </div>
            </main>
        </div>
    </div>
</body>
</html>"""
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"[SUCCESS] Premium Dashboard HTML Report created: {html_path}")
    except Exception as e:
        print(f"Error creating HTML from JSON: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# RAG ANALYSIS LOGIC
# ============================================================================

def _merge_step1_step2(step1_data, step2_data):
    """Merge Step 1 (initial) and Step 2 (deep audit) reports into ONE report.

    Strategy:
    - meta: take from Step 2 (more refined)
    - positive_feedback: union, dedup by subcategory
    - areas_for_improvement: union, dedup by (subcategory + similarity)
    - flags: union, dedup by subcategory
    - _reasoning_trace: combine both
    - scoring: take the LOWER (stricter) rating per subcategory
    - action_plan: union dedup
    """
    merged = {}

    # Meta — prefer Step 2 (refined)
    merged["meta"] = step2_data.get("meta", step1_data.get("meta", {}))

    # Reasoning trace — combine
    t1 = step1_data.get("_reasoning_trace", [])
    t2 = step2_data.get("_reasoning_trace", [])
    merged["_reasoning_trace"] = t1 + t2

    # Positive feedback — union dedup by subcategory (keep first seen)
    pos_seen = set()
    merged_pos = []
    for item in step1_data.get("positive_feedback", []) + step2_data.get("positive_feedback", []):
        key = (item.get("subcategory", "")).lower().strip()
        if key not in pos_seen:
            pos_seen.add(key)
            merged_pos.append(item)
    merged["positive_feedback"] = merged_pos

    # Areas for improvement — union dedup by subcategory text similarity
    merged_issues = []
    seen_keys = set()  # (subcategory_lower, first_30_chars_of_text)
    for item in step1_data.get("areas_for_improvement", []) + step2_data.get("areas_for_improvement", []):
        subcat = (item.get("subcategory", "")).lower().strip()
        text_sig = (item.get("text", ""))[:60].lower().strip()
        key = (subcat, text_sig)
        if key not in seen_keys:
            seen_keys.add(key)
            merged_issues.append(item)
    merged["areas_for_improvement"] = merged_issues

    # Flags — union dedup by subcategory
    flag_seen = set()
    merged_flags = []
    for flag in step1_data.get("flags", []) + step2_data.get("flags", []):
        key = (flag.get("subcategory", "")).lower().strip()
        if key not in flag_seen:
            flag_seen.add(key)
            merged_flags.append(flag)
    merged["flags"] = merged_flags

    # Scoring — take the LOWER (stricter) rating per subcategory
    all_cats = ["setup", "attitude", "preparation", "curriculum", "teaching"]
    merged_scoring = {}
    for cat in all_cats:
        s1_items = {item["subcategory"]: item for item in step1_data.get("scoring", {}).get(cat, [])}
        s2_items = {item["subcategory"]: item for item in step2_data.get("scoring", {}).get(cat, [])}
        all_subcats = set(s1_items.keys()) | set(s2_items.keys())
        cat_items = []
        for sc in sorted(all_subcats):
            r1 = s1_items.get(sc, {}).get("rating", 5)
            r2 = s2_items.get(sc, {}).get("rating", 5)
            lower = min(r1, r2)
            # Take the reason from whichever gave the lower rating
            reason = s2_items.get(sc, {}).get("reason", s1_items.get(sc, {}).get("reason", ""))
            if r1 < r2:
                reason = s1_items.get(sc, {}).get("reason", reason)
            cat_items.append({"subcategory": sc, "rating": lower, "reason": reason})
        merged_scoring[cat] = cat_items
    merged_scoring["averages"] = {}
    merged_scoring["final_weighted_score"] = 0
    merged["scoring"] = merged_scoring

    # Action plan — union dedup
    ap_seen = set()
    merged_ap = []
    for item in step1_data.get("action_plan", []) + step2_data.get("action_plan", []):
        sig = item[:40].lower().strip()
        if sig not in ap_seen:
            ap_seen.add(sig)
            merged_ap.append(item)
    merged["action_plan"] = merged_ap[:5]  # cap at 5

    return merged


def recalculate_score(json_text):
    """Recalculate ratings/scores using enforced per-subcategory deduction rules.

    Requested Step A logic (applied per subcategory):
    - 5: 0 improvements
    - 4: 1-4 improvements
    - 3: 5 improvements
    - 2: 6+ improvements OR 3+ yellow flags
    - 1: 10+ improvements
    - 0: no show / total failure (only if explicit no-show signal appears)

    Note: The user-provided rule overlaps at 6 improvements (both 3 and 2).
    This implementation resolves overlap by treating 6+ as rating 2.
    """
    def _has_total_failure_signal(payload):
        checks = []
        checks.extend(payload.get("flags", []) if isinstance(payload.get("flags"), list) else [])
        checks.extend(payload.get("areas_for_improvement", []) if isinstance(payload.get("areas_for_improvement"), list) else [])
        for item in checks:
            if not isinstance(item, dict):
                continue
            txt = " ".join(
                str(item.get(k, "")) for k in ("reason", "text", "cite", "subcategory", "level")
            ).lower()
            if "no show" in txt or "total failure" in txt:
                return True
        return False

    def _deduced_rating(issue_count, yellow_flag_count, total_failure=False):
        if total_failure:
            return 0
        if issue_count >= 10:
            return 1
        if yellow_flag_count >= 3 or issue_count >= 6:
            return 2
        if issue_count >= 5:
            return 3
        if issue_count >= 1:
            return 4
        return 5

    try:
        data = json.loads(json_text)
        if "scoring" in data:
            scoring = data["scoring"]
            weights = {"setup": 0.25, "attitude": 0.20, "preparation": 0.15, "curriculum": 0.15, "teaching": 0.25}

            # Build subcategory issue counters from current findings.
            issue_counts = {}
            for item in data.get("areas_for_improvement", []):
                if not isinstance(item, dict):
                    continue
                # V26: [Advice] items are informational — they do NOT count as deductions
                item_text = str(item.get("text", "")).strip()
                if item_text.startswith("[Advice]"):
                    continue
                sub = _normalize_subcategory_name(item.get("subcategory", ""))
                if not sub:
                    continue
                issue_counts[sub] = issue_counts.get(sub, 0) + 1

            yellow_counts = {}
            for item in data.get("flags", []):
                if not isinstance(item, dict):
                    continue
                lvl = str(item.get("level", "")).strip().lower()
                if "yellow" not in lvl:
                    continue
                sub = _normalize_subcategory_name(item.get("subcategory", ""))
                if not sub:
                    continue
                yellow_counts[sub] = yellow_counts.get(sub, 0) + 1

            total_failure = _has_total_failure_signal(data)

            total_score = 0
            new_averages = {}
            
            for cat, weight in weights.items():
                if cat in scoring and isinstance(scoring[cat], list):
                    ratings = []
                    for entry in scoring[cat]:
                        if not isinstance(entry, dict):
                            continue
                        sub = _normalize_subcategory_name(entry.get("subcategory", ""))
                        ic = issue_counts.get(sub, 0)
                        yc = yellow_counts.get(sub, 0)
                        new_rating = _deduced_rating(ic, yc, total_failure=total_failure)
                        entry["rating"] = new_rating
                        entry["reason"] = f"Auto-rated from findings: {ic} improvement(s), {yc} yellow flag(s)."
                        ratings.append(float(new_rating))

                    avg = sum(ratings) / len(ratings) if ratings else 0
                    new_averages[cat] = round(avg, 1)
                    total_score += (avg / 5) * 100 * weight
            
            if "averages" not in scoring: scoring["averages"] = {}
            scoring["averages"].update(new_averages)
            scoring["final_weighted_score"] = round(total_score, 1)
            return json.dumps(data, indent=2), data
        return json_text, {}
    except Exception as e:
        print(f"[WARNING] Score recalculation failed: {e}")
        return json_text, {}

def apply_issue_based_calibration(data):
    """
    Post-processing score calibration.
    
    DISABLED: The issue-count ceiling was found to OVERCORRECT scores.
    Statistical analysis showed issue count has ZERO correlation with
    human scores (3 issues spans 86-100, 6 issues spans 91-100).
    59% of flagged issues were phantom/fabricated (upload to dashboard,
    not full-screen, Zoom annotation vs mouse pointer).
    
    Now passes through raw scores unchanged. The prompt-level fixes
    (phantom issue suppression, frames disabled) address the root cause.
    """
    if not isinstance(data, dict):
        return data
    
    scoring = data.get("scoring", {})
    raw_score = scoring.get("final_weighted_score", 0)
    
    if raw_score <= 0:
        return data
    
    # Just log — no ceiling applied
    issues = data.get("areas_for_improvement", [])
    flags = data.get("flags", [])
    print(f"[CALIBRATION] Score={raw_score} (pass-through, {len(issues)} issues, {len(flags)} flags)")
    
    return data


def validate_and_clean_findings(data):
    """
    V19 post-processing: Remove invalid findings that cause inaccuracy.
    Based on human audit of V18 AI-only findings (4 inaccurate out of 23):
    1. Flags with empty descriptions
    2. Findings with non-Arabic evidence (English descriptions instead of quotes)
    3. Duplicate findings (same subcategory + similar text)
    4. Phantom visual issues (full-screen, annotation, camera from frames)
    """
    if not isinstance(data, dict):
        return data
    
    removed_count = 0
    
    # --- Clean flags ---
    flags = data.get("flags", [])
    clean_flags = []
    for flag in flags:
        reason = (flag.get("reason") or "").strip()
        cite = (flag.get("cite") or "").strip()
        subcategory = (flag.get("subcategory") or "").strip()
        
        # Rule 9: Empty flag descriptions
        if not reason or len(reason) < 10:
            print(f"[V19-CLEAN] Removed flag with empty/short description: {subcategory}")
            removed_count += 1
            continue
        
        # V33: Frame repetition rule for flags — visual flags need 5+ frame references
        FRAME_BASED_SUBCATEGORIES_FLAG = {
            "environment", "dress code", "camera quality", "camera",
            "backdrop", "background", "lighting", "screen sharing",
        }
        subcat_lower_flag = subcategory.lower()
        cite_lower_flag = cite.lower()
        is_frame_flag = any(fs in subcat_lower_flag for fs in FRAME_BASED_SUBCATEGORIES_FLAG)
        if not is_frame_flag and "frame" in cite_lower_flag:
            is_frame_flag = True
        if is_frame_flag:
            frame_refs_f = re.findall(r'frame\s*\d+|frame\s+\d+|\bf\d{1,2}\b', cite_lower_flag)
            time_refs_f  = re.findall(r'\d{2}:\d{2}:\d{2}', cite)
            total_refs_f = len(set(frame_refs_f)) + (len(set(time_refs_f)) if not frame_refs_f else 0)
            if total_refs_f < 5 and len(frame_refs_f) < 5:
                print(f"[V33-CLEAN] Removed frame-based flag (only {total_refs_f} frame refs, need 5+): {subcategory}: {cite[:80]}")
                removed_count += 1
                continue
        
        # V19b: Auto-delete "Language Used" flags about Arabic usage (session IS in Arabic)
        if "language used" in subcategory.lower() or "language used" in reason.lower():
            reason_lower = reason.lower()
            if any(kw in reason_lower for kw in ["arabic", "english", "consistency", "informal", "عربي"]):
                print(f"[V19-CLEAN] Removed Language Used flag (Arabic is the session language): {reason[:60]}")
                removed_count += 1
                continue
        
        # Rule 10: Non-Arabic evidence (if cite is all ASCII/English, it's not a real quote)
        if cite and len(cite) > 5:
            arabic_chars = sum(1 for c in cite if '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F')
            if arabic_chars == 0 and not any(ts in cite for ts in ['00:', '01:', '02:']):
                print(f"[V19-CLEAN] Removed flag with non-Arabic evidence: {subcategory}: {cite[:50]}")
                removed_count += 1
                continue
        elif not cite:
            print(f"[V19-CLEAN] Removed flag with no evidence: {subcategory}")
            removed_count += 1
            continue
        
        clean_flags.append(flag)
    
    data["flags"] = clean_flags
    
    # --- Clean areas_for_improvement ---
    improvements = data.get("areas_for_improvement", [])
    clean_improvements = []
    seen_subcategories = {}  # Track subcategory -> count for dedup
    
    # Known phantom issues to auto-delete (universal, any subcategory)
    # Sources: 524 human-reviewed sessions from iSchool Dashboard V2
    phantom_patterns = [
        # Fullscreen / slideshow  (92% FP rate, 44/48 inaccurate)
        "full-screen", "fullscreen", "browser tabs", "not in slideshow",
        "slideshow mode", "slide show mode",
        # Annotation tools — 100% FP rate in V37 audit (6/6 inaccurate), fully removed as deduction
        "mouse pointer instead of",
        "zoom annotation", "annotation tools", "use annotation", "using annotation",
        # iSchool virtual background from frames  (80% FP rate, 24/30 inaccurate)
        "virtual background", "ischool.*background", "ischool logo.*backdrop",
        "official.*ischool.*background", "iSchool.*virtual",
        # Student-side issues  (83% FP rate, 5/6 inaccurate)
        "student.*software.*not.*installed", "required.*software.*student",
        # Lesson planning / roadmap  (77% FP rate, 59/77 inaccurate)
        "no clear roadmap", "no roadmap", "learning objectives.*not communicated",
        "roadmap.*not.*communicated", "did not.*communicate.*objective",
    ]
    
    # V26: Subcategory-specific phantom patterns (FP items humans don't penalize)
    # Format: (subcategory_pattern, text_pattern) — both are regexes applied to lowercase
    subcat_specific_phantoms = [
        # Environment: mirrored/flipped logo is a Zoom cosmetic issue
        (r"environment", r"(mirror|flip|reverse)"),
        # Environment: posture is not penalized by human reviewers
        (r"environment", r"(posture|upright)"),
        # Environment: eye contact / secondary screen is normal for online tutoring
        (r"environment", r"(focus.*front|secondary.*screen|eye.?contact|shifting.*gaze)"),
        # Camera Quality: eye contact / looking at secondary screen is normal
        (r"camera", r"(focus.*front|secondary.*screen|eye.?contact|shifting.*gaze)"),
        # Microphone: mic on before student joins is pre-session, not a violation
        (r"microphone", r"(mute|muted).*until.*(student|session)"),
        # Internet Quality: student-side or minor (55% FP rate)
        (r"internet", r"(student.*internet|internet.*student|student.*connection|connection.*student|student.*side)"),
        # Voice Tone: single ambiguous impatient moment (67% FP rate)
        # Only keep if text has MULTIPLE timestamps — single instance = delete
        (r"voice.?tone|tone.*clarity", r"(slightly|brief|momentarily|at one point|once|hint of)"),
        # Friendliness: called wrong name once — not reportable (58% FP rate)
        (r"friendliness", r"(wrong name|incorrect name|mispronounce|wrong.*name|name.*wrong).*(?!repeated|again|multiple|persist)"),
        # Tools: annotation tools — 100% FP rate in V37 audit, delete ALL annotation preference flags
        # Zoom annotation vs mouse pointer is a style choice, NOT a deduction
        (r"tools.*(methodology|method)|methodology.*tools", r"(annotation|mouse.?pointer|zoom.?annotation)"),
        # Language Used: informal Arabic expressions (66% FP rate)
        (r"language.?used", r"(informal|casual|slang|colloquial|mixed.*arabic|natural.*arabic|everyday)"),
        # Lesson Planning: roadmap given verbally (77% FP rate)
        (r"lesson.?planning|material.?readiness|resource.?planning", r"(roadmap|objectives|agenda|plan)"),
        # Session Pacing: generic time management (36% FP rate — convert to advice not delete)
        # Handled in advice_patterns below
    ]
    
    # Language Used patterns that are ALWAYS wrong (session is in Arabic)
    language_used_false_patterns = [
        # Original patterns
        "consistency in using english", "use of arabic", "arabic connect",
        "arabic casual", "professional atmosphere",
        "mixing arabic", "ensure full consistency", "english throughout",
        # New from 524-session data (66% FP rate for Language Used informal)
        "informal language", "informal expression", "informal.*tone",
        "casual expression", "casual.*language", "casual.*tone",
        "slang", "colloquial", "everyday arabic", "natural arabic",
        "informal.*arabic", "arabic.*informal",
    ]
    
    for item in improvements:
        text = (item.get("text") or "").strip()
        cite = (item.get("cite") or "").strip()
        subcategory = (item.get("subcategory") or "").strip()
        
        # V33: Frame repetition rule — frame-based visual issues require 5+ frames
        # Visual subcategories that rely on frame evidence
        FRAME_BASED_SUBCATEGORIES = {
            "environment", "dress code", "camera quality", "camera",
            "backdrop", "background", "lighting", "screen sharing",
            "face visibility",
        }
        subcat_lower = subcategory.lower()
        text_lower = text.lower()
        cite_lower = cite.lower()
        is_frame_based = any(fs in subcat_lower for fs in FRAME_BASED_SUBCATEGORIES)
        # Also detect frame-based by cite mentioning "frame"
        if not is_frame_based and "frame" in cite_lower:
            is_frame_based = True
        if is_frame_based:
            # Count how many distinct frame references appear in cite
            frame_refs = re.findall(r'frame\s*\d+|frame\s+\d+|\bf\d{1,2}\b', cite_lower)
            # Also count timestamp refs like 00:xx:xx as secondary support
            time_refs = re.findall(r'\d{2}:\d{2}:\d{2}', cite)
            total_refs = len(set(frame_refs)) + (len(set(time_refs)) if not frame_refs else 0)
            if total_refs < 5 and len(frame_refs) < 5:
                print(f"[V33-CLEAN] Removed frame-based issue (only {total_refs} frame refs, need 5+): {subcategory}: {cite[:80]}")
                removed_count += 1
                continue
        if "language used" in subcat_lower or "language used" in text_lower:
            if any(pat in text_lower for pat in language_used_false_patterns):
                print(f"[V19-CLEAN] Removed Language Used finding (Arabic is session language): {text[:80]}")
                removed_count += 1
                continue
        
        # Check for phantom issues
        is_phantom = False
        for pattern in phantom_patterns:
            if re.search(pattern, text_lower):
                print(f"[V19-CLEAN] Removed phantom issue: {text[:60]}")
                removed_count += 1
                is_phantom = True
                break
        if is_phantom:
            continue
        
        # V26: Check subcategory-specific phantom patterns
        is_subcat_phantom = False
        for subcat_pat, text_pat in subcat_specific_phantoms:
            if re.search(subcat_pat, subcat_lower) and re.search(text_pat, text_lower):
                print(f"[V26-CLEAN] Removed subcat-specific phantom ({subcategory}): {text[:60]}")
                removed_count += 1
                is_subcat_phantom = True
                break
        if is_subcat_phantom:
            continue
        
        # V26/V27: Auto-convert generic advice items to [Advice] prefix (don't deduct)
        advice_patterns = [
            r"mention.*ischool", r"ischool.*brand", r"brand.*loyalty",
            r"mention.*brand", r"cultivat.*brand",
            # V27: Generic advice the AI gives to nearly every session
            r"extensive step-by-step",               # Project Implementation: too much hand-holding
            r"rely more on their own",               # Project Implementation: student independence
            r"student to take.*(lead|initiative)",    # Project Implementation: student independence variant
            r"not enough to just check.*homework",    # Homework: review methodology
            r"just check.*homework",                  # Homework: review methodology variant
            r"walk.*through.*homework",               # Homework: review methodology variant
            # V33: New from 524-session audit data
            r"session.?pacing|poor.?pacing|rushed.*content|rushed.*delivery",  # 36% FP
            r"time management.*lesson|lesson.*time management",  # pacing
            r"(partial|not fully|left incomplete).*project",     # 48% FP — partial project
            r"project.*not.*fully.*(complet|implement)",         # partial project variant
            r"homework.*explained.*verbally",                    # 49% FP — verbal homework
            r"homework.*not.*visual",                            # homework verification
            r"tutor.*(took over|took control|controlled).*(screen|implementation)",  # 54% FP
            r"tutor.?led.*implementation",                       # tutor-led coding
            r"iSchool.*at.*outset", r"outset.*iSchool",          # branding at start
        ]
        if not text.startswith("[Advice]"):
            for ap in advice_patterns:
                if re.search(ap, text_lower):
                    item["text"] = "[Advice] " + text
                    print(f"[V26-ADVICE] Converted to advice: {text[:60]}")
                    break
        
        # Rule 10: Non-Arabic evidence — only remove for phantom-like English descriptions
        # V21 FIX: Removed aggressive English-only filter that was killing valid findings.
        # Many valid findings have English evidence (timestamps, technical terms, code refs).
        # Only remove if cite is purely generic English with no specifics at all.
        
        # Rule 13: Dedup - max 4 findings per subcategory (raised from 2 to allow multi-issue subcats)
        subcat_key = subcategory.lower().strip()
        seen_subcategories[subcat_key] = seen_subcategories.get(subcat_key, 0) + 1
        if seen_subcategories[subcat_key] > 4:
            print(f"[V19-CLEAN] Removed duplicate (5th+) finding for: {subcategory}")
            removed_count += 1
            continue
        
        clean_improvements.append(item)
    
    data["areas_for_improvement"] = clean_improvements
    
    if removed_count > 0:
        print(f"[V19-CLEAN] Removed {removed_count} invalid/phantom findings total")
        # Recalculate score after cleaning
        data_json = json.dumps(data, ensure_ascii=False)
        recalc_json, recalc_data = recalculate_score(data_json)
        if recalc_data:
            data = recalc_data
    
    return data


def _normalize_subcategory_name(name):
    """Normalize subcategory labels for stable cross-step comparisons."""
    raw = (name or "").strip().lower()
    aliases = {
        "knowledge about subject": "Knowledge About Subject",
        "knowledge about subject ": "Knowledge About Subject",
        "tools used and methodology": "Tools used and Methodology",
        "class management": "Class Management",
        "student engagement": "Student Engagement",
        "session synchronization": "Session Synchronization",
        "session study": "Session study",
        "session initiation": "Session Initiation & Closure",
        "session initiation & closure": "Session Initiation & Closure",
        "project software and slides": "Project software & slides",
        "project software & slides": "Project software & slides",
        "knowledge about subject.": "Knowledge About Subject",
    }
    return aliases.get(raw, (name or "").strip())


def _extract_issue_subcategories(report_data):
    """Return normalized subcategory list from areas_for_improvement + flags."""
    if not isinstance(report_data, dict):
        return []

    out = []
    for item in report_data.get("areas_for_improvement", []):
        if isinstance(item, dict):
            sub = _normalize_subcategory_name(item.get("subcategory", ""))
            if sub:
                out.append(sub)

    for item in report_data.get("flags", []):
        if isinstance(item, dict):
            sub = _normalize_subcategory_name(item.get("subcategory", ""))
            if sub:
                out.append(sub)

    return out


def _safe_json_load(text):
    """Best-effort JSON parse for model outputs."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _build_step_flow_snapshot(step_name, report_data, score, priority_targets):
    """Build a compact diagnostics object for one analysis step."""
    issue_subcats = _extract_issue_subcategories(report_data)
    issue_set = set(issue_subcats)
    priority_hits = [s for s in priority_targets if s in issue_set]
    missing_priority = [s for s in priority_targets if s not in issue_set]

    return {
        "step": step_name,
        "score": score,
        "issue_count": len(report_data.get("areas_for_improvement", [])) if isinstance(report_data, dict) else 0,
        "flag_count": len(report_data.get("flags", [])) if isinstance(report_data, dict) else 0,
        "issue_subcategories": sorted(list(issue_set)),
        "priority_hits": priority_hits,
        "priority_missing": missing_priority,
    }


def perform_rag_analysis(video_path, output_report_path, transcript_path=None):
    frames_dir = None
    uploaded_files = []
    try:
        # 1. SETUP & EXTRACTION
        if transcript_path is None:
            transcript_path = TRANSCRIPT_PATH
        start_time = DEFAULT_START_TIME
        if os.path.exists(transcript_path):
            start_time = get_start_time_from_transcript(transcript_path)

        # Audit data flow container (Step 0 -> Step 3)
        flow_data = {
            "session": {
                "video_path": video_path,
                "transcript_path": transcript_path,
                "start_time": start_time,
            },
            "priority_subcategory_targets": list(PRIORITY_SUBCATEGORY_TARGETS),
            "step0": {},
            "step1": {},
            "step2": {},
            "step3": {},
        }

        # Validate source recording integrity before heavy processing
        video_duration_sec = get_video_duration(video_path)
        video_integrity = validate_video_input(video_path, video_duration_sec)
        flow_data["session"]["video_duration_sec"] = video_duration_sec
        flow_data["session"]["video_duration_min"] = round(video_duration_sec / 60.0, 2) if video_duration_sec > 0 else 0
        flow_data["session"]["video_integrity"] = video_integrity
            
        # V19: Frames RE-ENABLED with strict anti-phantom rules
        frames_dir = extract_resources(video_path, start_time)
        print(f"[INFO] Frame extraction enabled — frames + audio + transcript")

        # Extract Audio
        audio_path = "temp_audio.mp3"
        extracted_audio = extract_audio(video_path, audio_path)
        if not extracted_audio:
            raise RuntimeError("Audio extraction failed. The input recording may be invalid or corrupt.")
        audio_integrity = validate_audio_file(extracted_audio)
        flow_data["session"]["audio_integrity"] = audio_integrity

        # 2. UPLOAD SESSION FILES + INITIALIZE PDF KNOWLEDGE BASE
        # PDFs go into a persistent File Search Store (indexed once, reused across sessions)
        # Session files (audio, transcript, frames) are uploaded directly per-session
        
        if FILE_SEARCH_STORE_DISPLAY_NAME:
            print("\n--- Initializing PDF Knowledge Base (File Search Store) ---")
            file_search_store_name = get_or_create_file_search_store(PDF_REFERENCE_FILES)
        else:
            print("\n--- File Search Store Disabled: uploading all 4 PDFs every run ---")
            file_search_store_name = None
        
        print("\n--- Uploading Session Resources ---")
        files_to_upload = []
        
        if os.path.exists(audio_path):
            files_to_upload.append((audio_path, "audio/mp3"))
        
        # V38: PDF reference files are no longer uploaded — knowledge base is injected as JSON text in the prompt
        if os.path.exists(transcript_path):
            files_to_upload.append((transcript_path, "text/plain"))

        # V19: Frames RE-ENABLED with strict anti-phantom rules.
        # Upload up to 30 frames (sampled evenly) for visual context.
        if frames_dir and os.path.exists(frames_dir):
            all_frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
            # Select up to 30 frames evenly spaced
            target_count = min(30, len(all_frames))
            if target_count > 0 and len(all_frames) > target_count:
                step = len(all_frames) / target_count
                selected_frames = [all_frames[int(i * step)] for i in range(target_count)]
            else:
                selected_frames = all_frames[:target_count]
            for frame in selected_frames:
                files_to_upload.append((frame, "image/jpeg"))
            print(f"[FRAMES] Uploading {len(selected_frames)}/{len(all_frames)} frames")
            
        uploaded_files = upload_files_parallel(files_to_upload)
        
        # Categorize resources - sort by original path for deterministic order
        # uploaded_files is now a list of (file_object, original_path) tuples
        pdf_objs = sorted([t for t in uploaded_files if "pdf" in t[0].mime_type], key=lambda t: t[1])
        transcript_objs = sorted([t for t in uploaded_files if "text" in t[0].mime_type], key=lambda t: t[1])
        frame_objs = sorted([t for t in uploaded_files if "image" in t[0].mime_type], key=lambda t: t[1])
        audio_objs = sorted([t for t in uploaded_files if "audio" in t[0].mime_type], key=lambda t: t[1])
        
        print(f"Resources: {len(pdf_objs)} PDFs{' (via File Search Store)' if file_search_store_name and len(pdf_objs) == 0 else ''}, {len(transcript_objs)} Transcripts, {len(frame_objs)} Frames, {len(audio_objs)} Audio")
        wait_for_files_active(uploaded_files)

        # 3. INITIALIZE MODEL (Optimized for Gemini 3 Flash)
        print("\n--- Initializing Knowledge Base Chat ---")
        
        # Load expanded rules from vector DB
        rules_context = ""
        rules_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vector_db_simple", "rules.json")
        if os.path.exists(rules_json_path):
            with open(rules_json_path, "r", encoding="utf-8") as rf:
                loaded_rules = json.load(rf)
            # Build compact rules context for the prompt
            rules_lines = []
            rules_lines.append("\n**APPROVED SUBCATEGORIES AND RULES REFERENCE:**")
            rules_lines.append("Use ONLY these approved subcategory names in your comments:")
            current_cat = None
            for rule in loaded_rules:
                cat = rule.get("category", "")
                if cat == "Scoring":
                    continue  # Scoring anchors handled separately
                if cat != current_cat:
                    current_cat = cat
                    cat_letter = {"Setup": "S", "Attitude": "A", "Preparation": "P", "Curriculum": "C", "Teaching": "T", "Feedback": "F"}.get(cat, "?")
                    rules_lines.append(f"\n### {cat} ({cat_letter}):")
                subcat = rule.get("subcategory", "")
                severity = rule.get("severity", "low")
                deduction = rule.get("deduction", "-1")
                rules_lines.append(f"- **{subcat}** [{severity.upper()}] (deduction: {deduction}): {rule['rule']}")
                if rule.get("standard_comment"):
                    rules_lines.append(f"  Standard comment: \"{rule['standard_comment']}\"")
                if rule.get("scoring_note"):
                    rules_lines.append(f"  ⚠️ Scoring note: {rule['scoring_note']}")
            # Add official deduction system
            rules_lines.append("\n**OFFICIAL DEDUCTION SYSTEM (from Quality Guide):")
            rules_lines.append("- First-time comment (non-flag): -1 point")
            rules_lines.append("- First-time Yellow Flag: -1 point")
            rules_lines.append("- Red Flag (previous yellow): -1 (comment) + -1 (red flag) = -2 total")
            rules_lines.append("- Repeated Red Flag: -2 (repeated comment) + -1 (red flag) = -3 total")
            rules_lines.append("NOTE: Since you are reviewing a SINGLE session, treat all issues as first-time (-1 each).")
            rules_context = "\n".join(rules_lines)
            print(f"Loaded {len(loaded_rules)} rules from vector DB")
        else:
            print(f"WARNING: rules.json not found at {rules_json_path}")
        
        # Load audit calibration data from 495 human-audited sessions
        audit_calibration_context = ""
        audit_cal_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vector_db_simple", "audit_calibration.json")
        if os.path.exists(audit_cal_path):
            with open(audit_cal_path, "r", encoding="utf-8") as af:
                audit_cal = json.load(af)
            
            cal_lines = []
            cal_lines.append("\n**CALIBRATION GUIDANCE:**")
            cal_lines.append("**100% ACCURACY RULE: Only report issues you are fully certain about. If you have any doubt, omit the issue. A false positive harms the tutor unfairly.**")
            cal_lines.append("**Human reviewers typically find 3-8 issues per session, but ONLY report what they can clearly verify.**")
            cal_lines.append("**Your job is to find all REAL issues — certain, verifiable ones — not to hit a target count.**\n")
            
            # V33: Build high-FP table from the 524-session data
            # This directly tells the model WHICH patterns to suppress
            cal_lines.append("**⚠️ HIGH FALSE-POSITIVE PATTERNS — DO NOT REPORT THESE (backed by 524 human-audited sessions):**")
            cal_lines.append("The following patterns were marked INACCURATE by human auditors at high rates.")
            cal_lines.append("If your finding matches any of these, DELETE it before outputting:\n")
            fp_patterns = audit_cal.get("false_positive_patterns", [])
            for fp in fp_patterns:
                pct = fp.get("fp_rate_pct", 0)
                if pct >= 50:  # Only show the truly high-FP ones to avoid overwhelming the model
                    cal_lines.append(f"❌ [{pct}% INACCURATE] {fp['pattern']}")
                    cal_lines.append(f"   → {fp['instruction']}")
            
            cal_lines.append("\n**PATTERNS REQUIRING VERY STRONG EVIDENCE (borderline FP):**")
            for fp in fp_patterns:
                pct = fp.get("fp_rate_pct", 0)
                if 36 <= pct < 50:
                    cal_lines.append(f"⚠️ [{pct}% INACCURATE] {fp['pattern']}")
                    cal_lines.append(f"   → {fp['instruction']}")
            
            cal_lines.append("\n**KNOWN PHANTOM PATTERNS (always delete):**")
            cal_lines.append("1. 'Presentation not in full-screen / slideshow mode' — DELETE.")
            cal_lines.append("2. 'Mouse pointer instead of Zoom annotation tools' — DELETE (unless NEVER used in entire session).")
            cal_lines.append("3. 'iSchool virtual background not used' — DELETE (frame detection unreliable).")
            cal_lines.append("4. 'No roadmap/learning objectives communicated' — DELETE (verbal roadmaps are acceptable).")
            
            audit_calibration_context = "\n".join(cal_lines)
            print(f"Loaded audit calibration (524-session FP patterns)")
        else:
            print(f"NOTE: audit_calibration.json not found at {audit_cal_path} — using hardcoded calibration")
        
        system_instr = """You are the **Lead Quality Auditor** and **Senior Quality Compliance Auditor** for iSchool.
Your task is to conduct a **Forensic Quality Review** of online coding sessions.

**AUDIT PROTOCOL:**
1. **INGEST RULES:** The iSchool Quality Reference Knowledge Base (embedded as JSON at the start of the prompt) is your AUTHORITATIVE source of truth:
   - **quality_guide** — The master rulebook: scoring criteria, category definitions, deduction rules
   - **quality_comments** — Approved comment templates and standard phrasing
   - **flag_examples** — Yellow/Red flag definitions and real examples
   - **comments_bank** — Reference library of approved reviewer comments
   You MUST consult the embedded knowledge base when evaluating EVERY finding. If a finding contradicts the knowledge base, the knowledge base wins.
2. **ANALYZE SESSION:** Cross-reference Audio (MP3), Transcript (TXT), and SESSION FRAMES against the rules.
    - listen for MP3 from uploaded file between tutor and student.
    - read zoom transcript from uploaded file.
    - **SESSION FRAMES ARE PROVIDED** as visual snapshots from the Zoom recording. Use them to verify VISUAL/SETUP checks: face visibility, camera position, backdrop, dress code, screen sharing, environment. Frame evidence SUPPORTS but does not replace audio evidence for non-visual issues.

    **⚠️ CRITICAL: ZOOM FRAME ATTRIBUTION RULE ⚠️**
    In Zoom recordings, multiple participants are visible simultaneously as camera tiles:
    - The **TUTOR's tile** is typically the LARGE/main view or the largest camera tile — this is the ONLY tile you may evaluate for tutor issues.
    - **Student tiles** appear as SMALL thumbnail boxes (corners, edges, or gallery strips). Any issue visible ONLY in a small student tile is the STUDENT's problem — NOT the tutor's.
    - **BEFORE reporting ANY frame-based issue** (camera angle, backdrop, lighting, dress code, background), you MUST confirm that the issue is in the TUTOR's tile, not a student's tile.
    - **If you cannot clearly determine whose tile shows the problem → DO NOT flag it.**
    - Common false positives to avoid: 'student's blurry camera flagged as tutor camera issue', 'student's messy background flagged as tutor backdrop issue', 'student's informal dress flagged as tutor dress code issue'.
    - The TUTOR is the person whose name matches the session tutor ID. If unsure who is the tutor vs student in a frame, rely on the audio only.

    **⚠️ FRAME REPETITION RULE — MANDATORY ⚠️**
    A visual issue seen in only 1 or 2 frames is NOT reportable — it may be a momentary glitch, angle change, or framing artifact.
    - **MINIMUM THRESHOLD: A frame-based issue MUST be visible and consistent across AT LEAST 5 separate frames before it can be reported.**
    - If the issue appears in fewer than 5 frames → **DO NOT report it.** It is not a persistent problem.
    - This applies to: dress code, backdrop/environment, camera angle, lighting, face visibility, screen sharing state, background issues.
    - In your `cite` field for any frame-based issue, you MUST list the frame numbers/timestamps where the issue was observed (e.g. "Frame 3, 7, 11, 15, 22"). If you cannot list 5+ frames → DELETE the finding.
    - Example: If backdrop is only wrong in Frame 1 but correct in Frames 2–30 → NOT a reportable issue.

    **⚠️ CRITICAL: ARABIC TRANSCRIPT QUALITY WARNING ⚠️**
    The Zoom transcript is AUTO-GENERATED and its Arabic transcription is EXTREMELY UNRELIABLE:
    - Arabic words are frequently GARBLED, MISRECOGNIZED, or NONSENSICAL (e.g., "أغبرهم" instead of "أخبارهم", random words substituted for what was actually said).
    - The transcript text in Arabic CANNOT be trusted for understanding WHAT was said.
    - The transcript IS useful for: **timestamps**, **speaker identification** (who spoke when), and **detecting silence gaps** ONLY.
    - **YOU MUST LISTEN TO THE AUDIO (MP3) to understand the actual session content.** Gemini can understand Arabic speech directly — use YOUR OWN Arabic listening comprehension, NOT the garbled Zoom transcript text.
    - **EVIDENCE RULE: NEVER COPY TEXT FROM THE ZOOM TRANSCRIPT into the "cite" field.** The Zoom Arabic text is garbled garbage. Instead, LISTEN to the audio and WRITE YOUR OWN accurate Arabic transcription of what was actually said. Your Arabic hearing comprehension is FAR more accurate than Zoom's Arabic auto-transcription.
    - **EXAMPLE:** Zoom transcript says: "سلام عليكم... عبد الله... الأخبار أغبرهم النهاردة" ← This is GARBLED. Listen to the audio — the tutor actually said: "السلام عليكم يا عبدالله، إيه الأخبار؟ هنبدأ النهاردة...". Quote YOUR accurate version, not Zoom's broken version.
    - If you cannot clearly hear what was said at a timestamp, DO NOT fabricate or guess — skip that evidence.

    **AUDIO VS. TRANSCRIPT RULE (AUDIO IS PRIMARY):**
    You possess both the Recording (Audio) and the Script (Transcript).
- The **Audio is your PRIMARY source** — listen to it to understand what tutor and student actually said and did.
- The Transcript is for **Timestamping** and **Speaker Turns** only — do NOT rely on its Arabic text for content understanding.
- IF the transcript says "(silence)" but the audio contains keyboard clicking -> IT IS NOT SILENCE.
- IF the transcript looks polite but the audio sounds angry -> TRUST THE AUDIO.
- IF the transcript text seems nonsensical in Arabic -> it's a transcription error, LISTEN to the audio instead.
- You must prioritize Audio evidence for ALL findings, not just Attitude.
    **CRITICAL: PARTIAL TRANSCRIPT WARNING:**
    Zoom transcripts may capture ONLY A PORTION of the session (e.g., the last 2-5 minutes).
    - NEVER determine session duration from transcript timestamps alone.
    - The VIDEO (frames) and AUDIO (mp3) represent the FULL session. Use them to verify actual duration.
    - If the transcript spans only a few minutes but the video/audio is clearly ~60 minutes, the transcript is PARTIAL — NOT evidence of a short session.
    - Before flagging "short session duration", verify: How many frames were extracted? Does the audio file cover ~60 minutes? If YES, the session was full-length regardless of transcript coverage.
    - A partial transcript is still useful for timestamping events, but it does NOT represent the entire session content.
3. **STRICT GUIDELINES:**
    - List ALL issues you find with clear, specific evidence.
    - Minor issues without clear evidence should be OMITTED, not listed.
    - Prioritize issues that directly affect student learning outcomes.
    - Reference the provided PDFs for rule citations.
    - Do NOT cap your issue count. If a session has 8+ real issues, report ALL of them.
""" + rules_context + """

**ISSUE DETECTION GUIDELINES (CRITICAL — READ CAREFULLY):**
- **FIND ALL REAL ISSUES.** Human reviewers typically find 3-8 issues per session. Finding fewer than 3 should be EXTREMELY RARE.
- **100% ACCURACY RULE:** Every issue you report MUST be 100% certain and directly verifiable in the audio, transcript, or frames. Do NOT include borderline, uncertain, or inferred issues. If you are not fully certain an issue is real, DO NOT include it. It is better to report fewer issues accurately than to report uncertain ones.
- **QUALITY OVER QUANTITY:** A false issue that a human reviewer would delete hurts calibration more than a missed issue. Only include issues you would stake your reputation on.
- Do NOT flag student-side problems (student internet, student camera, student device, student backdrop, student dress) as tutor issues.
- **FRAME ATTRIBUTION (ZOOM RECORDING):** Before flagging ANY visual issue from a frame, confirm it is in the TUTOR's large tile — NOT a small student thumbnail. If a backdrop, camera angle, or dress issue is only in a small corner tile, it belongs to the student. DO NOT flag it.
- "Language Used" about speaking Arabic → SKIP (Arabic IS the session language; only flag profanity/insults).
- Known phantom patterns to skip: "Full-screen/slideshow mode" (code editors are correct), "Mouse pointer vs annotation tool".
- Frames ARE valid evidence for visual/setup checks (camera, face, backdrop, dress code) — but ONLY for the TUTOR's tile, not student tiles. Audio is primary for all other issues.

**HIGH-FP EVIDENCE THRESHOLDS (from 524 human-audited sessions — MANDATORY):**
These subcategories have HIGH false-positive rates. Apply strict evidence rules:
- **Frame-based issues (ANY visual finding)**: MUST appear in 5+ separate frames. If you cannot cite 5 frame numbers showing the same problem → DELETE the finding. One or two frames is never enough.
- **Internet Quality** (55% FP): Only report if TUTOR'S OWN connection caused 3+ documented disruptions. Never blame tutor for student-side lag.
- **Voice Tone / Harsh Tone** (67% FP): Require 3+ timestamps of harsh/impatient tone. A single moment of firmness is NORMAL — do NOT report it.
- **Language Used / Informal Arabic** (66% FP): Informal Arabic IS the normal tutoring register. Only flag actual profanity, explicit insults, or offensive language.
- **Friendliness / Wrong Name** (58% FP): Only flag if tutor kept using wrong name AFTER the student corrected them.
- **Virtual Background / iSchool Logo** (80% FP): Do NOT flag from frames — Zoom background detection is unreliable.
- **Lesson Planning / No Roadmap** (77% FP): Verbal roadmap is acceptable. Only flag if tutor had completely wrong session content.
- **Project Implementation (tutor-led/step-by-step)** (54% FP): Report as [Advice] only — never deduct.
- **Annotation Tools** (60% FP): Only flag if tutor NEVER used annotation tools across the entire session.
- **Fullscreen/Slideshow mode** (92% FP): NEVER report — always a phantom.

**EVIDENCE RULES:**
- Every flag MUST have a non-empty "reason" and a "cite" field with evidence.
- No duplicates: max 2 findings per subcategory. Merge if same behavior at same timestamp.
- Teaching style issues (e.g., step-by-step guidance) belong in areas_for_improvement, NOT flags.

**SCORING CALIBRATION (MANDATORY — CRITICAL FOR ACCURACY):**

**MANDATORY: EVALUATE ALL 19 SUBCATEGORIES:**
You MUST rate EVERY subcategory listed below. Do NOT skip any. Each category has multiple subcategories that must each receive an independent rating:
- **Setup (S):** Environment, Internet Quality, Camera Quality, Microphone Quality, Dress Code
- **Attitude (A):** Friendliness, Language Used, Session Initiation & Closure, Voice Tone & Clarity
- **Preparation (P):** Knowledge About Subject, Project software & slides, Session study
- **Curriculum (C):** Homework, Slides and project completion, Tools used and Methodology
- **Teaching (T):** Class Management, Project Implementation & Activities, Session Synchronization, Student Engagement

The category average is the MEAN of all subcategory ratings within that category. With more subcategories evaluated, individual deductions have the correct mathematical impact on the final score.

**Scoring guidelines (per subcategory — WHOLE NUMBERS ONLY: 5, 4, 3, 2, 1, 0):**
- **5/5 = PERFECT:** No issues found in this subcategory. This is the DEFAULT when you have no specific negative observation.
- **4/5 = GOOD:** 1-4 evidence-backed issues found. The subcategory has some areas for improvement.
- **3/5 = FAIR:** 5-6 "Areas for Improvement" from this subcategory. Multiple problems.
- **2/5 = WEAK:** 3+ Yellow Flags from this subcategory OR 6+ "Areas for Improvement". Significant concerns.
- **1/5 = CRITICAL:** 10+ "Areas for Improvement" from this subcategory. Major systemic problems.
- **0/5 = ZERO:** No-show or total failure.

**CRITICAL RATING RULES:**
1. When you identify improvement areas for a subcategory, the rating MUST be **4 or lower** (NOT 5).
2. The DEFAULT rating for a subcategory with ZERO issues is **5**, NOT 4.
3. Use WHOLE NUMBERS ONLY (5, 4, 3, 2, 1, 0). Do NOT use half-points like 4.5, 3.5.
4. Count issues PER SUBCATEGORY to determine each rating.

**MANDATORY SCORE-ISSUE CONSISTENCY CHECK (Calibrated from Real Human Data):**
Each improvement area reduces the affected subcategory from 5 to 4 (first issue) or lower:
- If you list 0 improvement areas → all subcategories at 5 → score should be 100
- If you list 1-2 improvement areas in 1-2 subcategories → score should be 96-100
- If you list 3-4 improvement areas in 2-3 subcategories → score should be 92-96
- If you list 5-6 improvement areas in 3-5 subcategories → score should be 84-92
- If you list 7-8 improvement areas in 5+ subcategories → score should be 76-88
- If you list 9+ improvement areas → score should be below 80
- **A score of 100 is ONLY valid if there are ZERO improvement areas.**
- **CRITICAL: Count issues PER SUBCATEGORY and apply the rating table above.**

**SCORE ANCHORING (Real human data):**
- Average human score across all sessions: ~93
- Sessions with 0 issues: 97-100 (rare — only ~5% of sessions)
- Sessions with 1-2 minor issues: 92-97
- Sessions with 3-4 issues: 84-92
- Sessions with 5-6 issues (common): 76-84
- Sessions with 7+ issues: below 76
- CRITICAL: Most sessions have at LEAST 3-5 real improvement areas. Finding fewer than 3 should be RARE.
- A session scoring 80 typically has 5 issues; a session scoring 86 typically has 3-4 issues.

**EVIDENCE-BASED ASSESSMENT:**
- When evidence clearly shows an issue with a specific timestamp and quote: REPORT IT.
- When evidence is ambiguous or unclear: DO NOT include it.
- Report ALL real issues you find — do NOT cap at 3-5 issues. If a session has 10 issues, report all 10.
- Do NOT force-find issues without evidence. But equally, do NOT suppress real issues to make a session seem better.

    - Use exact Category keys: S(Setup), A(Attitude), P(Preparation), C(Curriculum), T(Teaching), F(Feedback).
    - Results must be mathematically verified using the weighted formula.

**HUMAN REVIEWER CALIBRATION REFERENCE (MANDATORY):**
Study these REAL human reviewer patterns. Your output MUST match this style and scoring logic.

**How Human Reviewers Write Comments:**
- Format: "[Category Letter] - [Constructive comment]. [Timestamp if applicable]"
- Tone: Professional, constructive, uses "Kindly..." or "Please..." phrasing
- Each comment starts with the category letter (S, A, P, T, C, F) followed by a dash
- Timestamps are included in brackets like [00:33:40] or [01:10:54]
- Comments are specific, actionable, and reference exact moments in the session

**Standard Human Reviewer Positive Comment Templates:**
- S - The tutor maintained a stable internet connection throughout the session, ensuring smooth delivery with minimal disruptions.
- S - The tutor's setup was solid. Clear camera and microphone quality supported effective communication, while the tidy background and appropriate dress code reinforced a professional overall appearance.
- S - The tutor used the updated iSchool Zoom background, maintaining a polished and professional environment.
- A - The tutor greeted the student warmly and created a welcoming atmosphere.
- A - The tutor's clear and calm voice helped the student follow along easily and stay focused throughout the session.
- A - The tutor consistently uses professional language and chooses words carefully.
- P - Well-prepared throughout the session, with effective use of tools and solid subject knowledge that supports the learning process.
- P - The tutor came well-prepared, smoothly guiding the student through the lesson with clear explanations and no signs of confusion.
- C - The tutor explained all the slides smoothly, helping the student stay aligned with the session structure and objectives.
- C - The tutor effectively used Zoom's annotations to visualize key concepts.
- T - Early joining by the tutor before the scheduled session time is appreciated and noted.
- T - Your commitment to joining 10 minutes before the scheduled time is noted and greatly elevates professionalism.
- F - The tutor delivered feedback in a constructive and uplifting manner, fostering a supportive learning environment.
- F - The tutor provided constructive and positive feedback that was well-delivered and encouraging for the student.

**Standard Human Reviewer Improvement Comment Templates:**
- S - Please make sure to use the iSchool logo as your Zoom background to maintain a consistent and professional setup.
- S - The camera angle was not properly set. Adjusting the camera to eye level can enhance visual interaction.
- S - Kindly adjust the light source to ensure your entire face is clearly visible without strong shadows.
- S - Occasional internet lags disrupted the session flow and made it harder for the student to stay engaged.
- S - Kindly ensure that your microphone is muted until the student joins the session.
- T - Kindly engage the student during the checkup by encouraging their participation and involvement, rather than re-explaining the concepts.
- T - Kindly avoid spending long stretches explaining without involving the student. The session would benefit from asking more open-ended questions.
- T - It may be helpful to divide the code into sections followed by the student's implementation of each section.
- T - Please be advised that students must upload their projects to the dashboard during the session.
- T - Efficiently manage the session's timing; if the session ends early, utilize the remaining time by providing extra programming activities.
- C - Kindly note that it is not allowed for students to take a screenshot of the task, as all required material is provided on the student dashboard.
- C - Encouraging the student to think and respond during the Q&A slides, rather than reading the answers aloud.
- F - Kindly vary the feedback provided to the same student across sessions. Repeating the same comment limits their ability to grow.

**REAL HUMAN REVIEW EXAMPLES (Score Calibration):**
Example 1 — Score 100 (1 minor improvement area):
  Positive: S-stable connection, A-clear voice, P-well-prepared, C-slides explained smoothly, T-early joining, F-constructive feedback
  Improvement: "S - Kindly ensure that your microphone is muted until the student joins the session. [00:14:30]"
  → Human scored 100 because the single issue was extremely minor (pre-session mic management).

Example 2 — Score 99 (1 improvement area):
  Positive: S-solid setup, A-warm welcoming, P-strong preparation, C-well-paced project completion, T-early joining, F-constructive feedback
  Improvement: "C - Kindly note that it is not allowed for students to take a screenshot of the task. [01:10:54]"
  → Human scored 99 because the single issue was a minor policy reminder.

Example 3 — Score 96 (4 improvement areas):
  Positive: A-used student name warmly, P-followed approved slides, C-used annotations effectively, T-early joining, F-uplifting feedback
  Improvement: "S - Camera angle too high", "S - Adjust light source", "T - Student engagement could be improved during project explanations", "T - Students must upload projects to dashboard"
  → Human scored 96 because the 4 issues were all minor/procedural (camera, lighting, engagement tips).

Example 4 — Score 95 (4 improvement areas):
  Positive: S-good microphone, P-well-prepared, C-slides explained smoothly, F-constructive feedback
  Improvement: "S - Use iSchool Zoom background", "T - Dashboard accidentally displayed [00:33:23]", "T - Project descriptions need to be more descriptive"
  → Human scored 95. Multiple issues but none critical.

Example 5 — Score 93 (6 improvement areas including engagement):
  Positive: S-high-quality camera, S-stable connection, A-professional language, P-well-prepared, T-early joining
  Improvement: "T - Engage student during checkup [00:33:40]", "T - Avoid long stretches explaining without involving student", "T - Divide code into sections for student implementation", "T - Extensive step-by-step guidance at [01:01:40] — encourage independent problem-solving", "C - Encourage student to think during Q&A slides [01:20:00]", "F - Include reasons for ratings below 5"
  → Human scored 93. Many teaching engagement issues but none flagged as violations.

Example 6 — Score 70 (CRITICAL — genuine bad session):
  Multiple serious issues: attitude problems, unprofessional language, preparation failures, 
  systematic teaching methodology failures, incomplete curriculum delivery.
  → This score is RARE and requires MULTIPLE serious/systemic failures across categories.
  → Do NOT give scores below 80 unless you observe truly severe, evidence-backed issues across 3+ categories.

**CRITICAL SCORING INSIGHT FROM HUMAN DATA:**
- Human reviewers set subcategory ratings to 5 (perfect) by default
- The overall score is calculated from subcategory ratings using the weighted formula
- 0 issues = score 100 (all subcategories at 5)
- 1-2 issues in 1-2 subcategories = score 96-100
- 3-4 issues across 2-3 subcategories = score 92-96
- 5-6 issues across 3-5 subcategories = score 88-92
- 7+ issues across 5+ subcategories = score 80-88

**REPORT EVERYTHING WITH EVIDENCE:**
- If you have specific evidence (timestamp + quote or clear observation) for a tutor-side issue → REPORT IT.
- Do NOT suppress real issues. Report what you observe.
- When in doubt, INCLUDE the issue. Under-detection is the bigger risk.
- Most sessions have 3-8 real improvement areas. If you found fewer than 3, look harder.
- Frames are valid for visual/setup checks. Audio is primary for all other issues.

**OUTCOME:** Return ONLY a valid JSON object matching the required schema."""

        # Model configuration with seed for determinism
        # Gemini 3.0 Flash with thinking_level for controlled reasoning
        gen_config_kwargs = dict(
            temperature=MODEL_TEMPERATURE,
            candidate_count=1,
            response_mime_type="application/json",
            system_instruction=system_instr,
            seed=args.seed if hasattr(args, 'seed') else DEFAULT_SEED,
            media_resolution=args.media_resolution if hasattr(args, 'media_resolution') else DEFAULT_MEDIA_RESOLUTION,
        )
        
        # Apply Thinking Config — level-based control for Gemini 3
        # thinking_level controls internal reasoning depth (replaces thinking_budget in Gemini 3)
        thinking_level = getattr(args, 'thinking_level', DEFAULT_THINKING_LEVEL)
        if thinking_level:
            gen_config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=thinking_level
            )
            print(f"--- Thinking enabled (level={thinking_level}) ---")
        
        # Enable Google Search (Community Search) if requested
        # NOTE: File Search and Google Search CANNOT be combined (Gemini API limitation)
        if hasattr(args, 'use_google_search') and args.use_google_search:
            if file_search_store_name:
                print("--- WARNING: Google Search disabled — cannot combine with File Search ---")
            else:
                print("--- Google Search (Community Search) ENABLED ---")
                gen_config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
        
        # Enable File Search tool for PDF knowledge retrieval
        if file_search_store_name:
            print(f"--- File Search ENABLED (store: {file_search_store_name}) ---")
            gen_config_kwargs["tools"] = [
                types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=[file_search_store_name]
                    )
                )
            ]
            
        if args.max_output_tokens is not None:
            gen_config_kwargs["max_output_tokens"] = args.max_output_tokens
        else:
            gen_config_kwargs["max_output_tokens"] = DEFAULT_MAX_OUTPUT_TOKENS  # Ensure sufficient output budget

        generation_config = types.GenerateContentConfig(**gen_config_kwargs)
        
        # Compute session metadata for the prompt
        # video_duration_sec already computed during media validation above
        num_frames = len([f for f in os.listdir(frames_dir) if f.endswith('.jpg')]) if frames_dir and os.path.exists(frames_dir) else 0
        video_duration_min = video_duration_sec / 60 if video_duration_sec > 0 else 0

        # V38: Load JSON knowledge base (replaces PDF file uploads for reference docs)
        reference_kb_text = load_reference_knowledge_base()

        combined_prompt = f"""
Analyze the session files provided (Transcript, Audio, Frames).
The iSchool Quality Reference Knowledge Base is embedded below as structured JSON — use it to look up the exact rules, deduction criteria, and approved comment templates for EVERY finding you report.
Generate a comprehensive Quality Audit Report in JSON format.

**YOUR PRIMARY TASK:** Check whether the tutor violated ANY rule in the Quality Standards below. For each potential violation:
1. Check the embedded Knowledge Base (quality_guide, quality_comments, flag_examples, comments_bank) for the relevant rule
2. Verify the tutor's behavior against that specific rule
3. Only flag it if the tutor clearly violated the guideline with evidence

**ISSUE MATCHING PRIORITY (CRITICAL):**
To improve match with human reviews, you MUST explicitly check these frequently-missed subcategories:
- Session study
- Knowledge About Subject
- Tools used and Methodology
- Class Management
- Student Engagement
- Session Synchronization
If no issue is found in one of these, briefly justify why in `_reasoning_trace`.
Do NOT output a near-perfect score without completing this targeted check.

{reference_kb_text}

**CRITICAL: SESSION METADATA (Use this to verify session duration)**
- Video file duration: {video_duration_min:.0f} minutes ({video_duration_sec:.0f} seconds)
- The transcript may only cover a PORTION of this. Trust the video/audio duration over transcript timestamps for session length.
- **{num_frames} FRAMES are provided** as visual context for the session.

**STRICT FRAME EVIDENCE RULES (MANDATORY — prevents phantom issues):**
Frames provide VISUAL CONTEXT but are NOT primary evidence. Follow these rules:
1. **DO NOT flag display mode issues**: "not full-screen", "browser tabs visible", "code editor layout" are NORMAL. Delete any such finding.
2. **DO NOT flag annotation preferences**: "mouse pointer instead of Zoom annotation" is acceptable. Zoom annotation tools are a style preference — **NEVER flag as a deduction** even if used throughout the entire session. Only if a student was REPEATEDLY confused by verbal directions with no visual aid (3+ explicit confusion moments).
3. **VISUAL-ONLY checks ALLOWED from frames**: You MAY flag these based on frames alone WITHOUT audio confirmation:
   - **Face visibility**: Tutor's face not visible / camera off throughout session
   - **Camera position**: Camera pointing at ceiling/wall instead of face
   - **Backdrop/environment**: Clearly unprofessional background (bed, messy room)
   - **Dress code**: Tutor not dressed appropriately (pajamas, no shirt visible)
   - **Screen sharing**: Not sharing screen when teaching
   For all OTHER observations (tone, teaching quality, engagement), audio evidence is STILL required.
4. **Frame evidence supports audio for non-visual issues**: For teaching methodology, engagement, or behavior claims, you MUST also hear it confirmed in the audio. Frame-only claims about non-visual topics are DELETED.
5. **VALID frame uses**: Verifying tutor is sharing screen, confirming slide content is correct, checking if student is actively coding, confirming the tutor has a professional background, checking face visibility, verifying dress code compliance, checking camera angle and environment.

**CRITICAL: SESSION START TIME**
The session officially starts at timestamp **{start_time}**.
**IGNORE** all audio, text, or visual events before **{start_time}**.
Any "silence" or "waiting" before {start_time} is PRE-SESSION WAITING time and is NOT a violation.

**CRITICAL: TUTOR JOINING TIME DETECTION (MANDATORY CALCULATION)**
The tutor is required to join the session at least **15 minutes early** and wait (muted) for the student.
The audio recording starts when the tutor joins the meeting room. The transcript start time ({start_time}) is when the first meaningful speech/interaction begins.

**HOW TO DETECT TUTOR WAITING TIME:**
- **Waiting Time = Transcript Start Time - 00:00:00** (the gap of silence/mute at the beginning of the audio)
- Compare the audio file duration with the transcript start time:
  * Audio duration = {video_duration_min:.0f} minutes, Transcript starts at {start_time}
  * If {start_time} is >= 00:15:00 (15+ minutes into the recording) → The tutor joined early and waited. This is POSITIVE. Note it as: "T - Early joining by the tutor before the scheduled session time is appreciated."
  * If {start_time} is between 00:10:00 and 00:15:00 → The tutor joined within the acceptable joining window. This is fine — do NOT flag or advise.
  * If {start_time} is between 00:05:00 and 00:10:00 → Tutor joined within the minimum acceptable window. Do NOT flag.
  * If {start_time} is **< 00:05:00** (less than 5 minutes of waiting) → The tutor likely joined late or just on time without sufficient waiting. **Actively search** the audio and transcript for any signs of late joining:
    - Does the tutor apologize for being late? ("sorry I'm late", "آسف على التأخير")
    - Does the student mention they were waiting? ("كنت مستنيك", "you're late")
    - Does the session start immediately with no waiting period?
    - Is the audio recording very short at the start before speech begins?
    If ANY of these signals are present, OR if the start_time is **< 00:02:00** (virtually no waiting time), log:
    **T - Class Management: Tutor does not appear to have joined 15 minutes early as required. Waiting time before session start was only ~{start_time}. [00:00:00]**
- **IMPORTANT:** The student joining late is NOT the tutor's fault. If the waiting time is long (15+ min), this actually means the tutor WAS on time and the student was late.
- **DO NOT flag 'tutor joined late' based on the transcript start timestamp alone**.
- **V38 RULE: If start_time < 00:02:00 (essentially no pre-session waiting), this is a strong signal the tutor joined late or just barely on time. Always flag this unless you have explicit audio evidence the tutor had been waiting.**

**REQUIRED SCHEMA:**
{{
  "_reasoning_trace": ["Step 1: Analyzed setup...", "Step 2: Found issue X at timestamp...", "Step 3: Verified rule Y..."],
  "meta": {{"tutor_id": "str", "group_id": "str", "session_date": "str", "session_summary": "str"}},
  "positive_feedback": [{{"category": "str", "subcategory": "str", "text": "str", "cite": "str", "timestamp": "str"}}],
  "areas_for_improvement": [{{"category": "str", "subcategory": "str", "text": "str", "cite": "str", "timestamp": "str"}}],
  "flags": [{{"level": "Yellow/Red", "subcategory": "str", "reason": "str", "cite": "str", "timestamp": "str"}}],
  "scoring": {{
    "setup": [
      {{"subcategory": "Environment", "rating": 0, "reason": "str"}},
      {{"subcategory": "Internet Quality", "rating": 0, "reason": "str"}},
      {{"subcategory": "Camera Quality", "rating": 0, "reason": "str"}},
      {{"subcategory": "Microphone Quality", "rating": 0, "reason": "str"}},
      {{"subcategory": "Dress Code", "rating": 0, "reason": "str"}}
    ],
    "attitude": [
      {{"subcategory": "Friendliness", "rating": 0, "reason": "str"}},
      {{"subcategory": "Language Used", "rating": 0, "reason": "str"}},
      {{"subcategory": "Session Initiation & Closure", "rating": 0, "reason": "str"}},
      {{"subcategory": "Voice Tone & Clarity", "rating": 0, "reason": "str"}}
    ],
    "preparation": [
      {{"subcategory": "Knowledge About Subject", "rating": 0, "reason": "str"}},
      {{"subcategory": "Project software & slides", "rating": 0, "reason": "str"}},
      {{"subcategory": "Session study", "rating": 0, "reason": "str"}}
    ],
    "curriculum": [
      {{"subcategory": "Homework", "rating": 0, "reason": "str"}},
      {{"subcategory": "Slides and project completion", "rating": 0, "reason": "str"}},
      {{"subcategory": "Tools used and Methodology", "rating": 0, "reason": "str"}}
    ],
    "teaching": [
      {{"subcategory": "Class Management", "rating": 0, "reason": "str"}},
      {{"subcategory": "Project Implementation & Activities", "rating": 0, "reason": "str"}},
      {{"subcategory": "Session Synchronization", "rating": 0, "reason": "str"}},
      {{"subcategory": "Student Engagement", "rating": 0, "reason": "str"}}
    ],
    "averages": {{"setup": 0, "attitude": 0, "preparation": 0, "curriculum": 0, "teaching": 0}},
    "final_weighted_score": 0
  }},
  "action_plan": ["string", "string", "string"]
}}

**CRITICAL: The scoring section MUST include ALL 19 subcategories listed above (5+4+3+3+4=19). Do NOT omit any. Each subcategory gets its own independent rating.**

**RULES:**corroborating frame+audio evidenc
- **CHAIN OF THOUGHT:** You MUST populate the `_reasoning_trace` array FIRST with your step-by-step analysis. This is your "scratchpad" to ensure accuracy.
- Category Keys: **S** (Setup), **A** (Attitude), **P** (Preparation), **C** (Curriculum), **T** (Teaching), **F** (Feedback).
- Scoring Logic: 5 (Perfect, default=no issues) down to 0 (No-show). Whole numbers only (5/4/3/2/1/0). Apply weighted formula: (Setup 25%, Attitude 20%, Prep 15%, Curr 15%, Teach 25%).
- Math Verification: Re-calculate category averages and sum them based on weights before outputting the final score.
- **POSITIVE FEEDBACK:** You MUST include at least **3** specific positive observations in the `positive_feedback` array.
- language English or Arabic do not put language used in area of improvement.
- if the find with not storng and clraer evidence, do not include it.
- **WISE JUDGMENT:** Only report issues that ACTUALLY HAPPENED and have REAL EVIDENCE — a specific timestamp, an exact quote, or a clear frame. Do NOT report issues based on inference or assumptions.
- **IMPACT THRESHOLD:** Ask yourself: "Would a fair, experienced human reviewer note this?" If you have evidence and a reviewer would note it, INCLUDE it.
- **CONFIDENCE GATE:** For each potential issue, you must have specific evidence (timestamp + quote or frame). If you have evidence, INCLUDE the issue — do not self-censor.
- **REPORT ALL REAL ISSUES:** Do NOT cap your findings at 3-5 issues. If you find 8-12 real issues with evidence, report ALL of them. Suppressing real issues leads to over-scoring.
- **STUDENT-SIDE vs TUTOR-SIDE (CRITICAL — READ CAREFULLY):** If an issue originates from the STUDENT's device, internet, software, or behavior (e.g., student can't find an app, student's screen is laggy, student joins late, STUDENT's internet is cutting out), do NOT penalize the TUTOR for it. Only deduct for things within the TUTOR's control.
  * **LATE JOINING:** If the session starts after the scheduled time, do NOT automatically assume the TUTOR was late. The STUDENT may have joined late while the tutor was already waiting. Only flag tutor lateness if the tutor explicitly apologizes for being late or the student explicitly says they were waiting.
  * **WHO IS THE SUBJECT? (MANDATORY CHECK):** Before flagging ANY Setup issue, ask: "Is this about the TUTOR's own setup, or is the tutor INSTRUCTING THE STUDENT about the student's setup?" If the tutor is telling the student to fix their camera/mic/internet, that is POSITIVE coaching behavior — NOT a tutor issue.
  * **Arabic direction clues (tutor talking TO the student):**
    - "ممكن نزل الكاميرا شوية" = "Can you lower the camera a bit" → Tutor coaching STUDENT about student's camera = POSITIVE, NOT a flag.
    - "ارفع/نزل الكاميرا" = "raise/lower the camera" → Instruction TO student = NOT a tutor issue.
    - "النت عندك بيقطع" = "your internet is cutting" → Said TO student = STUDENT's problem, NOT tutor's.
    - "ممكن نقفل الكاميرا بس ثواني" = brief camera-off for internet = NORMAL.
    - Any imperative/request form ("ممكن ت...", "try to...", "can you...") directed at the student = instruction, not tutor fault.
  * **INTERNET:** Only flag TUTOR-SIDE internet disconnections lasting MORE THAN 1 MINUTE. Brief lags/freezes (<1 min) are NORMAL. Student-side internet issues are NEVER the tutor's fault.
  * **CAMERA OFF FOR INTERNET:** Brief camera-off (<1 min) to stabilize connection (tutor or student side) is STANDARD PRACTICE — NOT a flag.
- **HOLISTIC CATEGORY RULE:** One-off cosmetic slips (e.g., mouse pointer once instead of annotation) should be advice, not deductions. But recurring or impactful issues MUST be deducted even if the category is generally strong.
- **TUTOR ENFORCING RULES = POSITIVE:** If the tutor tells the student to follow iSchool policies (e.g., "don't take screenshots of slides", "use the desktop app", "turn on your camera", "use the iSchool background"), this is CORRECT behavior and should be in `positive_feedback`, NOT `areas_for_improvement`.
- **PRE-SESSION ACTIVITY IS NOT AN ISSUE:** Anything that happens BEFORE the start time ({start_time}) — mic being on, tutor adjusting settings, waiting — is NOT a violation. Only flag issues that occur DURING the session.
- **REMOTE CONTROL / BRIEF ASSISTANCE:** A tutor briefly taking control to help a stuck student (e.g., navigating a file, fixing a path) is NORMAL teaching behavior, not a violation. Only flag remote control if the tutor implements the ENTIRE project without student involvement.
- **HOMEWORK REVIEW PHASE (FIRST 25 MIN):** If the timestamp is **before 00:25:00** and the context looks like the tutor is reviewing the student's previous homework, going over their previous code, or demonstrating a correction on existing student work — this is **NORMAL homework review, NOT "remote control" or "step-by-step implementation"**. Do NOT flag Project Implementation issues for timestamps before 00:25:00 unless there is very clear evidence the tutor is implementing NEW project work from scratch.

---

### **PHASE 1: THE AUDIT PROTOCOL (Relaxed Enforcement)**
Check the session against these specific criteria. If a violation is found, it **MUST** be listed in "areas_for_improvement" or "flags".
Note the exact timestamp from the transcript where the issue occurs.

**IMPORTANT: 1-HOUR SESSION CONTEXT**
Do NOT report the following as issues:
- Brief moments of silence (under 2 min) while student is coding/thinking.
- Tutor briefly checking slides or materials (under 30 seconds).
- One or two instances of minor audio lag that don't disrupt flow.
- The session could be 1 hour and 30 minutes with the waiting time for the student.
 + Frames as visual context)**
**NOTE: Frames provide visual context. Audio and transcript remain PRIMARY evidence. Any frame-only finding without audio/transcript support must be DELETED
You MUST calculate the actual teaching duration:
- Find the timestamp when the student joins and meaningful interaction begins.
- Find the timestamp when the session ends (farewell/goodbye).
- Actual Duration = End Time - Start Time.
- **CRITICAL: CROSS-CHECK WITH AUDIO BEFORE FLAGGING:**
  - Check the audio file duration. If the audio is ~60+ minutes, the session is full-length.
  - The video duration metadata is provided above — use it to verify actual session length.
  - If the transcript covers only a few minutes but the audio/video is ~60 min, the TRANSCRIPT IS PARTIAL — the session is NOT short.
  - ONLY flag short duration if BOTH the transcript AND the audio/video duration confirm a short session.
- If the teaching duration is truly **under 50 minutes** (confirmed by video+audio), this is a violation: flag as **T - Class Management: Session duration was only X minutes (required: 60 min +/-10)**.
- If the teaching duration is **over 80 minutes** without justification, note it as advice. Sessions up to 80 minutes (1 hour + 20 min buffer) are acceptable without comment.
- Include the calculated duration AND your cross-check evidence in _reasoning_trace.
**1. AUDITORY COMPLIANCE (Check Transcript + Audio)**
**NOTE: Frames are provided for VISUAL/SETUP checks (face visibility, camera, backdrop, dress code). For non-visual findings, evidence must come from audio and transcript.**
*   **Dead Air (Silence):** Flag if silence exceeds **6 minutes** without tutor engagement or check-in. 
*   **Rapport & Warmth — DEEP TONE ANALYSIS (MANDATORY):**
    Listen to the AUDIO carefully at THREE points in the session:
    a) **Session START (first 5 minutes):** Does the tutor greet warmlyvisible in frames AND audio tone is cold), log:
      **A - Friendliness: Consider engaging in more friendly and encouraging interactions to build rapport.**
      (NOTE: Do NOT flag based on frames alone — must also have audio evidence of cold/flat tone.)rushed?
    c) **Session END (last 5 minutes):** Does the tutor close warmly with praise and encouragement? Or abruptly end?
    * DO NOT assume a greeting in the transcript = warm delivery. The TEXT may say "How are you?" but the AUDIO tone may be cold/rushed.
    * If the audio delivery sounds cold, flat, or mechanical despite polite words, log:
      **A - Friendliness: Session delivery lacked warmth and enthusiasm. Text was polite but tone was cold/mechanical.**
    * If the session opening is cold/abrupt with NO greeting or acknowledgement of the student at all, log:
      **A - Session Initiation: Session started abruptly without warm-up or friendly conversation to ease the student in.**
    * NOTE: Absence of a formal "ice-breaker" game or activity is NOT an issue. A normal warm greeting suffices.
    * If the tutor does not smile or show positive facial expressions (check frames), log:
      **A - Friendliness: Consider engaging in more friendly and encouraging interactions to build rapport.**
*   **Language:** Arabic is ALLOWED for communication. Do NOT flag Arabic usage as "Language Used" issue.
    * Arabic is the natural teaching language for these sessions. Casual Arabic greetings, connectors ("يعني", "طيب", "خلاص"), and transitions are NORMAL — do NOT flag.
    * Only flag "A - Language Used" if the tutor uses genuinely inappropriate, unprofessional, or vulgar language. Simple informality or mixing Arabic/English is NOT a violation.
    * **DEFAULT: Do NOT include any Language Used findings.** The threshold is very high — practically only profanity or insults qualify.
*   **STUDENT INAPPROPRIATE LANGUAGE (MANDATORY EVIDENCE REQUIREMENT):**
    * If a STUDENT says something inappropriate (e.g., swear word) during the session, this only becomes a Class Management issue if there is **clear evidence the tutor HEARD it and chose NOT to address it**.
    * If the tutor showed NO reaction to the student's language (no correction, continued normally, likely didn't hear it) → **DO NOT FLAG** as Class Management, or at most add as [Advice].
    * Evidence required to flag: tutor audibly reacted, the conversation paused, or the transcript shows the tutor's mic registered the inappropriate word clearly and the tutor continued without response.
    * **If unclear whether tutor heard it → DO NOT FLAG.**
*   **Internet & Audio Stability (MANDATORY CHECK — TUTOR-SIDE ONLY, >1 MINUTE):**
    - **CRITICAL: DISTINGUISH TUTOR vs STUDENT internet issues:**
      * If the STUDENT has internet problems (student's camera freezes, student disconnects, student asks to turn off camera due to their internet) → this is NOT the tutor's fault → DO NOT FLAG.
      * Arabic clue: "النت عندك بيقطع" (your internet is cutting) said TO the student = STUDENT has the problem, NOT the tutor.
      * Arabic clue: "ممكن نقفل الكاميرا بس ثواني كده لحد ما نشوف الإنترنت" = turning off camera briefly to stabilize internet = NORMAL, NOT a flag.
      * ONLY flag internet issues that are clearly FROM THE TUTOR'S SIDE.
    - **THRESHOLD: Tutor-side internet disconnection must last MORE THAN 1 MINUTE to be flagged.**
      * Brief lags, momentary freezes, or short disconnections (under 1 minute) are NORMAL in online sessions → DO NOT FLAG.
      * Only flag if tutor-side disconnection is >1 minute AND happens 3+ times, log:
        **S - Internet Quality: Tutor-side internet disconnections disrupted session flow (>1 min each, occurred X times)**.
    - **CAMERA OFF DUE TO INTERNET = NOT A FLAG:**
      * If the tutor or student briefly turns off the camera (<1 minute) to help stabilize a poor internet connection, this is STANDARD PRACTICE in online teaching.
      * DO NOT issue a Camera Quality flag for brief camera-off due to internet stabilization.
      * Only flag camera-off if the tutor deliberately keeps the camera off for >1 minute WITHOUT a connectivity reason.

**3. PREPARATION QUALITY (MANDATORY CHECK):**
*   **Materials Readiness:**
    * If slides are disorganized, contain typos, or seem hastily prepared, log:
      **P - Material Readiness: Materials were not well-organized or prepared in advance**.
    * If the tutor lacks resources/materials for the session topic, log:
      **P - Resource Planning: Required materials or setup were missing or not prepared**.
*   **Lesson Planning & Roadmap:**
    * If the tutor does not provide a clear lesson plan, roadmap, or learning objectives, log:
      **P - Lesson Planning: No clear lesson plan or learning objectives communicated**.
    * If the pacing is erratic (rushing parts, spending too long on others), log:
      **P - Session Pacing: Poor time management and pacing of lesson content**.
*   **Technical Setup Verification:**
    * If the tutor failed to test tools/platforms before the session or has technical issues that could have been prevented, log:
      **P - Technical Preparation: Technical tools or platforms were not tested/verified beforehand**.

**4. TEACHING QUALITY (MANDATORY DEEP CHECK):**
*   **Teaching Mode Balance:**
    * If teaching is mostly one-directional (explaining without questioning) for **>5 continuous minutes** outside of exempt phases, log:
      **T - Student Engagement: Session was largely lecture-based with limited student interaction**.
    * **EXEMPT PHASES (do NOT flag for one-directional talk during):**
      - Session checkup / review phase (first ~15-20 min): tutor re-explaining or reviewing homework = NORMAL.
      - Brief concept explanation (≤5 min): introducing a new concept verbally = NORMAL.
      - Student implementation silence: when the student is actively coding/typing, the tutor may be monitoring silently = NOT a violation.
*   **Guided Thinking vs Direct Answers:**
    * If the tutor provides full solutions, syntax, or logic immediately without prompting the student to think or respond first, log:
      **T - Teaching Methodology: Over-reliance on direct answers instead of guided discovery**.
*   **Check-for-Understanding (MANDATORY):**
    * If the tutor explains a concept without asking the student to confirm understanding, log:
      **T - Student Engagement: Lack of comprehension-check questions during explanation**.
*   **Student Independence During Project (CRITICAL):**
    * During the project implementation phase (usually after ~50 min), the STUDENT should be doing the coding/building.
    * If the tutor is giving EXTENSIVE step-by-step instructions (dictating every line) instead of letting the student try first, log:
      **[Advice] T - Project Implementation & Activities: Extensive step-by-step guidance was given. Kindly offer limited support and encourage the student to think critically and solve problems independently.**
    * Look for these indicators: student repeatedly asks "what do I do next?", tutor dictates code line by line, student NEVER attempts anything independently for >20 continuous minutes.
    * **NOTE:** This is ALWAYS an [Advice] item — NEVER a score deduction. Normal teaching support is NOT a violation. Only flag if truly excessive and persistent throughout the session.
    * **Repeated guidance** at multiple timestamps is a stronger signal — cite the timestamps.
    * **HOMEWORK REVIEW EXEMPTION:** If the guidance/dictation occurs **before 00:25:00**, it is very likely the tutor is reviewing or correcting the student's previous homework — this is NORMAL and should NOT be flagged as "step-by-step guidance". Check for homework-related keywords in the transcript (homework, واجب, المشروع القديم, review) to confirm.
*   **Code Sectioning:**
    * If the tutor explains a large block of code all at once without breaking it into sections for the student to implement piece by piece, log:
      **T - Teaching Methodology: It may be helpful to divide the code into sections followed by the student's implementation of each section.**

**4.5. FULL-SESSION ENGAGEMENT BALANCE (MANDATORY — Analyze Entire Session):**
Do NOT evaluate engagement from a single moment. Analyze the ENTIRE session in three 20-minute segments:
- **Segment 1 (0-20 min):** How much does the student speak vs tutor? Is the review/checkup interactive or lecture-only?
- **Segment 2 (20-40 min):** Are there comprehension checks? Does the student answer questions or just listen?
- **Segment 3 (40-60 min):** Is the student actively coding/building, or just watching the tutor?

**EXEMPT PHASES — Do NOT count as engagement issues:**
- **Session checkup / homework review phase (Segment 1, first 15-20 min):** The tutor re-explaining/reviewing homework is NORMAL. Even extended tutor talk during this phase is NOT an engagement problem.
- **Concept explanation (any time, ≤5 min):** A focused 3-5 min explanation of a new concept is expected — NOT a violation. Only flag if it exceeds 5 minutes with ZERO student interaction.
- **Student implementation quiet (Segment 3):** When the student is actively coding, tutor silence ≤3 min is NORMAL monitoring behavior.

Calculate an approximate **Student Talk Ratio**: What % of the session involves the student speaking/participating?
- If Student Talk Ratio is **under 20%** ACROSS the full session (excluding Segment 1 checkup and exempt concept explanations), the session is too one-directional. Log:
  **T - Student Engagement: Session was heavily tutor-led. Student participation was limited to less than ~20% of session time. Increase interactive moments and open-ended questions.**
- If the tutor talks for **more than 5 continuous minutes** without the student speaking at any point, AND this is NOT during an exempt phase (checkup/concept explanation), log:
  **T - Student Engagement: Extended tutor monologue detected (>5 min without student interaction). Break explanations into smaller segments with questions.**

Include your Segment 1/2/3 engagement assessment in the `_reasoning_trace`.

**4.55. BEHAVIORAL DETECTION — MANDATORY MICRO-CHECKS:**
These are specific behavioral patterns that human reviewers frequently flag. Check each one explicitly:

  a) **SILENCE DURING IMPLEMENTATION:** When the student is working on the project/code, does the tutor remain SILENT for extended periods (>5 min)? Or does the tutor check in, encourage, and guide? If the tutor is completely silent for >5 consecutive minutes while the student is coding:
     **T - Student Engagement: Tutor remained mostly silent during student implementation phase. Active engagement, check-ins, and encouragement are required during hands-on work. [timestamp]**
     NOTE: Brief silence (≤5 min) while the student is actively coding/typing is NORMAL — do NOT flag. Only flag if silence is extended AND there is no encouragement or check-in throughout.

  b) **ENCOURAGEMENT & PRAISE:** Does the tutor acknowledge the student's efforts, progress, or correct answers at ANY point? Count how many times the tutor says encouraging things ("great job", "well done", "ممتاز", "برافو", "أحسنت"). If encouragement is ABSENT or very rare (fewer than 2 instances in the whole session):
     **A - Friendliness: The tutor did not sufficiently acknowledge the student's efforts or improvements. Simple encouragement ("great job", "well done") boosts motivation. [timestamp of missed opportunity]**

  c) **MONOTONE DELIVERY CHECK:** Listen at the 30-minute mark. Does the tutor's voice have natural variation in pitch, pace, and energy? Or is it flat and robotic? If monotone:
     **A - Voice Tone & Clarity: Tutor's delivery was monotonous during explanations. Speaking with more energy and variation helps maintain student attention. [timestamp]**

  d) **SESSION OPENING QUALITY:** In the first 2 minutes, does the tutor:
     - Greet the student by name? Ask how they are doing? (warm greeting)
     - OR jump straight to the lesson content without ANY greeting at all?
     If the start is cold/abrupt with absolutely no greeting or acknowledgement:
     **A - Session Initiation & Closure: Session started without any friendly greeting. A simple "How are you doing?" before starting goes a long way. [00:00:00]**
     NOTE: A formal "ice-breaker" activity is NOT required. A normal warm greeting is sufficient. Do NOT flag if the tutor greeted the student normally — only flag if there is literally NO greeting at all.

  e) **DISTRACTION CHECK:** Does the tutor appear distracted (looking at phone, typing on something else, long pauses mid-sentence) while the student is presenting or asking a question?
     **T - Class Management: Tutor appeared distracted while the student was engaged. Focused attention and follow-up are essential. [timestamp]**

  f) **PROJECT COMPLETION CHECK:** Was the project/activity fully completed during the session? If not, was it due to poor time management?
     **C - Slides and project completion: Project was not fully completed during the session due to time management. Ensure activities and challenges are completed in-session. [timestamp]**

  g) **ONE-SIDED SESSION CHECK (CRITICAL):** Is the session largely the tutor talking while the student passively listens? Human reviewers frequently flag sessions where interaction is minimal. Listen for:
     - Does the student respond with ONLY "yes", "ok", "uh-huh" for extended periods?
     - Does the tutor give long explanations (>5 min) without pausing for the student to try or answer?
     - Is there a back-and-forth exchange, or is it a monologue?
     If the session is predominantly one-sided:
     **T - Student Engagement: The session was predominantly one-sided with the tutor delivering content and the student passively listening. Increase interactive moments, ask open-ended questions, and allow the student to attempt tasks independently. [timestamp range]**

  h) **HOMEWORK / PREVIOUS SESSION REVIEW CHECK:** At the start of the session, does the tutor:
     - Ask about or review the student's homework from the previous session?
     - Check if the student practiced or completed any assigned tasks?
     If homework review is completely absent when it should have occurred:
     **C - Homework: No homework review or follow-up from previous session was observed. Reviewing homework reinforces learning and accountability. [00:00:00]**

  i) **TUTOR DISTRACTION / MULTITASKING CHECK:** Listen carefully for signs the tutor is distracted:
     - Long unexplained pauses mid-sentence (not student thinking time)
     - Typing sounds unrelated to the teaching content
     - Tutor asking student to repeat because they weren't paying attention
     - Delayed responses when student asks a question
     If the tutor appears distracted:
     **T - Class Management: Tutor appeared distracted or multitasking during the session. Focused attention on the student is essential for quality teaching. [timestamp]**

  j) **SESSION CLOSURE QUALITY CHECK:** In the last 5 minutes, does the tutor:
     - Summarize what was learned?
     - Assign homework or next steps?
     - End with encouragement or positive remarks?
     If the session ends abruptly without proper closure:
     **A - Session Initiation & Closure: Session ended abruptly without a proper summary, homework assignment, or encouraging closure. A structured ending reinforces learning. [end timestamp]**

  k) **STUDENT INDEPENDENCE VS SPOON-FEEDING:** During the project/coding phase, does the student:
     - Type code independently while the tutor guides?
     - OR does the tutor dictate EVERY line/step without letting the student try first?
     If the tutor is spoon-feeding code line by line:
     **T - Project Implementation & Activities: The tutor dictated code/steps to the student without allowing independent attempts. Encourage the student to try first, then provide hints if needed. [timestamp range]**

  l) **TOOL/SOFTWARE USAGE CHECK:** Is the student using the correct version of the required software?
     - Desktop app vs web version (when desktop is required)
     - Correct programming environment for the lesson
     If wrong tool version is used without correction:
     **P - Project software & slides: Student was using the wrong version of the software (e.g., web version instead of desktop app). Tutor should verify and correct this early in the session. [timestamp]**

  m) **FACE VISIBILITY CHECK (USE FRAMES):** Look at the session frames. Is the tutor's face visible throughout?
     - Is the tutor's camera turned ON?
     - Is the face clearly visible (not just a ceiling, dark screen, or blurred image)?
     - If the tutor's camera is OFF or face is not visible for extended periods:
     **S - Camera Quality: Tutor's camera was off or face was not visible during the session. Camera should remain on with face clearly visible to create a personal learning experience. [frame evidence]**

  n) **SOFTWARE/TOOL PREPARATION CHECK:** At the very start of the session (first 2-3 minutes after student joins):
     - Is the required software/IDE already OPEN and ready to use?
     - Or does the tutor spend time opening, installing, or setting up tools AFTER the session starts?
     If the tutor was not prepared (opens software late, installs tools during session):
     **P - Resource Planning: Tutor was not prepared — required software/tools were not pre-opened before the session began. All tools should be ready before the student joins. [timestamp]**

  o) **PRESENTATION/DISPLAY QUALITY CHECK (USE FRAMES):** Look at the session frames during teaching:
     - Is the screen content clearly visible and readable?
     - Is the font size adequate for the student to read?
     - Is the zoom annotation color visible against the background? (e.g., dark annotation on dark background is hard to see)
     If display quality issues are evident:
     **P - Project software & slides: Screen content was difficult to read — annotation color blended with background / font too small / display unclear. Ensure all visual content is easily readable for the student. [frame evidence]**

  p) **DASHBOARD PROJECT UPLOAD — DO NOT FLAG:** Dashboard upload completion is not reliably detectable from video/audio alone and causes too many false positives. Do NOT flag missing dashboard upload under any circumstances.

**4.6. DASHBOARD PROJECT UPLOAD — DO NOT FLAG:**
- Dashboard upload is not reliably verifiable from video/audio alone. Do NOT flag missing dashboard upload.
- This was causing too many false positives where AI flagged upload issues that human reviewers did not flag.

**4.7. PRESENTATION MODE — DO NOT FLAG:**
- Full-screen mode vs windowed mode is cosmetic. Do NOT flag this.
- Browser tabs or taskbar being visible is NOT a deduction.

**4. PROCEDURAL COMPLIANCE (Check Transcript + Audio)**
*   **Opening/Closing:** Roadmap explained? Summary provided? Homework assigned?
*   **Platform:** Correct tool used ?
*   **Tool Language:** If tools appear in Arabic when English is required, log:
      **P - Project software & slides: Tool language not set to English as required**.
*   **Session End-Time Management (MANDATORY CHECK):**
    * If the session ends MORE THAN 10 minutes early (e.g., teaching only 45 min in a 60-min slot),
      check if the remaining time was used for enrichment activities or extra programming practice.
    * If the session ended early WITHOUT enrichment, log:
      **T - Class Management: Session ended [X] minutes early. Please utilize remaining time for extra programming activities or enrichment, not for solving homework.**
    * If remaining time was used to solve homework (instead of enrichment), also flag this.

**5. PROJECT COMPLETION VS EXPLANATION (MANDATORY CHECK):**
*   **FIRST 50 MINUTES EXEMPTION:** During the first 50 minutes of the session, it is ACCEPTABLE for the tutor to implement examples to demonstrate concepts. This is NOT a violation.
*   **AFTER 50 MINUTES:** Verify if project was fully implemented by student, partially, or only explained verbally.
*   If explained only (after 50 min), log:
    **C - Slides and Project Completion: Project was explained but not implemented by the student during the session**.
*   If tutor implements while student observes (after 50 min), log:
    **T - Project Implementation & Activities: Tutor-led implementation limited hands-on student practice**.
*   **Project Demonstration (MANDATORY):**
    * The tutor MUST run/execute the completed project so the student can observe the results and output.
    * If the project was implemented but never executed/demonstrated, log:
      **T - Project Implementation & Activities: Please run the project so the student can observe the results. This experience significantly enhances their understanding of the concepts involved.**
*   **Session Timeline Adherence:**
    * Check if the tutor followed the expected session structure (review → concept → practice → project).
    * If significant timeline deviations were observed (e.g., spending too long on review, rushing through project), log:
      **T - Class Management: Occasional timeline deviations were observed. The session segments could be managed more efficiently for better pacing and time utilization.**

**6. CONCEPT ACCURACY & MISCONCEPTIONS (CRITICAL CHECK):**
*   If the tutor explains a concept incorrectly or in a misleading way, lSOLELY on visual/frame observations without audio/transcript support
    **C - Knowledge About Subject: Concept explained inaccurately or misleadingly**.
*   **Examples include (but are not limited to):**
    - Incorrect definition of widget roles (e.g., layout vs content widgets)
    - Misuse or misinterpretation of properties 
    - Logical inaccuracies 
### **FINAL QUALITY CHECK & RE-SCAN GATE (REQUIRED)**
Before producing the final JSON:
1. Remove any KNOWN FALSE POSITIVE (full-screen mode, annotation tools preference without [Advice]).
2. Remove any finding based SOLELY on visual/frame evidence without audio support.
3. **MINIMUM ISSUE REALITY CHECK:**
   - Count your total areas_for_improvement.
   - Genuine perfect sessions exist (and should score 100 with zero issues), but they are rare.
   - If you have **0-2 issues**, pause and double-check:
     * Are you SURE the session was fully interactive (Student Engagement)?
     * Was the methodology effective, with visual aids when needed (Tools & Methodology)?
     * Did the tutor show absolute confidence (Session study / Knowledge)?
     * Did the session follow a structured flow (Session Synchronization)?
   - **DO NOT hallucinate issues just to meet a quota.** If the session is genuinely excellent, KEEP IT CLEAN. If you missed subtle flaws, add them.
4. Verify evidence quality. For structural/pedagogical issues, session-wide behavioral patterns with timestamps are VALID evidence.
5. Recalculate the score based on validated findings only.

---

### **PHASE 2: JSON GENERATION RULES**

***1. THE "FOCUS" RULE:**
*   List ALL findings backed by STRONG, SPECIFIC evidence (exact quotes or timestamps).
*   Do NOT cap your findings at 3-5. If you find 6-10 real issues with evidence, report ALL of them.
*   Human reviewers typically find 3-8 issues per session. If you find fewer than 3, look more carefully.
*   Minor or marginal findings should be excluded unless they form a clear pattern.
*   Check the transcript for timestamps and listen for the audio to validate the findings.
*   **EVIDENCE LANGUAGE:** The "cite" field should contain evidence supporting the finding:
    - For audio/behavioral issues: an Arabic quote from the audio is ideal. Write what you HEARD, never copy from the garbled Zoom transcript.
    - For structural/pedagogical issues (Session study, Knowledge, Synchronization, Tools, Slides completion, Class Management): a clear description of the observed PATTERN with specific timestamps is valid evidence. These are holistic issues that no single quote can prove.
    - For visual/setup issues: a description of what the frames show is valid.
    - **INVALID:** cite: "سلام عليكم... عبد الله... الأخبار أغبرهم النهاردة" (garbled Zoom auto-transcript — NEVER use)
    - **INVALID:** cite: "The tutor explains abstraction himself" (vague English inference with no timestamp — NOT evidence)


**2. FORMATTING RULE:**
*   For the `"text"` field in feedback lists, use the exact format: `[Category Letter] - [Subcategory]: [Description ] - [Evidence: Specific example with 1-2 timestamps only]`
*   **CRITICAL:** Include ONLY 1-2 representative timestamps per feedback item. DO NOT list 50+ timestamps.
*   **EVIDENCE REQUIREMENT:** ONLY include findings with STRONG, CLEAR, and SPECIFIC evidence. If a finding lacks concrete proof or is based on assumptions, DO NOT include it.
*   **Category Keys:** **S** = Setup, **A** = Attitude, **P** = Preparation, **C** = Curriculum, **T** = Teaching, **F** = Feedback.

---

### **PHASE 3: RIGOROUS SCORING LOGIC**
**You must calculate the score based strictly on the findings from Phase 2. Do not guess.**

**Step A: Determine Sub-Category Ratings (0-5)**
For **EACH** sub-category in the JSON `scoring` object, count the number of "areas_for_improvement" items belonging to that SAME subcategory, then apply:
*   **5 (Perfect):** 0 issues found in this subcategory.
*   **4 (Good):** 1-4 "Areas for Improvement" from this same subcategory.
*   **3 (Fair):** 5-6 "Areas for Improvement" from this same subcategory.
*   **2 (Weak):** 3+ Yellow Flags from this same subcategory OR 6+ "Areas for Improvement" from it.
*   **1 (Critical):** 10+ "Areas for Improvement" from this same subcategory.
*   **0 (Zero):** No show / Total failure.

**IMPORTANT:** Count issues PER SUBCATEGORY, not per category. A subcategory with 0 issues = 5, even if other subcategories in the same category have issues.

**Step B: Apply the Weighted Formula**
Calculate the `final_weighted_score` using these exact weights:
- **Setup (25%):** `(Setup Avg ÷ 5) × 100 × 0.25`
- **Attitude (20%):** `(Attitude Avg ÷ 5) × 100 × 0.20`
- **Preparation (15%):** `(Preparation Avg ÷ 5) × 100 × 0.15`
- **Curriculum (15%):** `(Curriculum Avg ÷ 5) × 100 × 0.15`
- **Teaching (25%):** `(Teaching Avg ÷ 5) × 100 × 0.25`

**Step C: MANDATORY MATH VERIFICATION (SELF-AUDIT)**
Before finalizing the JSON, you MUST internaly verify your calculations and facts:
1) **EVIDENCE CHECK:** Ensure EVERY "area_for_improvement" and "flag" has specific evidence (timestamp/quote). Remove any that do not.
2) **SCORE CHECK:** Re-calculate category averages and valid weights. The final score must match the individual ratings perfectly.
3) **POSITIVE CHECK:** Ensure at least 3 distinct positive highlights are included.
4) **LANGUAGE CHECK:** Arabic IS the session language — DELETE ALL \"Language Used\" findings about Arabic usage, mixing Arabic/English, or \"ensure consistency in using English\". The ONLY valid Language Used finding is profanity or insults. If the finding mentions Arabic in any way, DELETE IT.
5) **EVIDENCE QUALITY CHECK:** Re-read every \"cite\" field. Does it read like natural, grammatical Arabic? Or does it look garbled/nonsensical (copied from Zoom auto-transcript)? If garbled → REWRITE from audio or DELETE the finding.
6) **FORMAT CHECK:** Ensure \"text\" fields are concise and evidence is clear.

**Remove any comment that lacks evidence. and recale the score don't be like a robot**
Hard constraints:
- Return ONLY valid JSON matching the existing schema.
- Do NOT include any camera angle/framing/visibility/camera quality findings.
- Do NOT include generic session feedback items (category F / session feedback).
- Keep all "text" fields concise using the format: `[Category Letter] - [Subcategory]: [Description ] - [Evidence: ...]`
- Comments from the removed Comments Bank PDF should NOT be included.
"""
        # Extract file objects from tuples for content list
        pdf_files = [t[0] for t in pdf_objs]
        transcript_files = [t[0] for t in transcript_objs]
        frame_files = [t[0] for t in frame_objs]
        audio_files = [t[0] for t in audio_objs]

        # ================================================================
        # STEP 0 (MIXED INTO STEP 1): GUIDE-FIRST GROUNDING + SESSION INTAKE
        # No separate model call. We build deterministic preflight context,
        # then pass it directly inside Step 1 prompt.
        # ================================================================
        print("\n--- Step 0+1: Mixed Guide-First Grounding ---")
        response_0 = None

        step0_data = {
            "session_data_validation": {
                "audio_status": "valid" if extracted_audio else "invalid",
                "duration_check": (
                    f"Video duration is {video_duration_sec:.1f}s ({video_duration_min:.1f} min), "
                    "which is consistent with a full session."
                    if video_duration_sec > 0
                    else "Video duration could not be determined."
                ),
                "start_time_check": f"Transcript-derived start time: {start_time}",
                "notes": [
                    "Step 0 is mixed into Step 1 (single-pass grounding).",
                    "Guide-first ordering enforced: PDF rules are sent before session data.",
                ],
            },
            "rule_anchors": [
                "Use quality guide + comments PDFs as the first reference source.",
                "Reject phantom issues (fullscreen/mouse-pointer/dashboard-upload).",
                "Score by per-subcategory issue counts only.",
            ],
            "high_priority_issue_targets": list(PRIORITY_SUBCATEGORY_TARGETS),
            "target_subcategory_map": {
                "Session study": {"what_to_verify": "Preparation depth and confident flow", "evidence_type": "audio|transcript"},
                "Knowledge About Subject": {"what_to_verify": "Technical correctness of explanations", "evidence_type": "audio|transcript"},
                "Tools used and Methodology": {"what_to_verify": "Use of proper teaching tools/method", "evidence_type": "audio|frames|transcript"},
                "Class Management": {"what_to_verify": "Timing, pacing, and focus handling", "evidence_type": "audio|transcript"},
                "Student Engagement": {"what_to_verify": "Interaction ratio and open-ended questioning", "evidence_type": "audio|transcript"},
                "Session Synchronization": {"what_to_verify": "Logical sequence of session phases", "evidence_type": "audio|transcript"},
            },
            "grounding_summary": "Mixed Step 0+1 flow: apply PDF guide first, then audit session evidence.",
        }
        step0_context = json.dumps(step0_data, ensure_ascii=False, indent=2)

        step0_report_path = os.path.splitext(output_report_path)[0] + "_Step0.json"
        with open(step0_report_path, 'w', encoding='utf-8') as f:
            f.write(step0_context)
        print(f"[SUCCESS] Step 0 Intake Saved (mixed with Step 1): {step0_report_path}")

        flow_data["step0"] = {
            "parsed": True,
            "source": "local-preflight (mixed into step1)",
            "session_data_validation": step0_data.get("session_data_validation", {}),
            "rule_anchors": step0_data.get("rule_anchors", []),
            "high_priority_issue_targets": step0_data.get("high_priority_issue_targets", []),
            "target_subcategory_map": step0_data.get("target_subcategory_map", {}),
            "grounding_summary": step0_data.get("grounding_summary", ""),
        }

        if step0_context:
            combined_prompt += f"\n\n**STEP 0 GROUNDED CONTEXT (MANDATORY REFERENCE):**\n{step0_context}\n"

        combined_prompt += """

    **GUIDE-FIRST ORDER (MANDATORY):**
    1) Read and apply PDF Guide rules first.
    2) Then evaluate session evidence (transcript/audio/frames).
    """

        # Enforce a strict audit-flow contract from Step 0 into Step 1.
        flow_priority_lines = "\n".join([f"- {s}" for s in PRIORITY_SUBCATEGORY_TARGETS])
        combined_prompt += f"""

    **STEP 1 FLOW CONTRACT (MANDATORY — DETECTION GATE):**
    Before finalizing Step 1 JSON, run a focused RE-EXAMINATION for these priority subcategories:
    {flow_priority_lines}

    **For EACH priority subcategory, you MUST:**
    1. Re-listen to the audio specifically looking for evidence of problems in that area
    2. Write in `_reasoning_trace` exactly one line:
       `FLOW Step1 | <subcategory> | ISSUE|CLEAN | <evidence summary>`
    3. If CLEAN, you must cite POSITIVE evidence (e.g., "tutor asked 5+ open-ended questions" for Student Engagement)
    4. If you cannot cite positive evidence, the subcategory should be ISSUE

    **DETECTION BIAS RULE:** When in doubt between ISSUE and CLEAN, choose ISSUE.
    It is better to flag a marginal issue (which Step 2/3 can remove) than to miss a real one.
    Human reviewers consistently find more issues than AI. If your total issues < 4, re-scan harder.
    """

        # 4. STEP 1: INITIAL GENERATION
        print("\n--- Step 1: Generating Initial Analysis JSON ---")
        _pre_step1_delay = int(os.environ.get("GEMINI_PRE_STEP1_DELAY_SEC", "10"))
        print(f"--- Waiting {_pre_step1_delay}s before Step 1 (post-upload quota cooldown) ---")
        time.sleep(_pre_step1_delay)

        response_1 = _call_genai_with_backoff(
            lambda: client.models.generate_content(
                model=MODEL_NAME,
                contents=[combined_prompt] + transcript_files + audio_files + frame_files,
                config=generation_config,
            ),
            what="models.generate_content(step1)",
        )
        initial_json = response_1.text.strip()
        
        # Extract JSON from markdown or raw text
        if "```json" in initial_json:
            initial_json = initial_json.split("```json")[1].split("```")[0].strip()
        elif "```" in initial_json:
            initial_json = initial_json.split("```")[1].split("```")[0].strip()
        
        # Validate initial JSON
        is_valid_initial, initial_validation = validate_json_response(initial_json)
        retry_count = 0
        
        while not is_valid_initial and retry_count < MAX_EMPTY_JSON_RETRIES:
            retry_count += 1
            print(f"[WARNING] Invalid Initial JSON: {initial_validation}")
            print(f"[DEBUG] First 200 chars of response: {initial_json[:200]}")
            print(f"[RETRY] Regenerating Step 1 (retry {retry_count}/{MAX_EMPTY_JSON_RETRIES})...")
            
            # Increase max_output_tokens on each retry to prevent truncation
            retry_gen_kwargs = dict(gen_config_kwargs)
            retry_gen_kwargs["max_output_tokens"] = (gen_config_kwargs.get("max_output_tokens") or DEFAULT_MAX_OUTPUT_TOKENS) + (RETRY_TOKEN_BOOST * retry_count)
            retry_gen_config = types.GenerateContentConfig(**{k: v for k, v in retry_gen_kwargs.items() if v is not None})
            print(f"[RETRY] Boosted max_output_tokens to {retry_gen_kwargs['max_output_tokens']}")
            
            response_1 = _call_genai_with_backoff(
                lambda: client.models.generate_content(
                    model=MODEL_NAME,
                    contents=[combined_prompt + '\n\nCRITICAL: Your output MUST be a single, COMPLETE, valid JSON object with these EXACT top-level keys: "_reasoning_trace" (array), "meta" (object with tutor_id, group_id, session_date, session_summary), "positive_feedback" (array), "areas_for_improvement" (array), "flags" (array), "scoring" (object with setup/attitude/preparation/curriculum/teaching arrays, averages object, final_weighted_score number), "action_plan" (array). Do NOT truncate — output the FULL JSON from opening { to closing }.']
                    + transcript_files
                    + audio_files
                    + frame_files,
                    config=retry_gen_config,
                ),
                what="models.generate_content(step1-retry)",
            )
            initial_json = response_1.text.strip()
            if "```json" in initial_json:
                initial_json = initial_json.split("```json")[1].split("```")[0].strip()
            elif "```" in initial_json:
                initial_json = initial_json.split("```")[1].split("```")[0].strip()
            is_valid_initial, initial_validation = validate_json_response(initial_json)
        
        if not is_valid_initial:
             print(f"[ERROR] Step 1 Failed: {initial_validation}")
        elif isinstance(initial_validation, dict):
            # Re-serialize in case validate_json_response auto-fixed keys
            initial_json = json.dumps(initial_validation, ensure_ascii=False, indent=2)

        # Process Step 1 Result
        initial_json, data_step1 = recalculate_score(initial_json)
        score_step1 = data_step1.get("scoring", {}).get("final_weighted_score", 0)
        
        step1_report_path = os.path.splitext(output_report_path)[0] + "_Step1.json"
        with open(step1_report_path, 'w', encoding='utf-8') as f:
            f.write(initial_json)
        print(f"[SUCCESS] Step 1 Analysis Saved (Score: {score_step1}): {step1_report_path}")

        flow_data["step1"] = _build_step_flow_snapshot(
            "step1",
            data_step1 if isinstance(data_step1, dict) else {},
            score_step1,
            PRIORITY_SUBCATEGORY_TARGETS,
        )

        # Delay between steps to avoid hitting API rate limits
        step_delay = int(os.environ.get("GEMINI_STEP_DELAY_SEC", "30"))
        print(f"\n--- Waiting {step_delay}s before Step 2 (rate-limit cooldown) ---")
        time.sleep(step_delay)

        # 5. STEP 2: SELF-AUDIT (RESTORED)
        print("\n--- Step 2: Deep Audit & Verification (With Full Context) ---")
        
        audit_prompt = f"""
You are the **Senior Quality Compliance Auditor** for iSchool.
Your task is to **AUDIT and CORRECT** the "Draft Analysis JSON" below.

**THE DRAFT MAY CONTAIN ERRORS. DO NOT TRUST IT BLINDLY.**
You have the **ACTUAL** session evidence (Audio, Frames, Transcript) to verify against.
You also have access to the 4 iSchool Quality Guideline PDFs via File Search — use them to verify every finding against the official rules.

**Session Start Time: {start_time}** — ANY issue cited before this timestamp is INVALID and must be deleted.

**STEP 0 GROUNDED CONTEXT (MANDATORY REFERENCE):**
{step0_context}

**ISSUE MATCHING PRIORITY (CRITICAL):**
You MUST explicitly re-check these high-miss subcategories before finalizing:
- Session study
- Knowledge About Subject
- Tools used and Methodology
- Class Management
- Student Engagement
- Session Synchronization
If a subcategory remains at 5, state a brief evidence-backed reason in `_reasoning_trace`.
If total issues are fewer than 3, re-audit these six subcategories before final output.

**STEP 2 FLOW CONTRACT (MANDATORY):**
- Run a dedicated "Missed Subcategory Pass" BEFORE other phases.
- For each of the six priority subcategories, add exactly one `_reasoning_trace` line:
    `FLOW Step2 | <subcategory> | ISSUE|CLEAN | KEEP|ADD|REMOVE | <why>`
- If Step 1 marked a subcategory as ISSUE and Step 2 changes it to CLEAN, you must state the removal reason.
- If Step 1 marked CLEAN and Step 2 changes to ISSUE, you must cite the added evidence.
- Do not skip any of the six subcategories.

""" + rules_context + """

---

## AUDIT PROTOCOL (6 PHASES — Execute in order)

### PHASE 1: ADDITIVE AUDIT — FIND WHAT STEP 1 MISSED (MOST IMPORTANT PHASE)
**This is the MOST CRITICAL phase. Execute it FIRST, before filtering.**
**Step 1 frequently UNDER-DETECTS issues.** Human reviewers find 2-3x more issues than Step 1.
You have the ACTUAL audio and transcript. You MUST now actively look for issues Step 1 MISSED.

**RE-LISTEN to the audio and CHECK the transcript for EACH of these 7 commonly-missed themes:**

**Theme 1: INTERACTION & ENGAGEMENT (missed 32% of the time)**
- Is the session predominantly one-sided (tutor talks, student silent)?
- Does the tutor ask open-ended questions or just lecture?
- During project work, does the tutor check in with the student or remain silent?
- Is the student talk ratio below ~20%?
→ If YES, ADD: **T - Student Engagement: [specific description with timestamp]**

**Theme 2: PROJECT IMPLEMENTATION (missed 32% of the time)**
- Did the student actually implement/code the project themselves?
- Or did the tutor do all the coding while the student watched?
- Was the project completed and demonstrated/run?
→ If issues found, ADD: **T - Project Implementation & Activities: [specific description]** or **C - Slides and project completion: [description]**

**Theme 3: DISTRACTION / FOCUS (missed 29% of the time)**
- Does the tutor appear distracted (long pauses, delayed responses, typing unrelated things)?
- Is the student distracted (on bed, playing, not paying attention) and tutor doesn't address it?
- Are there signs of multitasking by the tutor?
→ If YES, ADD: **T - Class Management: [specific description with timestamp]**

**Theme 4: TIMING & PUNCTUALITY (missed 23% of the time)**
- Was the session significantly shorter than 60 minutes?
- Did the session end early without enrichment activities?
- Was there excessive waiting time that wasn't addressed?
→ If YES, ADD: **T - Class Management: [specific description with duration]**

**Theme 5: HOMEWORK & FOLLOW-UP (missed 16% of the time)**
- Was previous homework reviewed at the start?
- Was new homework assigned at the end?
- Was there follow-up on previous session topics?
→ If missing, ADD: **C - Homework: [specific description]**

**Theme 6: FEEDBACK QUALITY (missed 16% of the time)**
- Does the tutor give specific, constructive feedback or just "good" / "okay"?
- When the student makes an error, does the tutor explain WHY it's wrong?
- Is encouragement present but generic?
→ If weak, ADD: **A - Friendliness: [specific description]** or **T - Student Engagement: [description]**

**Theme 7: TONE & ENERGY (missed 16% of the time)**
- Is the tutor's tone warm and encouraging, or flat/monotone/harsh?
- Does the tutor sound bored or disengaged?
- Is there energy variation or is the entire session delivered in one flat tone?
→ If problematic, ADD: **A - Voice Tone & Clarity: [specific description with timestamp]**

**Theme 8: SESSION SYNCHRONIZATION (missed frequently — human reviewers flag this often)**
- Does the session follow ALL expected phases: greeting → review → concept intro → practice → project → closure?
- **CHECK EACH PHASE EXPLICITLY:**
  * Did the tutor review the PREVIOUS session's content? (If not: missing review phase)
  * Did the tutor introduce new concepts BEFORE jumping to implementation? (If not: missing concept phase)
  * Did the tutor wrap up with a summary of what was learned? (If not: missing closure)
  * Was homework from the previous session discussed? (If not: missing review)
- Are transitions between phases smooth or abrupt/disorganized?
- Does the tutor skip straight from greeting to coding without any concept explanation?
→ ONLY flag Session Synchronization if **3 or more major phases are COMPLETELY absent** AND the overall session structure is fundamentally chaotic/disorganized throughout. Do NOT flag if only 1–2 phases are shortened, informal, or briefly handled. Missing a full review AND missing a concept intro AND missing closure = flag. Missing just one phase = do NOT flag.

**Theme 9: KNOWLEDGE & PREPARATION GAPS (missed frequently)**
- Does the tutor explain any concept INCORRECTLY or give misleading definitions?
- When the student asks a question, does the tutor answer confidently and correctly, or hesitate/dodge/give wrong answer?
- Does the tutor struggle with the IDE, debugging, or technical setup during the session?
- Does the tutor rely too heavily on reading from slides instead of explaining in their own words?
- Does the tutor open the wrong file or project and spend time finding the right one?
→ If knowledge gaps found, ADD: **P - Knowledge About Subject: [specific description with timestamp]**
→ If preparation gaps found (wrong files, reading from slides, unconfident delivery), ADD: **P - Session study: [description]**

**Theme 10: SLIDES & PROJECT COMPLETION (missed frequently)**
- Were all required slides covered, or were some skipped?
- Was the project completed and demonstrated/run during the session?
- Was the project only explained verbally but never actually implemented?
→ If incomplete, ADD: **C - Slides and project completion: [specific description]**

**Theme 11: TOOLS & METHODOLOGY (missed frequently)**
- Does the tutor use the CORRECT version of required software? (Desktop app vs web version — desktop is usually required)
- Does the tutor use effective teaching METHODOLOGY? Consider:
  * Does the tutor use annotations, highlighting, or drawing during explanations?
  * Does the tutor use scaffolding (break complex tasks into steps)?
  * Does the tutor use analogies or relatable examples to explain abstract concepts?
  * Or does the tutor just talk verbally without any visual aids or structured methodology?
- Is the screen shared effectively so the student can follow along?
→ If wrong software version, ADD: **P - Project software & slides: [description of wrong tool]**
→ If weak methodology, ADD: **C - Tools used and Methodology: [specific description]**

**Theme 12: PERSONAL PLATFORM / CONFIDENTIALITY BREACH (always critical if present)**
- During screen sharing, did any personal, internal, or non-educational platform appear? (e.g., Discord chat, personal WhatsApp, internal staff channels, private email, non-iSchool apps)
- Was a video/media played WITHOUT pausing to discuss or interact with the student? (tutor plays video and watches passively instead of stopping/pausing to explain or ask questions)
- Was content shown that should NOT be visible to students (internal teacher communications, personal notifications, staff-only materials)?
→ If a personal/internal platform was accidentally displayed: ADD **S - Environment: [description + timestamp]** — this is a privacy/professionalism concern.
→ If video played without discussion: ADD **T - Student Engagement: [description + timestamp]** — student was passive viewer, not an active learner.

**Theme 13: DASHBOARD PROJECT UPLOAD — DO NOT FLAG**
- Dashboard upload is not reliably verifiable from video/audio analysis alone. This check has been disabled due to high false positive rate.
- Do NOT flag dashboard upload under any circumstances — not as a deduction, not as [Advice], not as a note.

**GROUND-TRUTH EXAMPLES: What these high-miss issues ACTUALLY look like in real sessions:**

**Session Synchronization = 3 (real example):**
The tutor jumped from greeting directly to coding without reviewing the previous session, skipped the concept explanation phase entirely, and ended without summarizing or assigning homework. The session structure was: greeting (2 min) → coding (55 min) → abrupt end. Missing: review, concept intro, closure.
→ Issue: "T - Session Synchronization: Session lacked structured phases. Jumped from greeting to coding without concept explanation or review. No summary or closure at end. [00:15:00-01:10:00]"

**Knowledge About Subject = 3 (real example):**
The tutor incorrectly explained that a function parameter and a variable are the same thing. When the student asked about return values, the tutor hesitated for 15 seconds and gave an incomplete answer. The tutor also confused "class" with "object" during explanation.
→ Issue: "P - Knowledge About Subject: Tutor showed knowledge gaps — confused parameter with variable [00:25:00], gave incomplete explanation of return values [00:38:00], mixed up class vs object concepts [00:42:00]"

**Session study = 3 (real example):**
The tutor opened the wrong project file at session start and spent 3 minutes looking for the correct one. During the lesson, the tutor visibly read explanations directly from slides rather than explaining concepts in their own words. At one point the IDE had compilation errors that the tutor couldn't quickly resolve.
→ Issue: "P - Session study: Tutor appeared underprepared — opened wrong project file [00:16:00], read directly from slides instead of explaining naturally [00:28:00], struggled with IDE errors [00:45:00]"

**Tools used and Methodology = 3 (real example):**
The tutor only used verbal explanation throughout the entire session with no visual aids. No annotations, no code highlighting, no diagrams. When explaining a complex concept, the tutor just talked without showing anything on screen. Never demonstrated the concept with actual code.
→ Issue: "C - Tools used and Methodology: Session relied entirely on verbal explanation without visual aids, annotations, or code demonstrations. Complex concepts were explained only verbally. [00:20:00-00:50:00]"

**RULES FOR ADDING NEW ISSUES:**
- Each new issue MUST have a specific timestamp
- For audio/behavioral issues: provide an Arabic quote from audio OR a clear behavioral description with timestamp
- For structural/pedagogical issues (Session study, Knowledge, Synchronization, Tools): a clear description of the pattern with timestamps is sufficient — these are holistic assessments, not single-quote issues
- Do NOT add phantom issues — only add what you can VERIFY in the audio/transcript/frames
- Do NOT flag the tutor for CORRECTLY enforcing iSchool policies
- **TARGET: A typical session has 3-8 issues. If Step 1 found fewer than 3, look harder BUT only add issues you are 100% certain about.**
- **100% ACCURACY RULE:** Every issue you add MUST be undeniably present in the evidence. If you have doubt, do NOT add it. A borderline or uncertain issue is worse than a missing issue because it lowers the score unfairly.

### PHASE 2: EVIDENCE VERIFICATION (Filter unsubstantiated claims)
For EACH item in "areas_for_improvement" and "flags" (including items you just added in Phase 1):
- **Timestamp Check:** Does the cited timestamp fall AFTER the start time ({start_time})?
- **Audio Verification:** If citing tone/audio issues, is it clearly audible?
- **FRAME EVIDENCE SCOPE:** Frames ARE valid for visual/setup checks (camera, face, backdrop, dress code, screen sharing). For non-visual issues, audio evidence is required.
- **Student-Side Check:** Is this issue caused by the STUDENT? If YES, REMOVE.
  * Before keeping ANY Setup issue: "Is this about the TUTOR's setup, or is the tutor INSTRUCTING THE STUDENT?" If instruction → REMOVE.
  * INTERNET: Only flag TUTOR-SIDE disconnections lasting >1 MINUTE.
  * CAMERA OFF <1 min for internet = STANDARD PRACTICE → REMOVE.
- **Contradiction Check:** Does the report praise AND penalize the same behavior? Trust the STRONGER evidence.
→ **REMOVE** findings that fail evidence checks.
→ **100% ACCURACY RULE:** When in doubt, REMOVE the finding. A false positive that a human reviewer would delete is worse than a missed issue. Only keep findings that are 100% certain and clearly evidenced.

### PHASE 2A: QUICK PHANTOM CHECK
Delete ONLY these specific phantom patterns:
1. "Presentation not in full-screen / slideshow mode" → DELETE.
2. "Mouse pointer instead of Zoom annotation tools" → DELETE.
3. "Zoom annotation tools not used / preferred" → DELETE (annotation is style preference, not violation).
4. "Language Used" about speaking Arabic → DELETE (keep only profanity/insults).
5. Student-side problems (student internet, student camera) → DELETE.

→ Keep ALL other issues. Do NOT over-filter.
→ **If fewer than 3 issues remain, go back to Phase 1 and look harder.**

### PHASE 3: MANDATORY SESSION DURATION CHECK
You MUST verify the actual teaching duration:
- Find the timestamp when the student joins and meaningful interaction begins.
- Find the timestamp when the session ends (farewell/goodbye).
- Actual Duration = End Time - Start Time.
- **TUTOR JOINING TIME:** Calculate waiting time = transcript start time (gap from 00:00:00 of audio). If >=15 min, tutor joined early (POSITIVE). If between 5 and 15 min, tutor joined within the acceptable window — do NOT flag. If <5 min, actively check for signs of late joining (apology, student complaint, very short audio gap). If start_time < 00:02:00 with no evidence of prior waiting → FLAG as late joining.
### PHASE 4: SUBCATEGORY VALIDATION
Verify that EVERY subcategory name in the JSON matches an **approved subcategory** from the rules reference above.
- Approved category keys: S (Setup), A (Attitude), P (Preparation), C (Curriculum), T (Teaching), F (Feedback)
- If a subcategory name doesn't match any approved name, correct it to the closest match
- Text format must be: `[Category Letter] - [Subcategory]: [Description] - [Evidence: ...]`

### PHASE 5: ISSUE-COUNT → SCORE ALIGNMENT
After removing invalid issues, align subcategory ratings with actual issue counts PER SUBCATEGORY:

| Issues in subcategory | Rating | Description |
|----------------------|--------|-------------|
| 0 issues | **5** (Perfect) | No problems found in this subcategory |
| 1 issue | **4** (Good) | One area for improvement noted |
| 2-3 issues | **3** (Fair) | Multiple issues — notable concern |
| 4+ issues OR 2+ Yellow Flags | **2** (Weak) | Significant concerns |
| 7+ issues | **1** (Critical) | Major systemic problems |
| No show / Total failure | **0** (Zero) | Complete failure |

**DEFAULT rating is 5 (Perfect) when zero issues found.** Only lower the rating when "areas_for_improvement" items exist for that specific subcategory.
Use whole numbers only (5, 4, 3, 2, 1, 0). Do NOT use half-points like 4.5, 3.5.

**[Advice] ITEMS ARE NOT DEDUCTIONS:** Items prefixed with '[Advice]' in areas_for_improvement are advisory suggestions only. Do NOT count them toward the per-subcategory issue count when assigning ratings. Only count items WITHOUT the '[Advice]' prefix.

**CRITICAL RULE:** Count issues PER SUBCATEGORY (excluding [Advice] items). If a subcategory has 1+ non-[Advice] improvement areas, it MUST be rated **4 or lower** (not 5).

**MANDATORY: Evaluate ALL 19 subcategories** (5 for Setup, 4 for Attitude, 3 for Preparation, 3 for Curriculum, 4 for Teaching). This ensures issues across multiple subcategories in the same category properly reduce the category average.

### PHASE 6: MATHEMATICAL SCORE RECALCULATION
Using ONLY the validated findings, recalculate the final score:

**Step A:** Calculate category averages (average of all subcategory ratings within each category)
**Step B:** Apply the EXACT weighted formula:
- Setup:      (Setup_Avg ÷ 5) × 100 × **0.25**
- Attitude:   (Attitude_Avg ÷ 5) × 100 × **0.20**
- Preparation:(Prep_Avg ÷ 5) × 100 × **0.15**
- Curriculum: (Curr_Avg ÷ 5) × 100 × **0.15**
- Teaching:   (Teach_Avg ÷ 5) × 100 × **0.25**

**Step C:** Sum all weighted scores = final_weighted_score
**Step D:** Verify the math adds up. If it doesn't, fix it.

### PHASE 6B: OVERALL SCORE VERIFICATION (MANDATORY)
After calculating the final score, verify it is reasonable:
- The score should reflect the TOTAL number and distribution of issues across subcategories.
- Subcategories with 0 issues = 5, subcategories with 1-4 issues = 4, etc.
- If you have many issues concentrated in few subcategories, those subcategories are lowered but others remain at 5.
- If issues are spread across many subcategories, more subcategories drop from 5 to 4, lowering the overall score more.
- **Sanity check:** Count total issues, count how many subcategories are affected. The score should be consistent.

**MANDATORY: ALL 19 SUBCATEGORIES** must be present in the scoring section. This is critical for accurate score calculation.

### PHASE 7: OUTPUT POLISH
- **Positive Highlights:** Ensure at least 3 distinct, evidence-backed positives
- **Language:** Arabic IS the session language. Remove ALL \"Language Used\" findings about Arabic usage or mixing Arabic/English. ONLY keep if profanity/insults.
- **Evidence Quality:** Re-read EVERY \"cite\" field. If it looks like garbled Zoom auto-transcription (nonsensical Arabic, disconnected words, \"...\"-separated fragments copied from transcript), REWRITE it from what you HEARD in the audio, or DELETE the finding.
- **Conciseness:** Merge repetitive points. Use "Consistently..." for recurring behaviors.
- **Action Plan:** 3 actionable, constructive recommendations

---

## SCORING CALIBRATION (FEW-SHOT REFERENCE)

Use these REAL human reviewer examples to calibrate your scoring:

**Example A (Score: 100/100 — 0 improvement areas, 0 flags):**
Human comment: "Tutor's Performance: Outstanding; no areas for improvement identified."
→ All 19 subcategories = 5 (0 issues each) → Setup 5, Attitude 5, Prep 5, Curriculum 5, Teaching 5 → 100
→ This is VALID when literally NOTHING can be improved. It is more common than you expect (~10-15% of sessions).

**Example B (Score: 96/100 — 2 issues in 2 different subcategories, 0 flags):**
Issues:
- "S - Internet Quality: Occasional internet lags" (1 issue in Internet Quality → rating 4)
- "C - Homework: No homework review observed at session start" (1 issue → rating 4)
NOTE: "Zoom annotation tools" is NOT a valid deduction — it is a style preference, not a violation.
→ Setup avg = (5+4+5+5+5)/5=4.8, Attitude 5, Prep 5, Curriculum avg = (4+5+5)/3=4.67, Teaching 5 → 96.6

**Example C (Score: 92/100 — 4 issues across 3 subcategories, 0 flags):**
Issues:
- "T - Student Engagement: Kindly avoid spending long stretches explaining without involving the student" (1 of 2 issues in Student Engagement)
- "T - Project Implementation: Student had limited opportunity to independently implement" (1 issue → rating 4)
- "T - Student Engagement: Limited interaction during project phase" (2nd issue in Student Engagement → rating 4)
- "S - Internet Quality: Occasional internet lags" (1 issue → rating 4)
→ Setup avg = 4.8, Attitude 5, Prep 5, Curriculum 5, Teaching avg = (5+4+5+4)/4=4.5 → ~95.2

**Example D (Score: 85/100 — 7 issues across 5 subcategories, 0 flags):**
Issues:
- "A - Friendliness: Cold session start" (1 of 2 issues in Friendliness → rating 4)
- "A - Voice Tone: Low energy" (1 issue → rating 4)
- "T - Student Engagement: Excessive lecturing" (1 of 2 issues → rating 4)
- "T - Student Engagement: Minimal student participation" (2nd issue)
- "T - Class Management: Session ended early" (1 issue → rating 4)
- "P - Project software: Presentation issues" (1 issue → rating 4)
- "C - Tools used: Not enough Zoom annotations" (1 issue → rating 4)
→ Setup 5, Attitude avg = (4+5+5+4)/4=4.5, Prep avg = (5+4+5)/3=4.67, Curriculum avg = (5+5+4)/3=4.67, Teaching avg = (4+5+5+4)/4=4.5 → ~92.3

**Example E (Score: 80/100 — 8+ issues, systemic problems across many subcategories):**
Issues:
- "T - Class Management: Session only 40 minutes" (1 of 2 issues → rating 4)
- "T - Student Engagement: Very low throughout" (1 of 2 → rating 4)
- "T - Project Implementation: No project by student" (1 issue → rating 4)
- "T - Class Management: Poor time management" (2nd issue)
- "T - Student Engagement: One-sided session" (2nd issue)
- "P - Project software: Web version instead of desktop" (1 issue → rating 4)
- "C - Slides and project: Slides rushed" (1 issue → rating 4)
- "A - Language Used: Informal language" (1 issue → rating 4)
- "S - Environment: Dim lighting" (1 issue → rating 4)
→ Setup avg = (4+5+5+5+5)/5=4.8, Attitude avg = (5+4+5+5)/4=4.75, Prep avg = (5+4+5)/3=4.67, Curriculum avg = (5+4+5)/3=4.67, Teaching avg = (4+4+5+4)/4=4.25 → ~91.2

**KEY CALIBRATION RULE:** Your final score MUST be consistent with the number and distribution of improvement areas. Count issues PER SUBCATEGORY to determine each rating.

## EVIDENCE QUALITY EXAMPLES (MANDATORY REFERENCE)

**✅ CORRECT evidence (Arabic quotes from audio):**
- cite: "امسح امسح دي حط هنا كده الموف" → Actual Arabic quote proving excessive step-by-step dictation
- cite: "برافو عليك تم برسنت إكس جدا يا عبد الله" → Actual praise quote (positive feedback)
- cite: "ممكن تنزل الكاميرا شوية" → Actual Arabic instruction
- cite: "يلا نبدأ النهاردة هنتكلم عن الفانكشنز" → Session roadmap evidence
- cite: "انت عملت الواجب؟" → Homework check evidence

**❌ WRONG evidence (English inference — NEVER USE):**
- cite: "The tutor explains abstraction himself instead of asking the student" → YOUR description, not a quote
- cite: "The tutor's microphone is active and picking up ambient noise" → YOUR observation, not what was said
- cite: "The tutor tells the student not to take screenshots" → English summary, not the actual Arabic quote
- cite: "Student was mostly silent during project work" → Inference, not a quote

**EVIDENCE FLEXIBILITY BY SUBCATEGORY TYPE:**
- **Audio/behavioral issues** (Tone, Friendliness, Language): Arabic quote from audio is best evidence.
- **Structural/pedagogical issues** (Session study, Knowledge About Subject, Session Synchronization, Tools used and Methodology, Slides and project completion, Class Management): A clear description of the observed PATTERN with timestamps is valid evidence. These are holistic assessments — no single Arabic quote can prove "the session lacked synchronization" or "the tutor's knowledge was weak." Describe what you observed across the session.
- **Visual issues** (Environment, Camera, Dress Code): Frame-based description is valid evidence.
- If you have NEITHER a quote NOR a clear behavioral description with timestamps, the issue has no valid evidence and must be DELETED.

## NON-ISSUES (DO NOT FLAG THESE):
1. Tutor telling student to follow rules = POSITIVE coaching, not an issue
2. Pre-session microphone/setup activity = before start time, NOT a violation
3. Tutor joined early and waited = POSITIVE punctuality

## ADDITIONAL FEW-SHOT EXAMPLES:

**Example F (Score: 100/100 — 0 issues, tutor enforced rules correctly):**
The tutor told the student not to take screenshots — this is correct policy enforcement, NOT an issue.
The tutor's mic was on before the student joined — this is PRE-SESSION, NOT an issue.
The tutor re-explained a concept during checkup — this is normal review, NOT an issue.
→ All subcategories = 5:  100

**Example G (Score: 96/100 — 2 real issues in 2 subcategories, with Arabic evidence):**
Issues:
- "T - Student Engagement: الطالب كان ساكت أغلب الوقت والمعلم بيشرح لوحده من غير ما يسأل" cite: "الفانكشن دي بتاخد باراميتر وبترجع ريتيرن فاليو" (10 min monologue without student interaction) [00:35:00]
- "C - Homework: لم يتم مراجعة الواجب" cite: Session started directly with new content, no homework review [00:15:00]
→ Student Engagement = 4, Homework = 4, all others = 5 → 96

**Example H (Score: 88/100 — 5 issues across 4 subcategories, Arabic session):**
Issues:
- "A - Session Initiation: المعلم بدأ الحصة بدون ترحيب" cite: "يلا نبدأ" (no greeting) [00:12:00]
- "A - Voice Tone: المعلم صوته مونوتون" cite: Flat tone audible throughout middle segment [00:30:00]
- "T - Student Engagement: الطالب مش بيشارك" cite: "فاهم؟ أيوه" (student only says "أيوه" for 20 min) [00:25:00-00:45:00]
- "T - Project Implementation: المعلم بيكتب الكود بدل الطالب" cite: "اكتب كده... لا استنى انا هعملهالك" [00:50:00]
- "T - Class Management: الحصة خلصت بدري بدون أنشطة" cite: Session ended at 48 min [00:48:00]
→ Session Initiation=4, Voice Tone=4, Student Engagement=4, Project Implementation=4, Class Management=4, all others=5 → 88

---

**INPUT DRAFT JSON:**
{initial_json}

**REQUIRED OUTPUT:**
Return ONLY the clean, corrected, finalized JSON object (no markdown, no explanation).
You MUST use EXACTLY this JSON schema — do NOT invent a different structure:
```
{{
  "_reasoning_trace": ["Step 1: ...", "Step 2: ..."],
  "meta": {{"tutor_id": "str", "group_id": "str", "session_date": "str", "session_summary": "str"}},
  "positive_feedback": [{{"category": "str", "subcategory": "str", "text": "str", "cite": "str", "timestamp": "str"}}],
  "areas_for_improvement": [{{"category": "str", "subcategory": "str", "text": "str", "cite": "str", "timestamp": "str"}}],
  "flags": [{{"level": "Yellow/Red", "subcategory": "str", "reason": "str", "cite": "str", "timestamp": "str"}}],
  "scoring": {{
    "setup": [{{"subcategory": "str", "rating": 0, "reason": "str"}}],
    "attitude": [{{"subcategory": "str", "rating": 0, "reason": "str"}}],
    "preparation": [{{"subcategory": "str", "rating": 0, "reason": "str"}}],
    "curriculum": [{{"subcategory": "str", "rating": 0, "reason": "str"}}],
    "teaching": [{{"subcategory": "str", "rating": 0, "reason": "str"}}],
    "averages": {{"setup": 0, "attitude": 0, "preparation": 0, "curriculum": 0, "teaching": 0}},
    "final_weighted_score": 0
  }},
  "action_plan": ["string", "string", "string"]
}}
```
Do NOT use alternative key names like "overall_score", "subcategory_scores", "positive_highlights", or any other structure. Use EXACTLY the keys shown above.
"""
        # Create a separate generation config for Step 2 with guaranteed output budget
        step2_gen_config_kwargs = dict(gen_config_kwargs)
        step2_gen_config_kwargs["max_output_tokens"] = max(
            gen_config_kwargs.get("max_output_tokens") or 0, DEFAULT_MAX_OUTPUT_TOKENS
        )
        step2_gen_config_kwargs["response_schema"] = QUALITY_REPORT_SCHEMA  # V33: enforce JSON structure
        step2_generation_config = types.GenerateContentConfig(**{
            k: v for k, v in step2_gen_config_kwargs.items() if v is not None
        })

        # Step 2 audits Step 1's JSON — no PDFs needed (rules already in prompt text, saves ~55k tokens)
        response_2 = _call_genai_with_backoff(
            lambda: client.models.generate_content(
                model=MODEL_NAME,
                contents=transcript_files + [audit_prompt],  # V33: context first, query last (per docs)
                config=step2_generation_config,
            ),
            what="models.generate_content(step2)",
        )
        final_json_text = response_2.text.strip()
        
        # Extract JSON from markdown
        if "```json" in final_json_text:
            final_json_text = final_json_text.split("```json")[1].split("```")[0].strip()
        elif "```" in final_json_text:
            final_json_text = final_json_text.split("```")[1].split("```")[0].strip()

        # Validate Final JSON with retry
        is_valid, validation_result = validate_json_response(final_json_text)
        
        step2_retry = 0
        while not is_valid and step2_retry < MAX_EMPTY_JSON_RETRIES:
            step2_retry += 1
            print(f"[WARNING] Step 2 Invalid JSON: {validation_result}")
            print(f"[DEBUG] First 200 chars of response: {final_json_text[:200]}")
            print(f"[RETRY] Regenerating Step 2 (retry {step2_retry}/{MAX_EMPTY_JSON_RETRIES})...")
            
            # Increase max_output_tokens on each retry to prevent truncation
            step2_retry_kwargs = dict(gen_config_kwargs)
            step2_retry_kwargs["max_output_tokens"] = max(
                gen_config_kwargs.get("max_output_tokens") or 0, DEFAULT_MAX_OUTPUT_TOKENS
            ) + (RETRY_TOKEN_BOOST * step2_retry)
            step2_retry_kwargs["response_schema"] = QUALITY_REPORT_SCHEMA  # V33: enforce JSON structure
            step2_retry_config = types.GenerateContentConfig(**{k: v for k, v in step2_retry_kwargs.items() if v is not None})
            print(f"[RETRY] Boosted max_output_tokens to {step2_retry_kwargs['max_output_tokens']}")
            
            response_2 = _call_genai_with_backoff(
                lambda: client.models.generate_content(
                    model=MODEL_NAME,
                    contents=transcript_files + [audit_prompt + '\n\nCRITICAL: Your output MUST be a single, COMPLETE JSON object with these EXACT top-level keys: "_reasoning_trace" (array), "meta" (object with tutor_id, group_id, session_date, session_summary), "positive_feedback" (array), "areas_for_improvement" (array), "flags" (array), "scoring" (object with setup/attitude/preparation/curriculum/teaching arrays, averages object, final_weighted_score number), "action_plan" (array). Do NOT use any alternative key names like overall_score or subcategory_scores. Do NOT truncate — output the FULL JSON from opening { to closing }.'],  # V33: context first
                    config=step2_retry_config,
                ),
                what="models.generate_content(step2-retry)",
            )
            final_json_text = response_2.text.strip()
            if "```json" in final_json_text:
                final_json_text = final_json_text.split("```json")[1].split("```")[0].strip()
            elif "```" in final_json_text:
                final_json_text = final_json_text.split("```")[1].split("```")[0].strip()
            is_valid, validation_result = validate_json_response(final_json_text)

        if not is_valid:
            print(f"[ERROR] Step 2 Invalid JSON after retries: {validation_result}. Falling back to Initial JSON.")
            final_json_text = initial_json
            is_valid = is_valid_initial
        elif isinstance(validation_result, dict):
            # Re-serialize in case validate_json_response auto-fixed keys
            final_json_text = json.dumps(validation_result, ensure_ascii=False, indent=2)
        
        # --- SCORE RECALCULATION ---
        # (Using global recalculate_score function)

        
        # Final Processing
        final_json_text, data2 = recalculate_score(final_json_text)
        score2 = data2.get("scoring", {}).get("final_weighted_score", 0)
        
        step2_report_path = os.path.splitext(output_report_path)[0] + "_Step2.json"
        with open(step2_report_path, 'w', encoding='utf-8') as f:
            f.write(final_json_text)
        print(f"[SUCCESS] Step 2 Analysis Saved (Score: {score2}): {step2_report_path}")

        flow_data["step2"] = _build_step_flow_snapshot(
            "step2",
            data2 if isinstance(data2, dict) else {},
            score2,
            PRIORITY_SUBCATEGORY_TARGETS,
        )

        # STEP 3: RECONCILIATION — Merge reports in Python, verify with audio
        print(f"\n--- Score Comparison ---")
        print(f"Step 1 Score: {score_step1}")
        print(f"Step 2 Score: {score2}")
        
        print(f"\n--- Step 3: Audio Verification (Merged Report) ---")

        # Delay between steps to avoid hitting API rate limits
        step_delay = int(os.environ.get("GEMINI_STEP_DELAY_SEC", "30"))
        print(f"--- Waiting {step_delay}s before Step 3 (rate-limit cooldown) ---")
        time.sleep(step_delay)

        # --- PYTHON MERGE: Combine Step 1 + Step 2 into one report ---
        step1_data_parsed = json.loads(initial_json) if isinstance(initial_json, str) else initial_json
        step2_data_parsed = data2 if isinstance(data2, dict) else json.loads(final_json_text)
        merged_report = _merge_step1_step2(step1_data_parsed, step2_data_parsed)
        # Recalculate scores on the merged report
        merged_json_str, merged_report = recalculate_score(json.dumps(merged_report, ensure_ascii=False, indent=2))
        merged_score = merged_report.get("scoring", {}).get("final_weighted_score", 0) if isinstance(merged_report, dict) else 0
        merged_issues = len(merged_report.get("areas_for_improvement", [])) if isinstance(merged_report, dict) else 0
        print(f"[MERGE] Combined Step1+Step2 in Python: {merged_issues} issues, score={merged_score}")

        step1_hits = flow_data.get("step1", {}).get("priority_hits", [])
        step2_hits = flow_data.get("step2", {}).get("priority_hits", [])
        step1_missing = flow_data.get("step1", {}).get("priority_missing", [])
        step2_missing = flow_data.get("step2", {}).get("priority_missing", [])
        priority_flow_summary = (
            f"Step1 hits: {step1_hits} | missing: {step1_missing}\n"
            f"Step2 hits: {step2_hits} | missing: {step2_missing}"
        )

        reconciliation_prompt = f"""
You are a **Quality Report Verifier** at iSchool.
You receive ONE pre-merged analysis report and the official quality rules.
You HAVE the session audio and transcript — use them to VERIFY each finding.

{rules_context}

---

## MERGED REPORT (Score: {merged_score}):
{merged_json_str}

## PRIORITY SUBCATEGORY FLOW FROM EARLIER STEPS:
{priority_flow_summary}

---

## YOUR TASK: VERIFY & FINALIZE (KEEP-BIASED)

**CRITICAL: Your job is to VERIFY findings, NOT to reduce them.**
The merged report was produced by two independent analysis passes. Both steps found these issues.
You should KEEP the vast majority of findings. Only remove a finding if audio CLEARLY CONTRADICTS it.

**REMOVAL BUDGET:** You may remove AT MOST 2 findings. If you want to remove more, the evidence must be overwhelming.
**ADDITION IS ENCOURAGED:** If you find issues that both steps missed, ADD them.

### 1. VERIFY EACH FINDING AGAINST AUDIO
For each item in areas_for_improvement:
- **LISTEN** to the audio at the cited timestamp
- If audio CONFIRMS or is CONSISTENT with the described behavior → **KEEP**
- If audio CLEARLY CONTRADICTS the claim (opposite behavior observed) → **REMOVE with explanation**
- If evidence is partial, ambiguous, or uncertain → **REMOVE** (100% accuracy rule: only keep what is undeniable)
- **DEFAULT ACTION when in doubt: REMOVE.** A false positive hurts the tutor unfairly. Only keep a finding if you can point to specific, clear evidence.

### 2. PHANTOM ISSUES (always remove — these 3 only)
- "Presentation not in full-screen / slideshow mode" → DELETE
- "Mouse pointer instead of annotation tools" → DELETE **ONLY IF** tutor used annotations at some point in the session. If tutor NEVER used annotations throughout the ENTIRE session → KEEP this finding.

**ZERO-FINDINGS PROTECTION:** If removing all findings would result in 0 issues (score = 100), you MUST use audio/transcript to AFFIRMATIVELY CONFIRM the session is genuinely perfect before outputting 0 issues. Simply not finding contradictory evidence is NOT sufficient — you must find positive evidence of excellence across all 19 subcategories. If in doubt, keep at least 1 finding and score accordingly.

### 3. RULES-BASED VALIDATION
- "Language Used" about speaking Arabic → DELETE (Arabic IS the session language; only flag profanity/insults)
- Student-side problems → NOT tutor deductions
- Any issue cited before session start time ({start_time}) is INVALID

### 4. PRIORITY SUBCATEGORY DEEP-CHECK
For each subcategory below, verify using audio. Only add findings if you hear CLEAR evidence of a problem. Do NOT guess or hallucinate issues on good sessions.

- **Session study** (P): Re-listen to the first 10 min. Does the tutor open the right files immediately? Does the tutor explain naturally or awkwardly read from slides? If ANY clear hesitation or wrong file is detected → ADD finding.
- **Knowledge About Subject** (P): Listen for ANY moment where the tutor gives technically incorrect info (e.g., confusing variable with parameter, wrong syntax), hesitates for >10s on a question, or gives a vague/dodging answer. If found → ADD finding.
- **Tools used and Methodology** (C): When explaining *complex or new* concepts, does the tutor use visual aids (annotations, diagrams, code demos)? Or is it unstructured verbal-only explanation that leaves the student confused? Does the tutor use scaffolding (break complex tasks into steps)? If teaching method systematically fails to support the student → ADD finding. Do NOT flag simple review conversations.
- **Class Management** (T): Calculate actual teaching duration. Any dead time >3 min without reason? Is the session <50 min or >80 min without justification? If yes → ADD finding. Sessions between 50–80 minutes are acceptable with no comment.
- **Student Engagement** (T): Listen to the WHOLE session: Is there a back-and-forth dialogue? If the tutor delivers unbroken monologues (>5 min) while the student is totally silent, or if the student only says "yes/ok/uh-huh" for large portions (>15 min) while struggling to understand → ADD finding. Do NOT flag if the student is actively engaged in coding or answering questions.
- **Session Synchronization** (T): Only flag if MULTIPLE MAJOR phases are COMPLETELY absent (e.g., NO review at all AND no concept intro AND no proper closure). Do NOT flag if a single phase is shortened, informal, or briefly touched. Minor deviations in an otherwise well-organized session = do NOT flag. The threshold is high: the overall structure must be clearly disorganized and educationally harmful to justify flagging.

### 5. SCORING
Per-subcategory rating (count "Areas for Improvement" WITHOUT '[Advice]' prefix for that subcategory only):
**[Advice] items are suggestions — do NOT count them as deductions when rating subcategories.**
- **5 (Perfect):** 0 non-[Advice] issues found.
- **4 (Good):** 1 "Areas for Improvement" from this subcategory.
- **3 (Fair):** 2-3 "Areas for Improvement" from this subcategory — notable concern.
- **2 (Weak):** 2+ Yellow Flags OR 4+ "Areas for Improvement" from this subcategory.
- **1 (Critical):** 7+ "Areas for Improvement" from this subcategory.
- **0 (Zero):** No show / Total failure.
- Weighted formula: Setup 25%, Attitude 20%, Preparation 15%, Curriculum 15%, Teaching 25%

### 6. ALL 19 SUBCATEGORIES REQUIRED
Setup: Environment, Internet Quality, Camera Quality, Microphone Quality, Dress Code
Attitude: Friendliness, Language Used, Session Initiation & Closure, Voice Tone & Clarity
Preparation: Knowledge About Subject, Project software & slides, Session study
Curriculum: Homework, Slides and project completion, Tools used and Methodology
Teaching: Class Management, Project Implementation & Activities, Session Synchronization, Student Engagement

### 7. FLOW CONTRACT (MANDATORY)
Add one `_reasoning_trace` line per priority subcategory:
`FLOW Step3 | <subcategory> | VERIFIED|ADDED|REMOVED | <audio_verification_summary>`

---

**OUTPUT:** Return ONLY valid JSON (no markdown). Use EXACTLY this schema:
```
{{
  "_reasoning_trace": ["Verified X at timestamp...", "Removed Y because audio shows...", "Score rationale: ..."],
  "meta": {{"tutor_id": "str", "group_id": "str", "session_date": "str", "session_summary": "str"}},
  "positive_feedback": [{{"category": "str", "subcategory": "str", "text": "str", "cite": "str", "timestamp": "str"}}],
  "areas_for_improvement": [{{"category": "str", "subcategory": "str", "text": "str", "cite": "str", "timestamp": "str"}}],
  "flags": [{{"level": "Yellow/Red", "subcategory": "str", "reason": "str", "cite": "str", "timestamp": "str"}}],
  "scoring": {{
    "setup": [{{"subcategory": "str", "rating": 0, "reason": "str"}}],
    "attitude": [{{"subcategory": "str", "rating": 0, "reason": "str"}}],
    "preparation": [{{"subcategory": "str", "rating": 0, "reason": "str"}}],
    "curriculum": [{{"subcategory": "str", "rating": 0, "reason": "str"}}],
    "teaching": [{{"subcategory": "str", "rating": 0, "reason": "str"}}],
    "averages": {{"setup": 0, "attitude": 0, "preparation": 0, "curriculum": 0, "teaching": 0}},
    "final_weighted_score": 0
  }},
  "action_plan": ["string", "string", "string"]
}}
```
"""
        step3_gen_config_kwargs = dict(gen_config_kwargs)
        step3_gen_config_kwargs["max_output_tokens"] = max(
            gen_config_kwargs.get("max_output_tokens") or 0, DEFAULT_MAX_OUTPUT_TOKENS
        )
        step3_gen_config_kwargs["response_schema"] = QUALITY_REPORT_SCHEMA  # V33: enforce JSON structure
        step3_generation_config = types.GenerateContentConfig(**{
            k: v for k, v in step3_gen_config_kwargs.items() if v is not None
        })

        # Step 3 reconciles Steps 1+2 JSON only — no PDFs needed (rules already in prompt text, saves ~55k tokens)
        response_3 = _call_genai_with_backoff(
            lambda: client.models.generate_content(
                model=MODEL_NAME,
                contents=transcript_files + [reconciliation_prompt],  # V33: context first, query last (per docs)
                config=step3_generation_config,
            ),
            what="models.generate_content(step3-verification)",
        )
        reconciled_json_text = response_3.text.strip()

        # Extract JSON from markdown
        if "```json" in reconciled_json_text:
            reconciled_json_text = reconciled_json_text.split("```json")[1].split("```")[0].strip()
        elif "```" in reconciled_json_text:
            reconciled_json_text = reconciled_json_text.split("```")[1].split("```")[0].strip()

        # Validate reconciled JSON with retry
        is_valid_r, validation_r = validate_json_response(reconciled_json_text)

        step3_retry = 0
        while not is_valid_r and step3_retry < MAX_EMPTY_JSON_RETRIES:
            step3_retry += 1
            print(f"[WARNING] Step 3 Invalid JSON: {validation_r}")
            print(f"[DEBUG] First 300 chars of Step 3 response: {reconciled_json_text[:300]}")
            print(f"[RETRY] Regenerating Step 3 (retry {step3_retry}/{MAX_EMPTY_JSON_RETRIES})...")
            
            step3_retry_kwargs = dict(gen_config_kwargs)
            step3_retry_kwargs["max_output_tokens"] = max(
                gen_config_kwargs.get("max_output_tokens") or 0, DEFAULT_MAX_OUTPUT_TOKENS
            ) + (RETRY_TOKEN_BOOST * step3_retry)
            step3_retry_kwargs["response_schema"] = QUALITY_REPORT_SCHEMA  # V33: enforce JSON structure
            step3_retry_config = types.GenerateContentConfig(**{k: v for k, v in step3_retry_kwargs.items() if v is not None})
            print(f"[RETRY] Boosted max_output_tokens to {step3_retry_kwargs['max_output_tokens']}")
            
            response_3 = _call_genai_with_backoff(
                lambda: client.models.generate_content(
                    model=MODEL_NAME,
                    contents=transcript_files + [reconciliation_prompt + '\n\nCRITICAL: Output a COMPLETE JSON object. Do NOT truncate.'],  # V33: context first
                    config=step3_retry_config,
                ),
                what="models.generate_content(step3-retry)",
            )
            reconciled_json_text = response_3.text.strip()
            if "```json" in reconciled_json_text:
                reconciled_json_text = reconciled_json_text.split("```json")[1].split("```")[0].strip()
            elif "```" in reconciled_json_text:
                reconciled_json_text = reconciled_json_text.split("```")[1].split("```")[0].strip()
            is_valid_r, validation_r = validate_json_response(reconciled_json_text)

        if not is_valid_r:
            print(f"[ERROR] Step 3 Invalid JSON after retries. Falling back to merged report.")
            reconciled_json_text = merged_json_str
        elif isinstance(validation_r, dict):
            reconciled_json_text = json.dumps(validation_r, ensure_ascii=False, indent=2)

        # Recalculate score for reconciled output
        reconciled_json_text, data3 = recalculate_score(reconciled_json_text)

        # Apply post-processing calibration (ceiling based on issue count)
        data3 = apply_issue_based_calibration(data3)
        
        # V19: Validate and clean findings (remove empty flags, phantom issues, non-Arabic evidence)
        data3 = validate_and_clean_findings(data3)
        
        reconciled_json_text = json.dumps(data3, ensure_ascii=False, indent=2)
        score3 = data3.get("scoring", {}).get("final_weighted_score", 0)

        # V29: LOWER-WINS LOGIC — pick the report with MORE issues (lower score)
        # This prevents Step 3 from "recovering" scores by removing valid findings.
        merged_report_clean = validate_and_clean_findings(merged_report.copy() if isinstance(merged_report, dict) else json.loads(merged_json_str))
        merged_json_recalc, merged_report_clean = recalculate_score(json.dumps(merged_report_clean, ensure_ascii=False, indent=2))
        merged_score_clean = merged_report_clean.get("scoring", {}).get("final_weighted_score", 0) if isinstance(merged_report_clean, dict) else 0
        merged_issues_clean = len(merged_report_clean.get("areas_for_improvement", [])) if isinstance(merged_report_clean, dict) else 0
        step3_issues = len(data3.get("areas_for_improvement", []))
        
        print(f"[LOWER-WINS] Merged: score={merged_score_clean}, issues={merged_issues_clean} | Step3: score={score3}, issues={step3_issues}")
        
        if merged_issues_clean > step3_issues:
            # Merged report found MORE issues — use it (lower score = more detection)
            print(f"[LOWER-WINS] Using MERGED report (more issues: {merged_issues_clean} vs {step3_issues})")
            data3 = merged_report_clean
            reconciled_json_text = merged_json_recalc
            score3 = merged_score_clean
        else:
            print(f"[LOWER-WINS] Keeping STEP 3 report (same/more issues: {step3_issues} vs {merged_issues_clean})")

        step3_report_path = os.path.splitext(output_report_path)[0] + "_Step3.json"
        with open(step3_report_path, 'w', encoding='utf-8') as f:
            f.write(reconciled_json_text)
        print(f"[SUCCESS] Step 3 Reconciliation Saved (Score: {score3}): {step3_report_path}")
        print(f"  Step 1: {score_step1} -> Step 2: {score2} -> Final: {score3}")

        flow_data["step3"] = _build_step_flow_snapshot(
            "step3",
            data3 if isinstance(data3, dict) else {},
            score3,
            PRIORITY_SUBCATEGORY_TARGETS,
        )
        flow_data["deltas"] = {
            "step1_to_step2": round(score2 - score_step1, 2),
            "step2_to_step3": round(score3 - score2, 2),
            "step1_to_step3": round(score3 - score_step1, 2),
        }

        # Use the reconciled result as final
        final_json_text = reconciled_json_text
        final_data = data3
        final_score = score3

        # Save Final Report
        json_report_path = os.path.splitext(output_report_path)[0] + ".json"
        with open(json_report_path, 'w', encoding='utf-8') as f:
            f.write(final_json_text)
        
        with open(output_report_path, 'w', encoding='utf-8') as f:
            f.write(final_json_text) 

        print(f"[SUCCESS] Final Structured Reports saved (.json and .txt)")

        # Save structured step flow diagnostics for auditability.
        flow_path = os.path.splitext(output_report_path)[0] + "_AuditFlow.json"
        with open(flow_path, 'w', encoding='utf-8') as f:
            json.dump(flow_data, f, indent=2, ensure_ascii=False)
        print(f"[SUCCESS] Audit Flow Data Saved: {flow_path}")

        # 6. GENERATE HTML
        generate_html_report_from_json(json_report_path)
        
        # 7. CONSISTENCY TRACKING
        # Track scores across runs for the same session to detect variance
        session_id = os.path.basename(os.path.dirname(output_report_path))  # e.g., "T-4092"
        consistency_log_path = os.path.join(os.path.dirname(output_report_path), f"{session_id}_consistency_log.json")
        
        try:
            if os.path.exists(consistency_log_path):
                with open(consistency_log_path, 'r', encoding='utf-8') as f:
                    consistency_data = json.load(f)
            else:
                consistency_data = {"session_id": session_id, "runs": []}
            
            # Add this run
            import datetime
            consistency_data["runs"].append({
                "timestamp": datetime.datetime.now().isoformat(),
                "score": final_score,
                "seed": args.seed,
                "model": MODEL_NAME,
                "temperature": MODEL_TEMPERATURE,
                "thinking_level": getattr(args, 'thinking_level', DEFAULT_THINKING_LEVEL)
            })
            
            # Calculate statistics
            all_scores = [r["score"] for r in consistency_data["runs"]]
            if len(all_scores) > 1:
                median_score = compute_median_score(all_scores)
                variance = max(all_scores) - min(all_scores)
                consistency_data["statistics"] = {
                    "total_runs": len(all_scores),
                    "all_scores": all_scores,
                    "median_score": round(median_score, 1),
                    "min_score": min(all_scores),
                    "max_score": max(all_scores),
                    "variance": round(variance, 1),
                    "is_reliable": variance <= SCORE_VARIANCE_THRESHOLD
                }
                
                # Print consistency warning if variance is high
                if variance > SCORE_VARIANCE_THRESHOLD:
                    print(f"\n[WARNING] CONSISTENCY: Score variance is HIGH: {variance:.1f} points")
                    print(f"   Previous scores: {all_scores[:-1]}")
                    print(f"   Current score:   {final_score}")
                    print(f"   Median score:    {median_score:.1f}")
                    print(f"   Recommended: Use median score ({median_score:.1f}) for reporting")
                else:
                    print(f"\n[OK] CONSISTENCY: Score variance: {variance:.1f} points (threshold: {SCORE_VARIANCE_THRESHOLD})")
                    print(f"   Scores: {all_scores}")
            
            # Save consistency log
            with open(consistency_log_path, 'w', encoding='utf-8') as f:
                json.dump(consistency_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"[WARNING] Consistency tracking failed: {e}")

        # ================================================================
        # FINAL COST DETAILS  (Gemini 3 Flash pricing, per 1M tokens)
        # Input: $0.50 text/image/video, $1.00 audio
        # Output (incl. thinking): $3.00
        # Batch API: 50 % discount on all rates
        # ================================================================

        # Estimate audio tokens from the video/audio duration (32 tok/s)
        audio_input_tokens = int(video_duration_sec * AUDIO_TOKENS_PER_SECOND) if video_duration_sec > 0 else 0

        def _step_tokens(resp):
            """Extract input / output / thinking token counts from a response."""
            um = resp.usage_metadata if resp else None
            return (
                getattr(um, 'prompt_token_count', 0) or 0,
                getattr(um, 'candidates_token_count', 0) or 0,
                getattr(um, 'thoughts_token_count', 0) or 0,
            )

        in_0, out_0, think_0 = _step_tokens(response_0)
        in_1, out_1, think_1 = _step_tokens(response_1)
        in_2, out_2, think_2 = _step_tokens(response_2)
        in_3, out_3, think_3 = _step_tokens(response_3)

        total_in     = in_0 + in_1 + in_2 + in_3
        total_out    = out_0 + out_1 + out_2 + out_3
        total_think  = think_0 + think_1 + think_2 + think_3

        # Steps 1 & 2 receive audio; Step 3 (reconciliation) does not
        def _step_cost(inp, out, think, has_audio, batch=False):
            """Calculate cost for one step, splitting audio from text input."""
            if batch:
                rate_txt = BATCH_COST_PER_MILLION_INPUT_TOKENS
                rate_aud = BATCH_COST_PER_MILLION_AUDIO_INPUT_TOKENS
                rate_out = BATCH_COST_PER_MILLION_OUTPUT_TOKENS
            else:
                rate_txt = COST_PER_MILLION_INPUT_TOKENS
                rate_aud = COST_PER_MILLION_AUDIO_INPUT_TOKENS
                rate_out = COST_PER_MILLION_OUTPUT_TOKENS
            aud = audio_input_tokens if has_audio else 0
            txt = max(inp - aud, 0)
            cost_in  = (txt / 1e6 * rate_txt) + (aud / 1e6 * rate_aud)
            cost_out = ((out + think) / 1e6 * rate_out)
            return cost_in + cost_out

        has_audio_step0 = response_0 is not None and in_0 > 0
        cost_step0 = _step_cost(in_0, out_0, think_0, has_audio=has_audio_step0)
        cost_step1 = _step_cost(in_1, out_1, think_1, has_audio=True)
        cost_step2 = _step_cost(in_2, out_2, think_2, has_audio=True)
        cost_step3 = _step_cost(in_3, out_3, think_3, has_audio=False)
        total_cost  = cost_step0 + cost_step1 + cost_step2 + cost_step3

        batch_cost_step0 = _step_cost(in_0, out_0, think_0, has_audio=has_audio_step0, batch=True)
        batch_cost_step1 = _step_cost(in_1, out_1, think_1, has_audio=True,  batch=True)
        batch_cost_step2 = _step_cost(in_2, out_2, think_2, has_audio=True,  batch=True)
        batch_cost_step3 = _step_cost(in_3, out_3, think_3, has_audio=False, batch=True)
        batch_total_cost  = batch_cost_step0 + batch_cost_step1 + batch_cost_step2 + batch_cost_step3

        print(f"\n{'='*60}")
        print(f"Analysis complete. Pricing: Gemini 3 Flash (text $0.50/M, audio $1.00/M, output $3.00/M)")
        print(f"Audio tokens (estimated): {audio_input_tokens:,} ({video_duration_sec:.0f}s × 32 tok/s)")
        print(f"{'='*60}")
        print(f"Step 0: In={in_0:,}  Out={out_0:,}  Think={think_0:,}  | ${cost_step0:.4f}  Batch=${batch_cost_step0:.4f}")
        print(f"Step 1: In={in_1:,}  Out={out_1:,}  Think={think_1:,}  | ${cost_step1:.4f}  Batch=${batch_cost_step1:.4f}")
        print(f"Step 2: In={in_2:,}  Out={out_2:,}  Think={think_2:,}  | ${cost_step2:.4f}  Batch=${batch_cost_step2:.4f}")
        print(f"Step 3: In={in_3:,}  Out={out_3:,}  Think={think_3:,}  | ${cost_step3:.4f}  Batch=${batch_cost_step3:.4f}")
        print(f"{'─'*60}")
        print(f"Total:  In={total_in:,}  Out={total_out:,}  Think={total_think:,}")
        print(f"Total Cost (Regular API): ${total_cost:.4f}")
        print(f"Total Cost (Batch API):   ${batch_total_cost:.4f}")
        print(f"Batch API Savings:        ${total_cost - batch_total_cost:.4f} (50% discount)")
        print(f"{'='*60}")
        
        # Save cost details to a JSON file alongside the report
        cost_data = {
            "step0": {
                "input_tokens": in_0,
                "output_tokens": out_0,
                "thinking_tokens": think_0,
                "regular_cost_usd": round(cost_step0, 6),
                "batch_cost_usd": round(batch_cost_step0, 6)
            },
            "step1": {
                "input_tokens": in_1,
                "output_tokens": out_1,
                "thinking_tokens": think_1,
                "regular_cost_usd": round(cost_step1, 6),
                "batch_cost_usd": round(batch_cost_step1, 6)
            },
            "step2": {
                "input_tokens": in_2,
                "output_tokens": out_2,
                "thinking_tokens": think_2,
                "regular_cost_usd": round(cost_step2, 6),
                "batch_cost_usd": round(batch_cost_step2, 6)
            },
            "step3": {
                "input_tokens": in_3,
                "output_tokens": out_3,
                "thinking_tokens": think_3,
                "regular_cost_usd": round(cost_step3, 6),
                "batch_cost_usd": round(batch_cost_step3, 6)
            },
            "total": {
                "input_tokens": total_in,
                "output_tokens": total_out,
                "thinking_tokens": total_think,
                "audio_tokens_estimated": audio_input_tokens,
                "regular_cost_usd": round(total_cost, 6),
                "batch_cost_usd": round(batch_total_cost, 6),
                "batch_savings_usd": round(total_cost - batch_total_cost, 6),
                "batch_savings_percent": 50
            },
            "pricing": {
                "regular_input_per_million": COST_PER_MILLION_INPUT_TOKENS,
                "regular_audio_input_per_million": COST_PER_MILLION_AUDIO_INPUT_TOKENS,
                "regular_output_per_million": COST_PER_MILLION_OUTPUT_TOKENS,
                "batch_input_per_million": BATCH_COST_PER_MILLION_INPUT_TOKENS,
                "batch_audio_input_per_million": BATCH_COST_PER_MILLION_AUDIO_INPUT_TOKENS,
                "batch_output_per_million": BATCH_COST_PER_MILLION_OUTPUT_TOKENS
            }
        }
        cost_path = os.path.splitext(output_report_path)[0] + "_cost.json"
        with open(cost_path, 'w', encoding='utf-8') as f:
            json.dump(cost_data, f, indent=2)
        print(f"Cost details saved: {cost_path}")

    except Exception as e:
        print(f"Error in RAG analysis: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)
    finally:
        # Always delete uploaded Gemini files to avoid hitting per-project storage limits
        try:
            delete_uploaded_gemini_files(uploaded_files)
        except Exception:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=VIDEO_FILE_PATH)
    parser.add_argument("--output_report", default=OUTPUT_REPORT_TXT)
    parser.add_argument("--transcript", default=TRANSCRIPT_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility")
    parser.add_argument("--thinking_level", type=str, default=DEFAULT_THINKING_LEVEL, choices=["minimal", "low", "medium", "high"], help="Thinking level for Gemini 3 (minimal/low/medium/high)")
    parser.add_argument("--media_resolution", type=str, default=DEFAULT_MEDIA_RESOLUTION, choices=["MEDIA_RESOLUTION_LOW", "MEDIA_RESOLUTION_MEDIUM", "MEDIA_RESOLUTION_HIGH"], help="Media resolution for vision processing (impacts token usage and latency)")
    parser.add_argument("--max_output_tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS, help="Maximum output tokens (None = model default)")
    parser.add_argument("--consistency_runs", type=int, default=DEFAULT_CONSISTENCY_RUNS, help="Number of analysis runs for consistency (1=single run, 3=high reliability)")
    parser.add_argument("--use_google_search", action="store_true", help="Enable Google Search grounding (Community Search) for factual verification")
    args = parser.parse_args()
    
    # Print configuration for reproducibility tracking
    print(f"=== CONFIGURATION ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Temperature: {MODEL_TEMPERATURE} (Gemini 3 default)")
    print(f"Seed: {args.seed}")
    print(f"Thinking Level: {args.thinking_level}")
    print(f"Media Resolution: {args.media_resolution}")
    print(f"Consistency Runs: {args.consistency_runs}")
    print(f"Google Search: {args.use_google_search}")
    print(f"=====================")
    
    perform_rag_analysis(args.input, args.output_report, args.transcript)