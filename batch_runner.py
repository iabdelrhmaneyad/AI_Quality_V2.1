"""
batch_runner.py  —  AI Quality Model V2.1
==========================================
Batch-processes multiple sessions through rag_video_analysis.py.

SETUP:
  1. Set GEMINI_API_KEY in your environment (or .env file)
  2. Add your sessions to the `runs` list below
  3. Run:  python batch_runner.py

Each run entry: (session_id, mp4_path, transcript_txt_path, output_txt_path)
  - Paths can be absolute or relative to this file's location
  - Output JSON is written next to the output .txt automatically
  - Already-completed sessions (JSON exists) are skipped automatically
"""

import os, subprocess, sys, time

# ── Configuration ──────────────────────────────────────────────────────────
BASE  = os.path.dirname(os.path.abspath(__file__))   # folder this script lives in
RAG   = os.path.join(BASE, "rag_video_analysis.py")  # path to core engine

# ── Sessions list ──────────────────────────────────────────────────────────
# Add one tuple per session: (id, mp4, transcript, output_report)
# Use forward slashes or raw strings; relative paths are from BASE.
runs = [
    # Example entries — replace with your actual sessions:
    # ("T-1004",
    #  "Sessions/T-1004/T-1004_Jan_12_2026_Slot 5.mp4",
    #  "Sessions/T-1004/T-1004_Jan_12_2026_Slot 5.txt",
    #  "Sessions/T-1004/Quality_Report_T-1004.txt"),
    #
    # ("T-1012",
    #  "Sessions/T-1012/T-1012_Jan_11_2026_Slot 1.mp4",
    #  "Sessions/T-1012/T-1012_Jan_11_2026_Slot 1.txt",
    #  "Sessions/T-1012/Quality_Report_T-1012.txt"),
]

# ── Runner ─────────────────────────────────────────────────────────────────
def main():
    if not runs:
        print("No sessions configured. Add entries to the `runs` list in batch_runner.py.")
        return

    total   = len(runs)
    success = 0
    failed  = []

    print(f"\n{'='*60}")
    print(f"  AI Quality Batch Runner — {total} session(s)")
    print(f"{'='*60}\n")

    for idx, (tid, mp4, txt, out) in enumerate(runs, 1):
        mp4_full = mp4 if os.path.isabs(mp4) else os.path.join(BASE, mp4)
        txt_full = txt if os.path.isabs(txt) else os.path.join(BASE, txt)
        out_full = out if os.path.isabs(out) else os.path.join(BASE, out)
        json_out = out_full.replace(".txt", ".json")

        # Skip if already done
        if os.path.exists(json_out):
            print(f"[{idx}/{total}] SKIP {tid} — output JSON already exists")
            success += 1
            continue

        # Validate inputs
        if not os.path.exists(mp4_full):
            print(f"[{idx}/{total}] SKIP {tid} — MP4 not found: {mp4_full}")
            failed.append((tid, "MP4 not found"))
            continue
        if not os.path.exists(txt_full):
            print(f"[{idx}/{total}] SKIP {tid} — transcript not found: {txt_full}")
            failed.append((tid, "TXT not found"))
            continue

        print(f"\n[{idx}/{total}] Starting {tid} ...")
        t0 = time.time()

        result = subprocess.run(
            [sys.executable, RAG,
             "--input",          mp4_full,
             "--output_report",  out_full,
             "--transcript",     txt_full],
            capture_output=False,
            text=True
        )

        elapsed = round(time.time() - t0, 1)

        if result.returncode == 0 and os.path.exists(json_out):
            print(f"[{idx}/{total}] OK {tid} — done in {elapsed}s")
            success += 1
        else:
            print(f"[{idx}/{total}] FAIL {tid} — exit={result.returncode} ({elapsed}s)")
            failed.append((tid, f"exit code {result.returncode}"))

        # Brief pause between API calls to avoid rate limits
        if idx < total:
            time.sleep(2)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Completed: {success}/{total}")
    if failed:
        print(f"  Failed ({len(failed)}):")
        for tid, reason in failed:
            print(f"    - {tid}: {reason}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
