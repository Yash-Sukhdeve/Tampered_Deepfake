#!/usr/bin/env python3
"""
Example Usage Script for Tampered Deepfake Pipeline

This script demonstrates how to use the pipeline programmatically.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode == 0:
        print(f"\n✓ {description} completed successfully")
        return True
    else:
        print(f"\n✗ {description} failed")
        return False

def main():
    """Main example workflow."""
    print("""
╔══════════════════════════════════════════════════════════╗
║   Tampered Deepfake Pipeline - Example Usage            ║
║   Demonstrating programmatic pipeline execution         ║
╚══════════════════════════════════════════════════════════╝
""")

    # Example 1: Run full pipeline
    print("\n## Example 1: Run Complete Pipeline ##")
    input("Press Enter to run full pipeline...")

    if not run_command(
        [sys.executable, "run_pipeline.py", "--all"],
        "Full Pipeline Execution"
    ):
        print("Pipeline failed. Check errors above.")
        sys.exit(1)

    # Example 2: Run individual steps
    print("\n\n## Example 2: Run Individual Steps ##")
    input("Press Enter to run step-by-step...")

    steps = [
        (["src/01_preprocess.py",
          "--input_dir", "data/sub096/original",
          "--output_dir", "output/preprocessed"],
         "Step 1: Preprocessing"),

        (["src/02_analyze.py",
          "--input_dir", "output/preprocessed",
          "--output_dir", "output/transcripts",
          "--whisper_model", "tiny"],
         "Step 2: Analysis (Whisper tiny)"),

        (["src/03_identify.py",
          "--audio_dir", "output/preprocessed",
          "--transcript_dir", "output/transcripts",
          "--output_dir", "output/metadata"],
         "Step 3: Candidate Identification"),

        (["src/04_tamper.py",
          "--candidate_dir", "output/metadata",
          "--original_audio_dir", "output/preprocessed",
          "--max_tampers_per_file", "5"],
         "Step 4: Audio Tampering (5 per file)"),
    ]

    for cmd, desc in steps:
        if not run_command([sys.executable] + cmd, desc):
            print(f"Step failed: {desc}")
            break

    # Example 3: Analyze Results
    print("\n\n## Example 3: Analyze Results ##")
    import json
    import glob

    candidate_files = glob.glob("output/metadata/*_candidates.json")
    print(f"\nTotal candidate files: {len(candidate_files)}")

    if candidate_files:
        # Load and summarize
        total_candidates = 0
        by_type = {}
        by_level = {}

        for f in candidate_files:
            with open(f) as fp:
                data = json.load(fp)
                total_candidates += data['total_candidates']

                for cand in data.get('candidates', []):
                    ctype = cand.get('type', 'unknown')
                    by_type[ctype] = by_type.get(ctype, 0) + 1

                    level = cand.get('difficulty', {}).get('level', 'unknown')
                    by_level[level] = by_level.get(level, 0) + 1

        print(f"\n📊 Results Summary:")
        print(f"  Total candidates: {total_candidates}")
        print(f"\n  By type:")
        for k, v in sorted(by_type.items()):
            print(f"    - {k}: {v}")
        print(f"\n  By difficulty:")
        for k, v in sorted(by_level.items()):
            print(f"    - {k}: {v}")

    tampered_files = glob.glob("output/tampered/*.wav")
    print(f"\n  Tampered audio files: {len(tampered_files)}")

    if tampered_files:
        print("\n  Sample tampered files:")
        for f in tampered_files[:5]:
            print(f"    - {Path(f).name}")

    print("\n\n✓ Example completed!")
    print("\nFor more details, see README.md")

if __name__ == "__main__":
    main()
