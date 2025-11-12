#!/usr/bin/env python3
"""
Master Pipeline Script for Tampered Deepfake Audio Generation

This script orchestrates the entire pipeline:
1. Preprocessing (m4a to 16kHz mono WAV)
2. Analysis (Whisper ASR + Montreal Forced Aligner)
3. Candidate Identification (NLP + Difficulty Scoring)
4. Audio Tampering (Deletion/Insertion/Substitution)

Usage:
    python run_pipeline.py --all  # Run entire pipeline
    python run_pipeline.py --step 1  # Run only preprocessing
    python run_pipeline.py --step 2  # Run only analysis
    python run_pipeline.py --step 3  # Run only candidate identification
    python run_pipeline.py --step 4  # Run only tampering

Author: Generated for Tampered_Deepfake research project
"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_step(step_name: str, script_path: str, args: list):
    """Run a pipeline step as a subprocess."""
    print(f"\n{'='*60}")
    print(f"Running Step: {step_name}")
    print(f"{'='*60}\n")

    cmd = [sys.executable, script_path] + args
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ {step_name} completed successfully\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {step_name} failed with error code {e.returncode}\n")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Master pipeline for tampered audio generation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all pipeline steps sequentially"
    )
    parser.add_argument(
        "--step",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run specific step: 1=preprocess, 2=analyze, 3=identify, 4=tamper"
    )
    parser.add_argument(
        "--whisper_model",
        default="large-v3",
        help="Whisper model for step 2 (default: large-v3)"
    )
    parser.add_argument(
        "--max_tampers",
        type=int,
        default=10,
        help="Max tampers per file for step 4 (default: 10)"
    )

    args = parser.parse_args()

    if not args.all and not args.step:
        parser.print_help()
        sys.exit(1)

    src_dir = Path("src")

    steps = {
        1: {
            "name": "Preprocessing",
            "script": src_dir / "01_preprocess.py",
            "args": [
                "--input_dir", "data/original",
                "--output_dir", "output/preprocessed"
            ]
        },
        2: {
            "name": "Analysis (Whisper + MFA)",
            "script": src_dir / "02_analyze.py",
            "args": [
                "--input_dir", "output/preprocessed",
                "--output_dir", "output/transcripts",
                "--whisper_model", args.whisper_model
            ]
        },
        3: {
            "name": "Candidate Identification",
            "script": src_dir / "03_identify.py",
            "args": [
                "--audio_dir", "output/preprocessed",
                "--transcript_dir", "output/transcripts",
                "--output_dir", "output/metadata"
            ]
        },
        4: {
            "name": "Audio Tampering",
            "script": src_dir / "04_tamper.py",
            "args": [
                "--candidate_dir", "output/metadata",
                "--original_audio_dir", "output/preprocessed",
                "--deepfake_base_dir", "data",
                "--output_dir", "output",
                "--max_tampers_per_file", str(args.max_tampers)
            ]
        }
    }

    # Run specified steps
    if args.all:
        print("\n" + "="*60)
        print("TAMPERED DEEPFAKE AUDIO PIPELINE")
        print("Running all 4 steps sequentially")
        print("="*60)

        for step_num in [1, 2, 3, 4]:
            step_info = steps[step_num]
            success = run_step(
                step_info["name"],
                str(step_info["script"]),
                step_info["args"]
            )
            if not success:
                print(f"\nPipeline failed at step {step_num}: {step_info['name']}")
                sys.exit(1)

        print("\n" + "="*60)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nOutputs:")
        print("  - Preprocessed audio: output/preprocessed/")
        print("  - Transcripts: output/transcripts/")
        print("  - Candidates: output/metadata/")
        print("  - Tampered audio: output/tampered/")
        print("  - Metadata: output/metadata/")

    else:
        step_info = steps[args.step]
        success = run_step(
            step_info["name"],
            str(step_info["script"]),
            step_info["args"]
        )
        if not success:
            sys.exit(1)

if __name__ == "__main__":
    main()
