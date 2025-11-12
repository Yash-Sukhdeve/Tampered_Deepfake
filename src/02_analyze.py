#!/usr/bin/env python3
"""
Analysis Pipeline Script for Tampered Deepfake Pipeline

This script runs Whisper ASR and Montreal Forced Aligner (MFA) on preprocessed audio files.
Part of the audio tampering research pipeline for deepfake detection.

Author: Generated for Tampered_Deepfake research project
"""

import argparse
import logging
import sys
import subprocess
import json
from pathlib import Path

from tqdm import tqdm
import whisper

# --- Logging Setup ---
def configure_logging(log_level=logging.INFO):
    """
    Configures the logging for the script.

    Args:
        log_level (int): The minimum level of messages to log (e.g., logging.INFO, logging.DEBUG).
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # Create console handler and set level
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Add formatter to ch
    ch.setFormatter(formatter)

    # Add ch to logger
    if not logger.handlers: # Prevent adding multiple handlers if called multiple times
        logger.addHandler(ch)
    return logger

# --- Whisper ASR Function ---
def run_whisper(input_dir: Path, output_dir: Path, model_name: str, logger: logging.Logger):
    """
    Runs Whisper ASR on all preprocessed audio files.

    Args:
        input_dir (Path): The path to the directory containing preprocessed WAV files.
        output_dir (Path): The path where the transcript JSON files will be saved.
        model_name (str): The name of the Whisper model to use.
        logger (logging.Logger): The logger instance for outputting messages.
    """
    logger.info(f"Starting Whisper ASR with model: {model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(list(input_dir.glob("*.wav")))
    if not audio_files:
        logger.warning(f"No .wav files found in the input directory: {input_dir}")
        return

    logger.info(f"Found {len(audio_files)} audio files to process.")

    try:
        model = whisper.load_model(model_name)
    except Exception as e:
        logger.critical(f"Failed to load Whisper model '{model_name}': {e}")
        sys.exit(1)

    for audio_file in tqdm(audio_files, desc="Running Whisper ASR"):
        transcript_path = output_dir / f"{audio_file.stem}.json"
        if transcript_path.exists():
            logger.info(f"Transcript for {audio_file.name} already exists. Skipping.")
            continue

        try:
            # Note: word_timestamps disabled due to triton compatibility issues
            # MFA will provide phoneme-level timestamps instead
            result = model.transcribe(str(audio_file), word_timestamps=False)
            with open(transcript_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            logger.info(f"Successfully transcribed {audio_file.name}")
        except Exception as e:
            logger.error(f"Error transcribing {audio_file.name}: {e}")

    logger.info("Whisper ASR finished.")

# --- Montreal Forced Aligner (MFA) Function ---
def run_mfa(input_dir: Path, transcript_dir: Path, output_dir: Path, num_jobs: int, logger: logging.Logger):
    """
    Runs Montreal Forced Aligner (MFA) on all files.

    Args:
        input_dir (Path): The path to the directory containing preprocessed WAV files.
        transcript_dir (Path): The path to the directory containing the transcript JSON files.
        output_dir (Path): The path where the TextGrid files will be saved.
        num_jobs (int): The number of CPU cores to use for parallel processing.
        logger (logging.Logger): The logger instance for outputting messages.
    """
    logger.info("Starting Montreal Forced Aligner (MFA)...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare corpus for MFA
    corpus_dir = output_dir / "mfa_corpus"
    corpus_dir.mkdir(exist_ok=True)
    logger.info(f"Preparing MFA corpus in: {corpus_dir}")

    audio_files = sorted(list(input_dir.glob("*.wav")))
    for audio_file in tqdm(audio_files, desc="Preparing MFA corpus"):
        transcript_path = transcript_dir / f"{audio_file.stem}.json"
        if not transcript_path.exists():
            logger.warning(f"Transcript for {audio_file.name} not found. Skipping.")
            continue

        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)

        text = transcript_data["text"]

        # Create a .lab file with the transcript
        lab_path = corpus_dir / f"{audio_file.stem}.lab"
        with open(lab_path, "w", encoding="utf-8") as f:
            f.write(text)

        # Create a symbolic link to the audio file
        symlink_path = corpus_dir / audio_file.name
        if symlink_path.exists():
            symlink_path.unlink()
        symlink_path.symlink_to(audio_file.resolve())


    # Run MFA
    # This assumes you have a pre-trained acoustic model and a dictionary.
    # Replace "english_us_arpa" with the appropriate dictionary and "english" for the acoustic model.
    mfa_command = [
        "mfa",
        "align",
        str(corpus_dir),
        "english_us_arpa",
        "english",
        str(output_dir),
        "--output_format",
        "json",
        "--json_pretty",
        "--num_jobs",
        str(num_jobs),
        "--clean",
    ]

    logger.info(f"Running MFA command: {' '.join(mfa_command)}")

    try:
        process = subprocess.Popen(mfa_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
        for line in iter(process.stdout.readline, ''):
            logger.info(line.strip())
        process.wait()
        if process.returncode == 0:
            logger.info("MFA alignment finished successfully.")
        else:
            logger.error(f"MFA alignment failed with return code {process.returncode}.")

    except FileNotFoundError:
        logger.critical("MFA command not found. Please ensure MFA is installed and in your system's PATH.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during MFA alignment: {e}")

    logger.info("MFA finished.")


# --- Main Function ---
def main():
    """
    Main function to parse arguments and run the analysis pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Run Whisper ASR and Montreal Forced Aligner (MFA) on audio files."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default="output/preprocessed",
        help="Directory with preprocessed audio files."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="output/transcripts",
        help="Directory to save Whisper transcripts and MFA TextGrids."
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        default="large-v3",
        help="Whisper model to use (e.g., tiny, base, small, medium, large, large-v2, large-v3)."
    )
    parser.add_argument(
        "--mfa_jobs",
        type=int,
        default=24,
        help="Number of jobs for MFA."
    )
    args = parser.parse_args()

    logger = configure_logging()
    logger.info("Starting analysis pipeline.")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")

    # Run Whisper
    run_whisper(args.input_dir, args.output_dir, args.whisper_model, logger)

    # Run MFA
    run_mfa(args.input_dir, args.output_dir, args.output_dir, args.mfa_jobs, logger)

    logger.info("Analysis pipeline finished.")

if __name__ == "__main__":
    main()
