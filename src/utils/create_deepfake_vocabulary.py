
#!/usr/bin/env python3
"""
Deepfake Vocabulary Generation Script

This script creates a detailed vocabulary of all words spoken in the deepfake
audio files. It performs the following steps:

1.  Finds all deepfake audio files in the data directory.
2.  Preprocesses the audio files (converts to 16kHz mono WAV, normalizes volume).
3.  Analyzes the audio files with Whisper ASR and Montreal Forced Aligner (MFA)
    to get transcripts and phoneme-level timestamps.
4.  Compiles the results into a detailed "deepfake vocabulary" JSON file.
"""

import argparse
import logging
import sys
from pathlib import Path
import json

# It's good practice to import the necessary functions from the other scripts
# so we can reuse the existing code.
from src.utils.audio import process_audio_file
from src.utils.prosody import extract_prosody_features
from src.utils.nlp import parse_transcript_with_timestamps
from src.utils.llm_analyzer import run_whisper_on_deepfake, run_mfa_on_deepfake

# --- Constants ---
DEEPFAKE_DATA_DIR = Path("data")
PREPROCESSED_DEEPFAKE_DIR = Path("output/preprocessed_deepfake")
DEEPFAKE_TRANSCRIPTS_DIR = Path("output/transcripts_deepfake")
VOCABULARY_FILE = Path("output/deepfake_vocabulary.json")

# --- Logging Setup ---
def configure_logging(log_level=logging.INFO):
    """Configures the logging for the script."""
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

LOGGER = configure_logging()

def find_deepfake_audio_files(data_dir: Path) -> list[Path]:
    """Finds all deepfake audio files."""
    LOGGER.info(f"Searching for deepfake audio files in {data_dir}...")
    # In this project, deepfake files are all .wav files
    deepfake_files = sorted(list(data_dir.glob("**/deepfake/**/*.wav")))
    LOGGER.info(f"Found {len(deepfake_files)} deepfake audio files.")
    return deepfake_files

def preprocess_deepfake_audio(deepfake_files: list[Path], output_dir: Path):
    """Preprocesses all deepfake audio files."""
    LOGGER.info("Preprocessing deepfake audio files...")
    output_dir.mkdir(parents=True, exist_ok=True)
    # This part will be filled in with the preprocessing logic.
    # For now, we'll just print a message.
    LOGGER.info("Preprocessing logic will be implemented here.")

def analyze_deepfake_audio(preprocessed_dir: Path, transcripts_dir: Path):
    """Analyzes all preprocessed deepfake audio files with Whisper and MFA."""
    LOGGER.info("Analyzing deepfake audio files...")
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    # This part will be filled in with the analysis logic.
    # For now, we'll just print a message.
    LOGGER.info("Analysis logic will be implemented here.")

def create_vocabulary_file(transcripts_dir: Path, preprocessed_dir: Path, output_file: Path):
    """Creates the deepfake vocabulary file."""
    LOGGER.info("Creating deepfake vocabulary file...")
    # This part will be filled in with the vocabulary creation logic.
    # For now, we'll just print a message.
    LOGGER.info("Vocabulary creation logic will be implemented here.")
    # The final vocabulary will be saved to the output_file.
    with open(output_file, "w") as f:
        json.dump([], f) # Save an empty list for now
    LOGGER.info(f"Deepfake vocabulary file created at {output_file}")

def main():
    """Main function to run the deepfake vocabulary generation."""
    parser = argparse.ArgumentParser(description="Generate a detailed vocabulary of all words spoken in the deepfake audio files.")
    parser.add_argument("--data_dir", type=Path, default=DEEPFAKE_DATA_DIR, help="Path to the main data directory.")
    parser.add_argument("--output_dir", type=Path, default=Path("output"), help="Path to the main output directory.")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set the logging level.")
    args = parser.parse_args()

    LOGGER.setLevel(args.log_level)

    # 1. Find all deepfake audio files
    deepfake_files = find_deepfake_audio_files(args.data_dir)
    if not deepfake_files:
        LOGGER.warning("No deepfake audio files found. Exiting.")
        return

    # 2. Preprocess the deepfake audio files
    preprocess_deepfake_audio(deepfake_files, PREPROCESSED_DEEPFAKE_DIR)

    # 3. Analyze the preprocessed deepfake audio files
    analyze_deepfake_audio(PREPROCESSED_DEEPFAKE_DIR, DEEPFAKE_TRANSCRIPTS_DIR)

    # 4. Create the vocabulary file
    create_vocabulary_file(DEEPFAKE_TRANSCRIPTS_DIR, PREPROCESSED_DEEPFAKE_DIR, VOCABULARY_FILE)

    LOGGER.info("Deepfake vocabulary generation complete.")

if __name__ == "__main__":
    main()
