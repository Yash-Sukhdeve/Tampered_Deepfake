#!/usr/bin/env python3
"""
Audio Preprocessing Script for Tampered Deepfake Pipeline

This script converts m4a audio files to 16kHz mono WAV format with -20dBFS normalization.
Part of the audio tampering research pipeline for deepfake detection.

Author: Generated for Tampered_Deepfake research project
"""

import argparse
import logging
import sys
from pathlib import Path

from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from tqdm import tqdm
import soundfile as sf

# --- Constants ---
TARGET_SAMPLE_RATE = 16000  # Hz
TARGET_CHANNELS = 1         # Mono
TARGET_DBFS = -20.0         # dBFS for normalization
TARGET_SUBTYPE = 'PCM_16'   # 16-bit PCM for WAV export and validation

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

# --- Audio Processing Function ---
def process_audio_file(input_path: Path, output_path: Path, logger: logging.Logger) -> bool:
    """
    Converts an audio file to 16kHz mono WAV and normalizes its volume to -20dBFS.

    Args:
        input_path (Path): The path to the input audio file (e.g., .m4a).
        output_path (Path): The path where the processed WAV file will be saved.
        logger (logging.Logger): The logger instance for outputting messages.

    Returns:
        bool: True if the file was processed successfully, False otherwise.
    """
    logger.info(f"Attempting to process: {input_path.name}")
    try:
        # Load audio file
        # PyDub relies on ffmpeg/ffprobe. If not found, it raises FileNotFoundError.
        audio = AudioSegment.from_file(input_path)

        # Convert to target sample rate
        if audio.frame_rate != TARGET_SAMPLE_RATE:
            audio = audio.set_frame_rate(TARGET_SAMPLE_RATE)
            logger.debug(f"Changed sample rate of {input_path.name} to {TARGET_SAMPLE_RATE}Hz.")

        # Convert to mono
        if audio.channels != TARGET_CHANNELS:
            audio = audio.set_channels(TARGET_CHANNELS)
            logger.debug(f"Changed channels of {input_path.name} to mono.")

        # Normalize audio to -20dBFS
        # Calculate change needed to reach target dBFS
        change_in_dBFS = TARGET_DBFS - audio.dBFS
        audio = audio.apply_gain(change_in_dBFS)
        logger.debug(f"Normalized {input_path.name} to {TARGET_DBFS}dBFS.")

        # Export to WAV format with 16-bit PCM encoding
        # The parameters=["-acodec", "pcm_s16le"] explicitly ensures 16-bit PCM.
        audio.export(output_path, format="wav", parameters=["-acodec", "pcm_s16le"])
        logger.info(f"Successfully converted and normalized '{input_path.name}' to '{output_path.name}'.")
        return True

    except FileNotFoundError:
        logger.error(
            f"Error: ffmpeg/ffprobe not found. Please ensure ffmpeg is installed "
            f"and its executable is in your system's PATH. Failed to process '{input_path.name}'."
        )
    except CouldntDecodeError:
        logger.error(
            f"Error: Could not decode '{input_path.name}'. This might be due to a corrupted "
            f"file or an unsupported format by ffmpeg. Skipping."
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred while processing '{input_path.name}': {e}")
    return False

# --- Validation Function ---
def validate_audio_file(filepath: Path, logger: logging.Logger) -> bool:
    """
    Validates the properties of a processed WAV audio file against target specifications.

    Args:
        filepath (Path): The path to the WAV file to validate.
        logger (logging.Logger): The logger instance for outputting messages.

    Returns:
        bool: True if the file meets all target specifications, False otherwise.
    """
    try:
        info = sf.info(filepath)
        is_valid = True
        validation_messages = []

        if info.samplerate != TARGET_SAMPLE_RATE:
            validation_messages.append(
                f"Sample rate mismatch: Expected {TARGET_SAMPLE_RATE}Hz, got {info.samplerate}Hz."
            )
            is_valid = False
        if info.channels != TARGET_CHANNELS:
            validation_messages.append(
                f"Channel count mismatch: Expected {TARGET_CHANNELS} (mono), got {info.channels}."
            )
            is_valid = False
        # soundfile's subtype string might be 'PCM_16', 'FLOAT', etc.
        # We check if our target subtype is part of the reported subtype.
        if TARGET_SUBTYPE not in info.subtype:
            validation_messages.append(
                f"Bit depth/Subtype mismatch: Expected '{TARGET_SUBTYPE}', got '{info.subtype}'."
            )
            is_valid = False

        if is_valid:
            logger.info(
                f"Validation successful for '{filepath.name}': "
                f"{info.samplerate}Hz, {info.channels} channel(s), {info.subtype}."
            )
        else:
            logger.warning(
                f"Validation FAILED for '{filepath.name}': "
                f"{'; '.join(validation_messages)}"
            )
        return is_valid
    except FileNotFoundError:
        logger.error(f"Validation Error: File not found at '{filepath}'.")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during validation of '{filepath.name}': {e}")
        return False

# --- Main Function ---
def main():
    """
    Main function to parse arguments, process audio files, and log results.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess audio files (m4a to 16kHz mono WAV, -20dBFS normalization)."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Path to the directory containing original .m4a audio files."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Path to the directory where preprocessed .wav files will be saved."
    )
    args = parser.parse_args()

    logger = configure_logging()
    logger.info("Starting audio preprocessing script.")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Target format: {TARGET_SAMPLE_RATE}Hz, {TARGET_CHANNELS} channel(s), {TARGET_DBFS}dBFS, {TARGET_SUBTYPE}.")

    # Create output directory if it doesn't exist
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured output directory exists: {args.output_dir}")
    except Exception as e:
        logger.critical(f"Failed to create output directory '{args.output_dir}': {e}")
        sys.exit(1) # Exit if output directory cannot be created

    # Find all .m4a files in the input directory
    input_files = sorted(list(args.input_dir.glob('*.m4a')))
    if not input_files:
        logger.warning(f"No .m4a files found in the input directory: {args.input_dir}")
        return

    processed_count = 0
    skipped_count = 0
    failed_count = 0
    validated_count = 0

    # Process files with a progress bar
    for input_file in tqdm(input_files, desc="Preprocessing audio files"):
        output_file = args.output_dir / (input_file.stem + '.wav')

        if output_file.exists():
            logger.info(f"Skipping '{input_file.name}': Output file '{output_file.name}' already exists.")
            skipped_count += 1
            # Optionally validate skipped files too, to ensure existing files are correct
            if validate_audio_file(output_file, logger):
                validated_count += 1
            continue

        if process_audio_file(input_file, output_file, logger):
            processed_count += 1
            if validate_audio_file(output_file, logger):
                validated_count += 1
        else:
            failed_count += 1

    logger.info("-" * 50)
    logger.info("Preprocessing Summary:")
    logger.info(f"Total files found: {len(input_files)}")
    logger.info(f"Successfully processed: {processed_count}")
    logger.info(f"Skipped (already existed): {skipped_count}")
    logger.info(f"Failed to process: {failed_count}")
    logger.info(f"Validated successfully: {validated_count} (includes newly processed and existing valid files)")
    logger.info("Script finished.")

if __name__ == "__main__":
    # Check if ffmpeg is available
    import shutil
    if not shutil.which("ffmpeg"):
        logger = configure_logging()
        logger.critical(
            "ffmpeg not found. Please install ffmpeg and ensure it's in your system's PATH. "
            "Refer to https://ffmpeg.org/download.html for installation instructions."
        )
        sys.exit(1)

    main()
