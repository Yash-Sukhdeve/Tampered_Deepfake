#!/usr/bin/env python3
"""
Candidate Identification Script for Tampered Deepfake Pipeline

This script identifies tampering candidates from Whisper transcripts and calculates
difficulty scores based on prosodic analysis and acoustic properties.

Author: Generated for Tampered_Deepfake research project
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

from tqdm import tqdm

# Import utilities
from utils.nlp import (
    identify_negation_candidates,
    identify_quantifier_candidates,
    identify_adjective_adverb_deletions,
    parse_transcript_with_timestamps
)
from utils.prosody import extract_prosody_features, calculate_prosody_distance

# --- Constants ---
TYPE_WEIGHTS = {
    "deletion": 1.0,
    "insertion": 2.0,
    "substitution": 2.5
}

DIFFICULTY_THRESHOLDS = {
    "easy": 4.0,
    "medium": 8.0
    # hard is anything > medium
}

# --- Logging Setup ---
def configure_logging(log_level=logging.INFO):
    """Configures the logging for the script."""
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(ch)
    return logger

# --- Helper Functions ---
def match_word_to_timestamp(word: str, position: int, whisper_words: List[Dict]) -> Dict:
    """
    Matches an NLP-identified word to its timestamp from Whisper output.

    Args:
        word: The word text from NLP analysis
        position: Character position in the full text
        whisper_words: List of word dicts with text, start_time, end_time

    Returns:
        Dict with word info including timestamps, or None if not found
    """
    # Simple fuzzy matching - find word at approximately the right position
    # This is a basic implementation; could be enhanced with edit distance
    word_lower = word.lower().strip()

    for w in whisper_words:
        if w['text'].lower().strip() == word_lower:
            return w

    # Fallback: return None if exact match not found
    return None

def calculate_difficulty_score(
    candidate: Dict,
    audio_path: Path,
    whisper_words: List[Dict],
    logger: logging.Logger
) -> Dict:
    """
    Calculates difficulty score for a tampering candidate.

    Score = Type_Weight + Acoustic_Penalty + Rhythm_Penalty

    Args:
        candidate: Candidate dict with word, action, etc.
        audio_path: Path to audio file for prosody extraction
        whisper_words: List of word timestamps
        logger: Logger instance

    Returns:
        Dict with difficulty score and components
    """
    # Match candidate to timestamp
    matched_word = match_word_to_timestamp(
        candidate['word'],
        candidate.get('position', 0),
        whisper_words
    )

    if not matched_word:
        logger.warning(f"Could not match word '{candidate['word']}' to timestamp")
        return {
            "score": 5.0,  # Default medium score
            "level": "medium",
            "type_weight": 2.0,
            "acoustic_penalty": 2.0,
            "rhythm_penalty": 1.0
        }

    # Determine action type
    action = candidate.get('action', 'deletion')
    if 'add' in action or 'insert' in action or action == 'insertion':
        tamper_type = 'insertion'
    elif 'replace' in action or 'swap' in action or 'substitution' in action:
        tamper_type = 'substitution'
    else:
        tamper_type = 'deletion'

    type_weight = TYPE_WEIGHTS.get(tamper_type, 1.0)

    # Calculate rhythm penalty (duration-based)
    start_time = matched_word.get('start_time', 0)
    end_time = matched_word.get('end_time', 0.1)
    duration_ms = (end_time - start_time) * 1000
    rhythm_penalty = duration_ms / 100

    # Calculate acoustic penalty (prosodic discontinuity)
    # Extract prosody for the word and its neighbors
    acoustic_penalty = 0.0
    try:
        # Get word's prosody
        word_features = extract_prosody_features(audio_path, start_time, end_time)

        if word_features:
            # Find neighbors
            word_idx = None
            for idx, w in enumerate(whisper_words):
                if w == matched_word:
                    word_idx = idx
                    break

            if word_idx is not None and word_idx > 0 and word_idx < len(whisper_words) - 1:
                prev_word = whisper_words[word_idx - 1]
                next_word = whisper_words[word_idx + 1]

                # Extract neighbor prosody
                prev_features = extract_prosody_features(
                    audio_path,
                    prev_word['start_time'],
                    prev_word['end_time']
                )
                next_features = extract_prosody_features(
                    audio_path,
                    next_word['start_time'],
                    next_word['end_time']
                )

                if prev_features and next_features:
                    # Calculate discontinuity
                    prev_distance = calculate_prosody_distance(word_features, prev_features)
                    next_distance = calculate_prosody_distance(word_features, next_features)
                    acoustic_penalty = (prev_distance + next_distance) / 2
    except Exception as e:
        logger.debug(f"Error calculating acoustic penalty: {e}")
        acoustic_penalty = 1.0  # Default penalty

    # Total score
    total_score = type_weight + acoustic_penalty + rhythm_penalty

    # Determine difficulty level
    if total_score < DIFFICULTY_THRESHOLDS['easy']:
        level = 'easy'
    elif total_score < DIFFICULTY_THRESHOLDS['medium']:
        level = 'medium'
    else:
        level = 'hard'

    return {
        "score": float(total_score),
        "level": level,
        "type_weight": float(type_weight),
        "acoustic_penalty": float(acoustic_penalty),
        "rhythm_penalty": float(rhythm_penalty),
        "start_time": float(start_time),
        "end_time": float(end_time),
        "duration_ms": float(duration_ms)
    }

def identify_candidates_for_file(
    audio_path: Path,
    transcript_path: Path,
    output_path: Path,
    logger: logging.Logger
) -> bool:
    """
    Identifies tampering candidates for a single audio file.

    Args:
        audio_path: Path to preprocessed audio file
        transcript_path: Path to Whisper JSON transcript
        output_path: Path to save candidates JSON
        logger: Logger instance

    Returns:
        True if successful
    """
    try:
        # Load Whisper transcript
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)

        full_text = transcript_data.get('text', '')
        whisper_words = parse_transcript_with_timestamps(transcript_data)

        if not full_text or not whisper_words:
            logger.warning(f"No text or words found in {transcript_path.name}")
            return False

        logger.info(f"Processing {audio_path.name}: {len(whisper_words)} words")

        # Identify candidates using NLP
        negation_candidates = identify_negation_candidates(full_text)
        quantifier_candidates = identify_quantifier_candidates(full_text)
        deletion_candidates = identify_adjective_adverb_deletions(full_text)

        logger.info(f"Found {len(negation_candidates)} negation, {len(quantifier_candidates)} quantifier, {len(deletion_candidates)} deletion candidates")

        # Combine all candidates
        all_candidates = []

        # Process negations
        for cand in negation_candidates:
            cand['type'] = 'negation'
            cand['tamper_type'] = 'insertion' if 'add' in cand['action'] else 'deletion'
            difficulty = calculate_difficulty_score(cand, audio_path, whisper_words, logger)
            cand['difficulty'] = difficulty
            all_candidates.append(cand)

        # Process quantifiers
        for cand in quantifier_candidates:
            cand['type'] = 'quantifier'
            cand['action'] = 'substitute'
            cand['tamper_type'] = 'substitution'
            difficulty = calculate_difficulty_score(cand, audio_path, whisper_words, logger)
            cand['difficulty'] = difficulty
            all_candidates.append(cand)

        # Process deletions
        for cand in deletion_candidates:
            cand['type'] = 'adjective_adverb'
            cand['action'] = 'delete'
            cand['tamper_type'] = 'deletion'
            difficulty = calculate_difficulty_score(cand, audio_path, whisper_words, logger)
            cand['difficulty'] = difficulty
            all_candidates.append(cand)

        # Sort by difficulty score
        all_candidates.sort(key=lambda x: x['difficulty']['score'])

        # Save candidates
        output_data = {
            "source_audio": str(audio_path.name),
            "total_candidates": len(all_candidates),
            "candidates_by_level": {
                "easy": len([c for c in all_candidates if c['difficulty']['level'] == 'easy']),
                "medium": len([c for c in all_candidates if c['difficulty']['level'] == 'medium']),
                "hard": len([c for c in all_candidates if c['difficulty']['level'] == 'hard'])
            },
            "candidates": all_candidates
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(all_candidates)} candidates to {output_path.name}")
        return True

    except Exception as e:
        logger.error(f"Error processing {audio_path.name}: {e}")
        return False

# --- Main Function ---
def main():
    """Main function to identify tampering candidates."""
    parser = argparse.ArgumentParser(
        description="Identify tampering candidates from audio transcripts."
    )
    parser.add_argument(
        "--audio_dir",
        type=Path,
        default="output/preprocessed",
        help="Directory with preprocessed audio files."
    )
    parser.add_argument(
        "--transcript_dir",
        type=Path,
        default="output/transcripts",
        help="Directory with Whisper transcripts."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="output/metadata",
        help="Directory to save candidate JSON files."
    )
    args = parser.parse_args()

    logger = configure_logging()
    logger.info("Starting candidate identification.")
    logger.info(f"Audio directory: {args.audio_dir}")
    logger.info(f"Transcript directory: {args.transcript_dir}")
    logger.info(f"Output directory: {args.output_dir}")

    # Find all audio files
    audio_files = sorted(list(args.audio_dir.glob("*.wav")))
    if not audio_files:
        logger.warning(f"No audio files found in {args.audio_dir}")
        return

    logger.info(f"Found {len(audio_files)} audio files to process")

    successful = 0
    failed = 0

    for audio_file in tqdm(audio_files, desc="Identifying candidates"):
        transcript_path = args.transcript_dir / f"{audio_file.stem}.json"
        output_path = args.output_dir / f"{audio_file.stem}_candidates.json"

        if output_path.exists():
            logger.info(f"Candidates for {audio_file.name} already exist. Skipping.")
            continue

        if not transcript_path.exists():
            logger.warning(f"Transcript not found for {audio_file.name}")
            failed += 1
            continue

        if identify_candidates_for_file(audio_file, transcript_path, output_path, logger):
            successful += 1
        else:
            failed += 1

    logger.info("-" * 50)
    logger.info(f"Candidate identification complete: {successful} successful, {failed} failed")

if __name__ == "__main__":
    main()
