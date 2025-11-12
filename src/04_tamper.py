#!/usr/bin/env python3
"""
Audio Tampering Script for Tampered Deepfake Pipeline

This script generates tampered audio files based on candidates identified by
03_identify.py. It performs deletion, insertion, and substitution operations.

- Deletion: Removes a word or phrase from the original audio.
- Insertion: Finds a suitable word from a corpus of deepfake audio and
             inserts it into the original audio.
- Substitution: Replaces a word in the original audio with a word from the
                deepfake corpus.

The script uses prosody analysis to find the best-matching deepfake segments
for insertions and substitutions to create more natural-sounding tampers.

Author: Generated for Tampered_Deepfake research project
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import collections

from pydub import AudioSegment
from tqdm import tqdm

# Import utilities
from utils import audio as audio_utils
from utils import prosody as prosody_utils
from utils.nlp import parse_transcript_with_timestamps

# --- Constants ---
CROSSFADE_MS = 5  # Crossfade duration in milliseconds (2-5ms recommended)
DEEPFAKE_ROOT_DIR = Path('data')
TRANSCRIPT_DIR = Path('output/transcripts')

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

# --- Deepfake Word Searching ---

def find_deepfake_word_candidates(
    word_to_find: str,
    deepfake_audio_dirs: List[Path]
) -> List[Dict[str, Any]]:
    """
    Searches all deepfake transcripts to find occurrences of a specific word.

    Args:
        word_to_find: The word to search for (case-insensitive).
        deepfake_audio_dirs: A list of directories containing deepfake audio files.

    Returns:
        A list of candidate dictionaries, each containing file path, timestamps,
        and prosody features for an occurrence of the word.
    """
    candidates = []
    word_lower = word_to_find.lower().strip()
    LOGGER.debug(f"Searching for word '{word_lower}' in {len(deepfake_audio_dirs)} deepfake sets.")

    for df_dir in deepfake_audio_dirs:
        # Assuming deepfake audio files are .wav, .mp3, or .m4a
        audio_files = list(df_dir.glob("*.wav")) + list(df_dir.glob("*.m4a"))

        for audio_path in audio_files:
            transcript_path = TRANSCRIPT_DIR / f"{audio_path.stem}.json"
            if not transcript_path.exists():
                LOGGER.debug(f"No transcript found for deepfake audio {audio_path.name}")
                continue

            try:
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    transcript_data = json.load(f)

                whisper_words = parse_transcript_with_timestamps(transcript_data)

                for word_info in whisper_words:
                    if word_info['text'].lower().strip() == word_lower:
                        start_s = word_info['start_time']
                        end_s = word_info['end_time']

                        # Extract prosody for this word occurrence
                        prosody_features = prosody_utils.extract_prosody_features(
                            audio_path, start_s, end_s
                        )

                        if prosody_features:
                            candidates.append({
                                "audio_path": audio_path,
                                "start_ms": int(start_s * 1000),
                                "end_ms": int(end_s * 1000),
                                "prosody": prosody_features,
                                "word": word_info['text']
                            })
            except Exception as e:
                LOGGER.warning(f"Could not process deepfake transcript {transcript_path.name}: {e}")

    LOGGER.info(f"Found {len(candidates)} deepfake candidates for word '{word_to_find}'.")
    return candidates

# --- Tampering Operations ---

def perform_deletion(
    original_audio: AudioSegment,
    candidate: Dict[str, Any]
) -> Tuple[Optional[AudioSegment], Dict[str, Any]]:
    """Performs a deletion tamper."""
    start_ms = int(candidate['difficulty']['start_time'] * 1000)
    end_ms = int(candidate['difficulty']['end_time'] * 1000)

    # Adjust to zero-crossings for cleaner cuts
    start_ms = audio_utils.find_zero_crossing(original_audio, start_ms)
    end_ms = audio_utils.find_zero_crossing(original_audio, end_ms)

    if start_ms >= end_ms:
        LOGGER.error(f"Invalid deletion times after zero-crossing: {start_ms} >= {end_ms}")
        return None, {}

    before_segment = original_audio[:start_ms]
    after_segment = original_audio[end_ms:]

    # Combine with crossfade
    tampered_audio = before_segment.append(after_segment, crossfade=CROSSFADE_MS)

    metadata = {
        "tamper_type": "deletion",
        "keyword": candidate['word'],
        "splice_points_ms": [start_ms, end_ms],
        "original_duration_ms": len(original_audio),
        "tampered_duration_ms": len(tampered_audio),
    }
    return tampered_audio, metadata

def perform_substitution(
    original_audio: AudioSegment,
    original_audio_path: Path,
    candidate: Dict[str, Any],
    df_word_candidates: List[Dict[str, Any]]
) -> Tuple[Optional[AudioSegment], Dict[str, Any]]:
    """Performs a substitution tamper."""
    start_s = candidate['difficulty']['start_time']
    end_s = candidate['difficulty']['end_time']

    # 1. Find the best replacement word from deepfakes
    target_prosody = prosody_utils.extract_prosody_features(original_audio_path, start_s, end_s)
    if not target_prosody:
        LOGGER.error("Could not extract prosody for target word in original audio.")
        return None, {}

    prosody_candidates = [
        (i, cand['prosody']) for i, cand in enumerate(df_word_candidates)
    ]

    best_match_idx = prosody_utils.find_best_prosody_match(target_prosody, prosody_candidates)
    if best_match_idx is None:
        LOGGER.error(f"No suitable deepfake replacement found for '{candidate.get('replacement', '')}'.")
        return None, {}

    best_replacement = df_word_candidates[best_match_idx]

    # 2. Extract the replacement audio segment
    replacement_segment = audio_utils.extract_audio_segment(
        best_replacement['audio_path'],
        best_replacement['start_ms'],
        best_replacement['end_ms']
    )
    if not replacement_segment:
        LOGGER.error("Failed to extract replacement audio segment.")
        return None, {}

    # 3. Perform the replacement
    start_ms = audio_utils.find_zero_crossing(original_audio, int(start_s * 1000))
    end_ms = audio_utils.find_zero_crossing(original_audio, int(end_s * 1000))

    before = original_audio[:start_ms]
    after = original_audio[end_ms:]

    # Combine with crossfades
    result = before.append(replacement_segment, crossfade=CROSSFADE_MS)
    result = result.append(after, crossfade=CROSSFADE_MS)

    metadata = {
        "tamper_type": "substitution",
        "original_keyword": candidate['word'],
        "replacement_keyword": candidate.get('replacement', ''),
        "splice_points_ms": [start_ms, start_ms + len(replacement_segment)],
        "prosody_comparison": {
            "target_prosody": target_prosody,
            "replacement_prosody": best_replacement['prosody'],
            "distance": prosody_utils.calculate_prosody_distance(target_prosody, best_replacement['prosody'])
        },
        "replacement_source": str(best_replacement['audio_path'].name),
    }
    return result, metadata

def perform_insertion(
    original_audio: AudioSegment,
    original_audio_path: Path,
    candidate: Dict[str, Any],
    df_word_candidates: List[Dict[str, Any]]
) -> Tuple[Optional[AudioSegment], Dict[str, Any]]:
    """Performs an insertion tamper."""
    # Insertion point is at the end of the anchor word
    insert_s = candidate['difficulty']['end_time']

    # 1. Find best word from deepfakes based on context prosody
    context_prosody = prosody_utils.extract_prosody_features(original_audio_path, max(0, insert_s - 0.1), insert_s + 0.1)
    if not context_prosody:
        LOGGER.error("Could not extract context prosody from original audio.")
        return None, {}

    prosody_candidates = [(i, cand['prosody']) for i, cand in enumerate(df_word_candidates)]
    best_match_idx = prosody_utils.find_best_prosody_match(context_prosody, prosody_candidates)

    if best_match_idx is None:
        LOGGER.error(f"No suitable deepfake word found for insertion.")
        return None, {}

    best_insertion = df_word_candidates[best_match_idx]

    # 2. Extract the audio segment to insert
    insert_segment = audio_utils.extract_audio_segment(
        best_insertion['audio_path'],
        best_insertion['start_ms'],
        best_insertion['end_ms']
    )
    if not insert_segment:
        LOGGER.error("Failed to extract insertion audio segment.")
        return None, {}

    # Add a tiny silence for better separation
    insert_segment = AudioSegment.silent(duration=20) + insert_segment

    # 3. Perform the insertion
    insert_ms = audio_utils.find_zero_crossing(original_audio, int(insert_s * 1000))

    before = original_audio[:insert_ms]
    after = original_audio[insert_ms:]

    result = before.append(insert_segment, crossfade=CROSSFADE_MS)
    result = result.append(after, crossfade=0) # No crossfade into the original stream

    metadata = {
        "tamper_type": "insertion",
        "inserted_keyword": candidate.get('word', 'not'),
        "anchor_keyword": candidate['word'],
        "splice_points_ms": [insert_ms, insert_ms + len(insert_segment)],
        "prosody_comparison": {
            "context_prosody": context_prosody,
            "insertion_prosody": best_insertion['prosody'],
            "distance": prosody_utils.calculate_prosody_distance(context_prosody, best_insertion['prosody'])
        },
        "insertion_source": str(best_insertion['audio_path'].name),
    }
    return result, metadata


# --- Main Orchestrator ---

def process_candidates(
    candidate_path: Path,
    original_audio_dir: Path,
    deepfake_base_dir: Path,
    output_dir: Path,
    tamper_id_counters: Dict[str, int],
    max_tampers: int
):
    """Processes a single candidate JSON file to generate tampers."""
    try:
        with open(candidate_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        LOGGER.error(f"Could not read or parse candidate file {candidate_path.name}: {e}")
        return

    source_id = Path(data['source_audio']).stem
    original_audio_path = original_audio_dir / f"{source_id}.wav"
    if not original_audio_path.exists():
        LOGGER.error(f"Original audio not found: {original_audio_path}")
        return

    try:
        original_audio = AudioSegment.from_file(original_audio_path)
    except Exception as e:
        LOGGER.error(f"Could not load original audio {original_audio_path}: {e}")
        return

    # Find all associated deepfake directories for this source
    # e.g., df_sub096 -> find all df_sub096* directories
    base_id = '_'.join(source_id.split('_')[:2]) # e.g., df_sub096
    subject_id = source_id.split('_')[1] if len(source_id.split('_')) > 1 else source_id.split('_')[0]

    # Try both patterns to find deepfake directories
    deepfake_audio_dirs = list(DEEPFAKE_ROOT_DIR.glob(f"{subject_id}/deepfake/{base_id}*"))
    if not deepfake_audio_dirs:
        deepfake_audio_dirs = list(DEEPFAKE_ROOT_DIR.glob(f"*/{subject_id}/deepfake/{base_id}*"))
    if not deepfake_audio_dirs:
        LOGGER.warning(f"No deepfake audio directories found for source {source_id}")
        return

    # Limit candidates to process
    candidates_to_process = data.get('candidates', [])
    if len(candidates_to_process) > max_tampers:
        # Simple strategy: take top N by score
        candidates_to_process.sort(key=lambda c: c['difficulty']['score'])
        candidates_to_process = candidates_to_process[:max_tampers]

    # --- Pre-fetch all needed deepfake words for this file ---
    df_word_cache = {}
    words_to_find = set()
    for cand in candidates_to_process:
        if cand['tamper_type'] == 'substitution':
            words_to_find.add(cand.get('replacement', ''))
        elif cand['tamper_type'] == 'insertion':
            # For negation insertions, typically "not"
            words_to_find.add("not")

    for word in words_to_find:
        if word:
            df_word_cache[word] = find_deepfake_word_candidates(word, deepfake_audio_dirs)

    # --- Process each candidate ---
    for cand in candidates_to_process:
        tamper_type = cand['tamper_type']
        difficulty = cand['difficulty']['level'][0].upper() # E, M, H
        keyword = cand['word'].replace(" ", "_")

        # Generate unique tamper ID
        tamper_id_counters[source_id] += 1
        tamper_id = f"T{tamper_id_counters[source_id]:03d}"

        output_filename_base = f"{source_id}_{tamper_id}_{difficulty}_{tamper_type.upper()[:3]}_{keyword}"
        output_audio_path = output_dir / 'tampered' / f"{output_filename_base}.wav"
        output_meta_path = output_dir / 'metadata' / f"{output_filename_base}.json"

        output_audio_path.parent.mkdir(exist_ok=True, parents=True)
        output_meta_path.parent.mkdir(exist_ok=True, parents=True)

        tampered_audio, metadata = None, None

        try:
            if tamper_type == 'deletion':
                tampered_audio, metadata = perform_deletion(original_audio, cand)

            elif tamper_type == 'substitution':
                replacement_word = cand.get('replacement')
                if replacement_word and df_word_cache.get(replacement_word):
                    tampered_audio, metadata = perform_substitution(
                        original_audio, original_audio_path, cand, df_word_cache[replacement_word]
                    )
                else:
                    LOGGER.warning(f"Skipping substitution for '{keyword}', no replacement candidates found.")

            elif tamper_type == 'insertion':
                # For negation insertions, typically inserting "not"
                word_to_add = "not"
                if df_word_cache.get(word_to_add):
                    tampered_audio, metadata = perform_insertion(
                        original_audio, original_audio_path, cand, df_word_cache[word_to_add]
                    )
                else:
                    LOGGER.warning(f"Skipping insertion for '{keyword}', no insertion candidates found.")

            # Save results if successful
            if tampered_audio and metadata:
                tampered_audio.export(output_audio_path, format="wav")

                final_metadata = {
                    "tamper_filename": output_audio_path.name,
                    "source_audio": data['source_audio'],
                    "candidate_details": cand,
                    "tamper_details": metadata
                }
                with open(output_meta_path, 'w', encoding='utf-8') as f:
                    json.dump(final_metadata, f, indent=2)

                LOGGER.info(f"Successfully created tamper: {output_audio_path.name}")

        except Exception as e:
            LOGGER.error(f"Failed to process candidate for '{keyword}' in {source_id}: {e}", exc_info=True)


def main():
    """Main function to run the audio tampering script."""
    parser = argparse.ArgumentParser(description="Generate tampered audio files.")
    parser.add_argument(
        "--candidate_dir", type=Path, default="output/metadata",
        help="Directory with candidate JSON files from 03_identify.py."
    )
    parser.add_argument(
        "--original_audio_dir", type=Path, default="output/preprocessed",
        help="Directory with original preprocessed audio files."
    )
    parser.add_argument(
        "--deepfake_base_dir", type=Path, default="data",
        help="Base directory for deepfake data (e.g., 'data/sub096/deepfake')."
    )
    parser.add_argument(
        "--output_dir", type=Path, default="output",
        help="Base directory to save tampered audio and metadata."
    )
    parser.add_argument(
        "--max_tampers_per_file", type=int, default=10,
        help="Maximum number of tampers to generate per original audio file."
    )
    parser.add_argument(
        "--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level."
    )
    args = parser.parse_args()

    LOGGER.setLevel(args.log_level)
    LOGGER.info("--- Starting Audio Tampering Script ---")

    candidate_files = sorted(list(args.candidate_dir.glob("*_candidates.json")))
    if not candidate_files:
        LOGGER.error(f"No candidate files found in {args.candidate_dir}")
        return

    # Initialize tamper ID counters for each source file
    tamper_id_counters = collections.defaultdict(int)

    progress_bar = tqdm(candidate_files, desc="Processing files")
    for candidate_path in progress_bar:
        source_id = candidate_path.name.replace("_candidates.json", "")
        progress_bar.set_postfix({"file": source_id})
        process_candidates(
            candidate_path,
            args.original_audio_dir,
            args.deepfake_base_dir,
            args.output_dir,
            tamper_id_counters,
            args.max_tampers_per_file
        )

    LOGGER.info("--- Audio Tampering Script Finished ---")

if __name__ == "__main__":
    main()
