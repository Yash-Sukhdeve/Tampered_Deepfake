"""
NLP utilities for text processing and candidate identification.

This module provides functions for identifying tampering candidates using spaCy:
- Negation candidates (add/remove "not")
- Quantifier swap candidates (all/some, many/few, etc.)
- Adjective/Adverb deletion candidates
- Transcript parsing from Whisper JSON output
"""

import spacy
from typing import List, Dict, Any

# Load the spaCy English model
# Make sure to run: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    # As a fallback, create a blank model, though it won't have POS tagging or entities
    nlp = spacy.blank("en")

def identify_negation_candidates(text: str) -> List[Dict[str, Any]]:
    """
    Identifies verbs that could have negation added or existing negations that could be removed.

    Args:
        text: The input text to analyze.

    Returns:
        A list of dictionaries, each representing a negation candidate.
        Each dictionary contains 'word', 'position' (character offset), and 'action' ('add_not' or 'remove_not').
    """
    if not text:
        return []

    doc = nlp(text)
    candidates = []
    negation_tokens = {"not", "n't"}

    for token in doc:
        # Identify verbs that don't have negation
        if token.pos_ == "VERB":
            has_negation = False
            # Check for preceding negation (e.g., "do not go", "did not see")
            for child in token.children:
                if child.dep_ == "neg" and child.text.lower() in negation_tokens:
                    has_negation = True
                    break
            # Check for attached negation (e.g., "isn't", "don't")
            if token.text.lower().endswith("n't"):
                has_negation = True

            if not has_negation:
                candidates.append({
                    "word": token.text,
                    "position": token.idx,
                    "action": "add_not"
                })
            else:
                # Identify existing negations that could be removed
                # This is a simplification; a more robust solution would analyze the full phrase
                candidates.append({
                    "word": token.text,
                    "position": token.idx,
                    "action": "remove_not"
                })
    return candidates

def identify_quantifier_candidates(text: str) -> List[Dict[str, Any]]:
    """
    Identifies quantifiers in the text that could be swapped for their opposites.

    Args:
        text: The input text to analyze.

    Returns:
        A list of dictionaries, each representing a quantifier swap candidate.
        Each dictionary contains 'word', 'replacement', and 'position' (character offset).
    """
    if not text:
        return []

    doc = nlp(text)
    candidates = []
    # Basic quantifier pairs for swapping
    quantifier_pairs = {
        "all": "some", "some": "all",
        "many": "few", "few": "many",
        "always": "never", "never": "always",
        "every": "no", "no": "every", # 'no' as in 'no one', 'no thing'
        "most": "few", "few": "most",
        "much": "little", "little": "much"
    }

    for token in doc:
        lower_text = token.text.lower()
        if lower_text in quantifier_pairs:
            candidates.append({
                "word": token.text,
                "replacement": quantifier_pairs[lower_text],
                "position": token.idx
            })
    return candidates

def identify_adjective_adverb_deletions(text: str) -> List[Dict[str, Any]]:
    """
    Identifies adjectives and adverbs that could potentially be deleted from the text.

    Args:
        text: The input text to analyze.

    Returns:
        A list of dictionaries, each representing a deletion candidate.
        Each dictionary contains 'word', 'pos_tag', and 'position' (character offset).
    """
    if not text:
        return []

    doc = nlp(text)
    candidates = []

    for token in doc:
        if token.pos_ in ["ADJ", "ADV"]:
            candidates.append({
                "word": token.text,
                "pos_tag": token.pos_,
                "position": token.idx
            })
    return candidates

def parse_transcript_with_timestamps(transcript_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parses Whisper JSON output format to extract words with their start and end timestamps.

    Handles two cases:
    1. word_timestamps=True: Uses word-level timestamps directly
    2. word_timestamps=False: Estimates word timestamps from segment timestamps

    Args:
        transcript_data: A dictionary containing the Whisper JSON output.

    Returns:
        A list of dictionaries, each representing a word with its text, start_time, and end_time.
    """
    if not transcript_data or "segments" not in transcript_data:
        return []

    parsed_words = []
    for segment in transcript_data.get("segments", []):
        # Check if word-level timestamps are available
        if "words" in segment and segment["words"]:
            # Case 1: word_timestamps=True
            for word_info in segment["words"]:
                parsed_words.append({
                    "text": word_info.get("word", "").strip(),
                    "start_time": word_info.get("start"),
                    "end_time": word_info.get("end")
                })
        else:
            # Case 2: word_timestamps=False - estimate from segment
            segment_text = segment.get("text", "").strip()
            segment_start = segment.get("start", 0.0)
            segment_end = segment.get("end", segment_start + 1.0)
            segment_duration = segment_end - segment_start

            # Split into words
            words = segment_text.split()
            if not words:
                continue

            # Estimate time per word (simple linear distribution)
            time_per_word = segment_duration / len(words)

            for i, word in enumerate(words):
                word_start = segment_start + (i * time_per_word)
                word_end = word_start + time_per_word
                parsed_words.append({
                    "text": word.strip(),
                    "start_time": word_start,
                    "end_time": word_end
                })
    return parsed_words
