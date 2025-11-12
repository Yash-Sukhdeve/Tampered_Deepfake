"""
Utility modules for the Tampered Deepfake audio pipeline.

This package contains helper functions for:
- prosody: Prosodic feature extraction and matching
- audio: Audio processing, splicing, and manipulation
- nlp: Natural language processing and text analysis
"""

from .prosody import (
    extract_prosody_features,
    calculate_prosody_distance,
    find_best_prosody_match
)

from .audio import (
    extract_audio_segment,
    normalize_audio,
    find_zero_crossing,
    splice_audio,
    replace_audio_segment
)

from .nlp import (
    identify_negation_candidates,
    identify_quantifier_candidates,
    identify_adjective_adverb_deletions,
    parse_transcript_with_timestamps
)

__all__ = [
    # Prosody
    'extract_prosody_features',
    'calculate_prosody_distance',
    'find_best_prosody_match',
    # Audio
    'extract_audio_segment',
    'normalize_audio',
    'find_zero_crossing',
    'splice_audio',
    'replace_audio_segment',
    # NLP
    'identify_negation_candidates',
    'identify_quantifier_candidates',
    'identify_adjective_adverb_deletions',
    'parse_transcript_with_timestamps',
]
