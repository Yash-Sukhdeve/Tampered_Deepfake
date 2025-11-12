"""
A module for prosody analysis using the librosa library.

This module provides functions to extract prosodic features (pitch, energy)
from audio files, calculate the distance between feature sets, and find the
best matching audio segment from a list of candidates based on prosody.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import librosa
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_prosody_features(
    audio_path: Path,
    start_time: float,
    end_time: float
) -> Optional[Dict[str, float]]:
    """
    Extracts prosodic features from a specified segment of an audio file.

    The function loads a segment of an audio file and computes features related
    to pitch (F0) and energy (RMS).

    Args:
        audio_path: The path to the audio file.
        start_time: The start time of the segment in seconds.
        end_time: The end time of the segment in seconds.

    Returns:
        A dictionary containing the mean and standard deviation for pitch and
        energy, along with the duration. Returns None if the audio segment
        cannot be loaded or processed.
        Keys: 'pitch_mean', 'pitch_std', 'energy_mean', 'energy_std', 'duration'.
    """
    if not audio_path.exists():
        logging.error(f"Audio file not found at: {audio_path}")
        return None

    duration = end_time - start_time
    if duration <= 0:
        logging.warning(f"Invalid time segment: start_time {start_time} must be less than end_time {end_time}.")
        return None

    try:
        # Load the specified audio segment
        y, sr = librosa.load(
            path=audio_path,
            offset=start_time,
            duration=duration
        )

        if len(y) == 0:
            logging.warning(f"No audio data loaded for segment {start_time}-{end_time} in {audio_path}.")
            return {
                'pitch_mean': 0.0,
                'pitch_std': 0.0,
                'energy_mean': 0.0,
                'energy_std': 0.0,
                'duration': duration
            }

        # 1. Extract pitch (F0) using pyin
        # f0 is an array of fundamental frequency estimates
        # voiced_flag is a boolean array indicating voiced/unvoiced frames
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

        # Use nan-aware functions to calculate stats only on voiced frames
        pitch_mean = np.nanmean(f0) if np.any(voiced_flag) else 0.0
        pitch_std = np.nanstd(f0) if np.any(voiced_flag) else 0.0

        # 2. Extract energy (Root-Mean-Square)
        rms = librosa.feature.rms(y=y)[0]
        energy_mean = np.mean(rms)
        energy_std = np.std(rms)

        return {
            'pitch_mean': float(pitch_mean),
            'pitch_std': float(pitch_std),
            'energy_mean': float(energy_mean),
            'energy_std': float(energy_std),
            'duration': float(duration)
        }

    except Exception as e:
        logging.error(f"Failed to process audio segment from {audio_path}: {e}")
        return None

def calculate_prosody_distance(features1: Dict[str, float], features2: Dict[str, float]) -> float:
    """
    Calculates a normalized Euclidean distance between two prosody feature sets.

    To ensure that features with different scales (e.g., pitch in Hz vs. RMS
    energy) contribute fairly to the distance, the features are first
    standardized (Z-score normalized) before calculating the Euclidean distance.

    Args:
        features1: The first dictionary of prosodic features.
        features2: The second dictionary of prosodic features.

    Returns:
        A single float representing the normalized distance score.
    """
    # Define the feature keys to be used in the distance calculation
    feature_keys = ['pitch_mean', 'pitch_std', 'energy_mean', 'energy_std']

    # Create feature vectors from the input dictionaries
    vec1 = np.array([features1.get(key, 0.0) for key in feature_keys])
    vec2 = np.array([features2.get(key, 0.0) for key in feature_keys])

    # Combine vectors into a 2xN array for normalization
    data = np.vstack([vec1, vec2])

    # Calculate mean and std dev for each feature column
    # Add a small epsilon to std dev to avoid division by zero if features are identical
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0) + 1e-9

    # Apply Z-score normalization
    normalized_data = (data - mean) / std

    # Calculate Euclidean distance on the normalized vectors
    distance = np.linalg.norm(normalized_data[0] - normalized_data[1])

    return float(distance)

def find_best_prosody_match(
    target_features: Dict[str, float],
    candidates: List[Tuple[int, Dict[str, float]]]
) -> Optional[int]:
    """
    Finds the candidate with the minimum prosodic distance to a target.

    Args:
        target_features: The feature dictionary of the target segment.
        candidates: A list of tuples, where each tuple contains an
                    integer index and a candidate's feature dictionary.

    Returns:
        The integer index of the best-matching candidate. Returns None if the
        candidate list is empty.
    """
    if not candidates:
        return None

    min_distance = float('inf')
    best_match_index = None

    for index, candidate_features in candidates:
        distance = calculate_prosody_distance(target_features, candidate_features)
        if distance < min_distance:
            min_distance = distance
            best_match_index = index

    return best_match_index
