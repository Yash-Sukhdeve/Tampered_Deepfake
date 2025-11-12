"""
Audio processing utilities for the audio tampering pipeline.

This module provides helper functions for audio splicing, segment extraction,
zero-crossing detection, and normalization using PyDub and librosa.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import librosa
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_audio_segment(
    audio_path: Path,
    start_ms: int,
    end_ms: int
) -> Optional[AudioSegment]:
    """
    Extracts a segment from an audio file.

    Args:
        audio_path: Path to the audio file.
        start_ms: Start time in milliseconds.
        end_ms: End time in milliseconds.

    Returns:
        An AudioSegment object containing the extracted segment, or None if extraction fails.
    """
    if not audio_path.exists():
        logging.error(f"Audio file not found: {audio_path}")
        return None

    try:
        audio = AudioSegment.from_file(audio_path)

        # Validate time boundaries
        if start_ms < 0:
            start_ms = 0
        if end_ms > len(audio):
            end_ms = len(audio)
        if start_ms >= end_ms:
            logging.error(f"Invalid time range: start_ms ({start_ms}) >= end_ms ({end_ms})")
            return None

        segment = audio[start_ms:end_ms]
        return segment

    except CouldntDecodeError as e:
        logging.error(f"Could not decode audio file {audio_path}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error extracting audio segment from {audio_path}: {e}")
        return None


def normalize_audio(
    audio_segment: AudioSegment,
    target_dbfs: float = -20.0
) -> AudioSegment:
    """
    Normalizes an AudioSegment to a target dBFS level.

    Args:
        audio_segment: The AudioSegment to normalize.
        target_dbfs: Target dBFS level (default: -20.0).

    Returns:
        Normalized AudioSegment.
    """
    try:
        change_in_dBFS = target_dbfs - audio_segment.dBFS
        normalized = audio_segment.apply_gain(change_in_dBFS)
        return normalized
    except Exception as e:
        logging.warning(f"Error normalizing audio: {e}. Returning original segment.")
        return audio_segment


def find_zero_crossing(
    audio_segment: AudioSegment,
    position_ms: int,
    search_window_ms: int = 10
) -> int:
    """
    Finds the nearest zero-crossing point near a given position.

    Zero-crossings are points where the audio waveform crosses the zero amplitude line,
    making them ideal splice points to avoid clicks and pops.

    Args:
        audio_segment: The AudioSegment to analyze.
        position_ms: Target position in milliseconds.
        search_window_ms: Search window size in milliseconds (default: 10ms).

    Returns:
        Adjusted position in milliseconds at the nearest zero-crossing.
    """
    try:
        # Convert AudioSegment to numpy array for librosa analysis
        samples = np.array(audio_segment.get_array_of_samples())

        # Handle stereo audio - convert to mono for zero-crossing detection
        if audio_segment.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)

        # Convert position from milliseconds to sample index
        sample_rate = audio_segment.frame_rate
        position_samples = int(position_ms * sample_rate / 1000)
        search_window_samples = int(search_window_ms * sample_rate / 1000)

        # Define search range
        start_search = max(0, position_samples - search_window_samples)
        end_search = min(len(samples), position_samples + search_window_samples)

        # Find zero crossings in the search window
        search_region = samples[start_search:end_search]
        zero_crossings = librosa.zero_crossings(search_region, pad=False)

        # Find the zero crossing closest to the target position
        if np.any(zero_crossings):
            # Get indices of zero crossings
            zc_indices = np.where(zero_crossings)[0]
            # Find the one closest to the middle of our search region (the original position)
            target_in_region = position_samples - start_search
            closest_zc_idx = zc_indices[np.argmin(np.abs(zc_indices - target_in_region))]
            # Convert back to absolute sample position
            adjusted_position_samples = start_search + closest_zc_idx
            # Convert back to milliseconds
            adjusted_position_ms = int(adjusted_position_samples * 1000 / sample_rate)
            return adjusted_position_ms
        else:
            # No zero crossing found in window, return original position
            logging.debug(f"No zero crossing found within {search_window_ms}ms window at {position_ms}ms")
            return position_ms

    except Exception as e:
        logging.warning(f"Error finding zero crossing: {e}. Returning original position.")
        return position_ms


def splice_audio(
    original_path: Path,
    insert_segment: AudioSegment,
    insert_position_ms: int,
    crossfade_ms: int,
    output_path: Path
) -> bool:
    """
    Splices an audio segment into an original audio file with crossfading.

    This function inserts a segment at a specified position, replacing any audio
    at that location and applying crossfades at the splice points for smooth transitions.

    Args:
        original_path: Path to the original audio file.
        insert_segment: AudioSegment to insert.
        insert_position_ms: Position in milliseconds where to insert the segment.
        crossfade_ms: Duration of crossfade in milliseconds at splice points.
        output_path: Path where the spliced audio will be saved.

    Returns:
        True if splicing was successful, False otherwise.
    """
    try:
        # Load original audio
        original = AudioSegment.from_file(original_path)

        # Validate insert position
        if insert_position_ms < 0 or insert_position_ms > len(original):
            logging.error(f"Invalid insert position: {insert_position_ms}ms (audio length: {len(original)}ms)")
            return False

        # Ensure insert_segment matches original's properties
        if insert_segment.frame_rate != original.frame_rate:
            insert_segment = insert_segment.set_frame_rate(original.frame_rate)
        if insert_segment.channels != original.channels:
            insert_segment = insert_segment.set_channels(original.channels)

        # Calculate end position of insert
        insert_end_ms = insert_position_ms + len(insert_segment)

        # Split original audio into three parts: before, (replaced section), after
        before = original[:insert_position_ms]
        after = original[insert_end_ms:] if insert_end_ms < len(original) else AudioSegment.empty()

        # Apply crossfades if possible
        if len(before) > 0 and crossfade_ms > 0:
            # Crossfade between 'before' and 'insert_segment'
            crossfade_ms_actual = min(crossfade_ms, len(before), len(insert_segment))
            if crossfade_ms_actual > 0:
                spliced = before.append(insert_segment, crossfade=crossfade_ms_actual)
            else:
                spliced = before + insert_segment
        else:
            spliced = before + insert_segment

        if len(after) > 0 and crossfade_ms > 0:
            # Crossfade between 'spliced' (which includes insert_segment) and 'after'
            crossfade_ms_actual = min(crossfade_ms, len(insert_segment), len(after))
            if crossfade_ms_actual > 0:
                spliced = spliced.append(after, crossfade=crossfade_ms_actual)
            else:
                spliced = spliced + after
        else:
            spliced = spliced + after

        # Export spliced audio
        output_path.parent.mkdir(parents=True, exist_ok=True)
        spliced.export(output_path, format="wav", parameters=["-acodec", "pcm_s16le"])

        logging.info(f"Successfully spliced audio and saved to {output_path}")
        return True

    except Exception as e:
        logging.error(f"Error splicing audio: {e}")
        return False


def replace_audio_segment(
    original_path: Path,
    start_ms: int,
    end_ms: int,
    replacement_segment: AudioSegment,
    crossfade_ms: int,
    output_path: Path
) -> bool:
    """
    Replaces a segment of audio with a replacement segment, using crossfades.

    Args:
        original_path: Path to the original audio file.
        start_ms: Start time of the segment to replace (milliseconds).
        end_ms: End time of the segment to replace (milliseconds).
        replacement_segment: AudioSegment to insert as replacement.
        crossfade_ms: Crossfade duration in milliseconds.
        output_path: Path where the result will be saved.

    Returns:
        True if successful, False otherwise.
    """
    try:
        # Load original audio
        original = AudioSegment.from_file(original_path)

        # Extract parts before and after the segment to be replaced
        before = original[:start_ms]
        after = original[end_ms:]

        # Ensure replacement segment matches original's properties
        if replacement_segment.frame_rate != original.frame_rate:
            replacement_segment = replacement_segment.set_frame_rate(original.frame_rate)
        if replacement_segment.channels != original.channels:
            replacement_segment = replacement_segment.set_channels(original.channels)

        # Combine with crossfades
        if len(before) > 0 and crossfade_ms > 0:
            crossfade_ms_actual = min(crossfade_ms, len(before), len(replacement_segment))
            result = before.append(replacement_segment, crossfade=crossfade_ms_actual)
        else:
            result = before + replacement_segment

        if len(after) > 0 and crossfade_ms > 0:
            crossfade_ms_actual = min(crossfade_ms, len(replacement_segment), len(after))
            result = result.append(after, crossfade=crossfade_ms_actual)
        else:
            result = result + after

        # Export
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.export(output_path, format="wav", parameters=["-acodec", "pcm_s16le"])

        logging.info(f"Successfully replaced audio segment and saved to {output_path}")
        return True

    except Exception as e:
        logging.error(f"Error replacing audio segment: {e}")
        return False
