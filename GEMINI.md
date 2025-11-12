# Tampered Deepfake Audio Pipeline

This project is a research pipeline for generating semantically-altered tampered audio by injecting deepfake word segments into original speech recordings. The goal is to create challenging data for deepfake detection research.

## Project Overview

The pipeline consists of four main stages:

1.  **Preprocessing**: Converts raw audio files to a standardized format (16kHz mono WAV) and normalizes volume.
2.  **Analysis**: Transcribes the audio using Whisper ASR and then uses the Montreal Forced Aligner (MFA) to obtain phoneme-level timestamps.
3.  **Candidate Identification**: Identifies potential tampering points in the audio using NLP techniques (spaCy) and calculates a difficulty score for each candidate based on prosodic analysis (librosa).
4.  **Audio Tampering**: Performs the actual audio manipulation, including deletions, insertions, and substitutions. For insertions and substitutions, it uses a prosody-aware search to find the best-matching deepfake word from a corpus.

## Building and Running

### Prerequisites

*   Python 3.8+
*   CUDA-capable GPU (for Whisper)
*   ffmpeg (for PyDub)
*   Montreal Forced Aligner

### Installation

1.  Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Download spaCy model:
    ```bash
    python -m spacy download en_core_web_sm
    ```
3.  Install Montreal Forced Aligner:
    ```bash
    conda install -c conda-forge montreal-forced-aligner
    ```
4.  Download MFA models:
    ```bash
    mfa model download acoustic english
    mfa model download dictionary english_us_arpa
    ```

### Running the Pipeline

To run the entire pipeline:
```bash
python run_pipeline.py --all
```

To run a specific step:
```bash
# Step 1: Preprocessing
python run_pipeline.py --step 1

# Step 2: Analysis (Whisper + MFA)
python run_pipeline.py --step 2 --whisper_model large-v3

# Step 3: Candidate Identification
python run_pipeline.py --step 3

# Step 4: Audio Tampering
python run_pipeline.py --step 4 --max_tampers 10
```

## Development Conventions

*   The project is organized into a modular pipeline, with each stage implemented as a separate Python script in the `src` directory.
*   The `run_pipeline.py` script orchestrates the execution of the pipeline stages.
*   Utility functions for audio processing, NLP, and prosody analysis are located in the `src/utils` directory.
*   The project uses `argparse` for command-line argument parsing.
*   Logging is used throughout the scripts to provide information about the pipeline's execution.
*   The `tqdm` library is used to display progress bars for long-running operations.
