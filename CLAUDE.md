# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project focused on deepfake detection and audio/video tampering analysis. The project is in its initial setup phase.

## Research Requirements

**Scientific Rigor**: All claims, methods, and findings must be supported by scientific evidence from peer-reviewed sources. When implementing detection algorithms or discussing deepfake techniques:
- Cite relevant papers and sources
- Reference established methodologies from the literature
- Validate findings against published benchmarks where applicable

## Related Projects Context

This project exists alongside other deepfake-related work on this system:
- DeepFake Audio Collection: Dataset of deepfake audio samples
- Deep-Live-Cam: Real-time deepfake generation
- denoised_data & denoiser_docker: Audio preprocessing pipeline
- Neurotec_Biometric_13_1_SDK: Biometric verification SDK (read-only except for licensing issues)

## Development Setup

**Language**: Python 3.8+

**Installation**:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
# MFA: conda install -c conda-forge montreal-forced-aligner
```

**Hardware Requirements**:
- GPU: RTX 4080 16GB (for Whisper ASR)
- CPU: Ryzen 9 5900X 24-threads (for parallel MFA processing)
- RAM: 62GB

## Pipeline Architecture

**Objective**: Create tampered audio by injecting deepfake word segments into original audio to alter meaning, generating Easy/Medium/Hard difficulty samples for detection research.

### Pipeline Stages

1. **Preprocessing** (`src/01_preprocess.py`)
   - Convert m4a to 16kHz mono WAV (match deepfake format)
   - Normalize audio levels
   - Validate file integrity

2. **Analysis** (`src/02_analyze.py`)
   - **GPU Task**: Whisper transcription with word timestamps (~10s for all files)
   - **CPU Task**: Montreal Forced Aligner for phoneme-level timestamps (parallel, 24 threads)
   - Output: JSON (Whisper) + TextGrid (MFA) files

3. **Candidate Identification** (`src/03_identify.py`)
   - Parse transcripts with spaCy NLP
   - Apply three alteration strategies:
     - **Semantic Inverter**: Add/remove negations ("not", "never")
     - **Quantifier Flip**: all↔some, many↔few, is↔is not
     - **Simple Deletion**: Remove adjectives/adverbs
   - Calculate difficulty scores: Type Weight + Acoustic Penalty + Rhythm Penalty
   - Label as Easy (<4.0), Medium (4.0-8.0), Hard (>8.0)

4. **Tampering** (`src/04_tamper.py`)
   - **Insertion**: Exhaustive prosody search across deepfake instances (pitch, energy, duration matching)
   - **Deletion**: Phoneme-boundary splicing using MFA timestamps
   - Apply 2-5ms crossfade at splice points (PyDub)
   - Volume normalization

5. **Metadata Generation**
   - JSON per tampered file: original/tampered text, timestamps, difficulty breakdown, prosody metrics

### File Naming Convention

```
[SourceID]_[TamperID]_[Difficulty]_[TamperType]_[Keyword].wav

Examples:
- df_sub096_LP1_T001_H_DEL_always.wav
- df_sub096_T002_M_INS_not.wav
- df_sub096_LP2_T003_E_SUB_some.wav
```

### Directory Structure

```
data/
├── original/          # Original m4a files
├── deepfake/          # YourTTS/XTTS generated WAVs (16kHz mono)
└── scripts/           # PDF transcripts (df_script.pdf, LP_script.pdf)

output/
├── preprocessed/      # Converted 16kHz mono WAVs
├── transcripts/       # Whisper JSON + MFA TextGrids
├── tampered/          # Final tampered audio files
└── metadata/          # JSON metadata per tampered file

src/
├── 01_preprocess.py
├── 02_analyze.py
├── 03_identify.py
├── 04_tamper.py
└── utils/
    ├── prosody.py     # Prosody analysis with librosa
    ├── audio.py       # Audio splicing helpers
    └── nlp.py         # Text processing utilities
```

## Key Implementation Details

**Prosody Matching**: Exhaustive search algorithm
- With only ~2 minutes of deepfake audio, exhaustive search is computationally negligible (<1ms) but guarantees optimal quality
- Extract pitch (F0), energy (RMS), duration for all instances
- Calculate Euclidean distance to target context
- Select best match

**Phoneme-Boundary Splicing**:
- Use MFA TextGrid for precise phoneme timestamps
- Splice at phoneme boundaries to avoid co-articulation artifacts
- Apply minimal crossfade (2-5ms) to eliminate clicks

**Parallelization Strategy**:
- GPU: Whisper transcription (batch all 12 files)
- CPU: MFA alignment (24 parallel processes)
- CPU: Tampering operations (parallel across 24 threads)

## Scientific Foundation

**Cited Methodologies**:
- Forced alignment for speech synthesis: McAuliffe et al. (2017), Montreal Forced Aligner
- Prosodic feature extraction: librosa library (McFee et al., 2015)
- Audio tampering detection: surveys by Malik (2022), Almutairi & Elgibreen (2022)

## Notes for Future Development

- Ensure all datasets are properly documented with provenance
- Track experimental configurations for reproducibility
- Consider ethical implications of deepfake detection research
- Maintain clear separation between training, validation, and test datasets
