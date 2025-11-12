# Tampered Deepfake Audio Pipeline

An automated research pipeline for generating semantically-altered tampered audio by injecting deepfake word segments into original speech recordings. Created for deepfake detection research.

## Overview

This pipeline creates tampered audio samples with varying difficulty levels (Easy/Medium/Hard) by:
- **Deletions**: Removing adjectives/adverbs to change emphasis
- **Insertions**: Adding negations ("not", "never") to reverse meaning
- **Substitutions**: Swapping quantifiers (all↔some, many↔few) to alter scope

All operations use prosody-aware splicing to create natural-sounding tampers that challenge detection algorithms.

## Architecture

### Pipeline Stages

1. **Preprocessing** ([01_preprocess.py](src/01_preprocess.py))
   - Converts m4a → 16kHz mono WAV
   - Normalizes to -20dBFS
   - Validates audio properties

2. **Analysis** ([02_analyze.py](src/02_analyze.py))
   - Whisper ASR (GPU): Word-level transcription
   - Montreal Forced Aligner (CPU): Phoneme-level timestamps
   - Parallel processing: GPU + 24 CPU threads

3. **Candidate Identification** ([03_identify.py](src/03_identify.py))
   - NLP analysis (spaCy): Identifies tamper candidates
   - Prosody scoring: Calculates difficulty based on acoustic properties
   - Labels: Easy (<4.0), Medium (4.0-8.0), Hard (>8.0)

4. **Audio Tampering** ([04_tamper.py](src/04_tamper.py))
   - Exhaustive prosody search: Finds best-matching deepfake words
   - Phoneme-boundary splicing: Clean cuts at zero-crossings
   - Crossfading: 2-5ms fades for smooth transitions

### Directory Structure

```
Tampered_Deepfake/
├── data/
│   ├── original/          # Original m4a files
│   ├── deepfake/          # YourTTS/XTTS deepfake audio
│   └── scripts/           # PDF transcripts
├── output/
│   ├── preprocessed/      # 16kHz mono WAVs
│   ├── transcripts/       # Whisper JSON + MFA TextGrids
│   ├── tampered/          # Final tampered audio files
│   └── metadata/          # Candidate & tamper metadata JSON
├── src/
│   ├── 01_preprocess.py
│   ├── 02_analyze.py
│   ├── 03_identify.py
│   ├── 04_tamper.py
│   └── utils/
│       ├── prosody.py     # Prosody analysis (librosa)
│       ├── audio.py       # Audio processing (PyDub)
│       └── nlp.py         # Text analysis (spaCy)
├── run_pipeline.py        # Master orchestration script
├── requirements.txt
└── CLAUDE.md             # Development guide
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (for Whisper)
- ffmpeg (for PyDub)
- Montreal Forced Aligner

### Setup

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Download spaCy model
python -m spacy download en_core_web_sm

# 3. Install Montreal Forced Aligner
conda install -c conda-forge montreal-forced-aligner

# 4. Download MFA models
mfa model download acoustic english
mfa model download dictionary english_us_arpa
```

## Usage

### Run Complete Pipeline

```bash
python run_pipeline.py --all
```

### Run Individual Steps

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

### Custom Paths

```bash
# Preprocessing
python src/01_preprocess.py --input_dir data/original --output_dir output/preprocessed

# Analysis
python src/02_analyze.py --input_dir output/preprocessed --output_dir output/transcripts

# Identification
python src/03_identify.py --audio_dir output/preprocessed --transcript_dir output/transcripts

# Tampering
python src/04_tamper.py --candidate_dir output/metadata --original_audio_dir output/preprocessed
```

## Output Format

### File Naming Convention

```
{SourceID}_T{TamperID}_{Difficulty}_{TamperType}_{Keyword}.wav

Examples:
- df_sub096_LP1_T001_H_DEL_always.wav    # Hard deletion
- df_sub096_T002_M_INS_not.wav           # Medium insertion
- df_sub096_LP2_T003_E_SUB_some.wav      # Easy substitution
```

### Metadata Structure

Each tampered file has accompanying JSON metadata:

```json
{
  "tamper_filename": "df_sub096_T001_H_DEL_always.wav",
  "source_audio": "df_sub096.wav",
  "candidate_details": {
    "word": "always",
    "type": "adjective_adverb",
    "difficulty": {
      "score": 8.75,
      "level": "hard",
      "type_weight": 1.0,
      "acoustic_penalty": 2.25,
      "rhythm_penalty": 5.5
    }
  },
  "tamper_details": {
    "tamper_type": "deletion",
    "splice_points_ms": [4530, 5080]
  }
}
```

## Hardware Requirements

**Tested Configuration:**
- **CPU**: AMD Ryzen 9 5900X (24 threads, 12 cores)
- **GPU**: NVIDIA RTX 4080 16GB
- **RAM**: 62GB
- **Storage**: 304GB available

**Minimum:**
- CPU: 8+ cores for MFA
- GPU: 8GB VRAM for Whisper large models
- RAM: 16GB
- Storage: 50GB

## Scientific Foundation

### Cited Methodologies

- **Forced Alignment**: McAuliffe et al. (2017), Montreal Forced Aligner
- **Prosodic Features**: librosa library (McFee et al., 2015)
- **Audio Tampering Detection**: Malik (2022), Almutairi & Elgibreen (2022)

### Key Design Decisions

1. **Exhaustive Prosody Search**: With only ~2 min of deepfake audio per speaker, exhaustive search (<1ms overhead) guarantees optimal quality over "first match" heuristics.

2. **Phoneme-Boundary Splicing**: MFA provides phoneme-level timestamps enabling cuts at natural linguistic boundaries, avoiding co-articulation artifacts.

3. **Difficulty Scoring**:
   ```
   Score = Type_Weight + Acoustic_Penalty + Rhythm_Penalty
   - Type: deletion (1.0), insertion (2.0), substitution (2.5)
   - Acoustic: Prosodic discontinuity with neighbors
   - Rhythm: Duration penalty (ms / 100)
   ```

## Performance

**Expected Processing Time** (12 audio files, ~2 min each):
- Preprocessing: ~30 seconds
- Whisper (large-v3): ~10 seconds (GPU)
- MFA: ~2 minutes (24 parallel jobs)
- Candidate Identification: ~1 minute
- Tampering (10 per file): ~5 minutes

**Total**: ~8-10 minutes for complete pipeline

## Troubleshooting

### Common Issues

**1. ffmpeg not found**
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

**2. MFA command not found**
```bash
# Verify installation
which mfa

# Add to PATH if needed
export PATH="$HOME/anaconda3/bin:$PATH"
```

**3. CUDA out of memory (Whisper)**
```bash
# Use smaller model
python run_pipeline.py --step 2 --whisper_model medium
```

**4. spaCy model not found**
```bash
python -m spacy download en_core_web_sm
```

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{tampered_deepfake_pipeline,
  title={Tampered Deepfake Audio Pipeline},
  author={Your Name},
  year={2025},
  url={https://github.com/Yash-Sukhdeve/Tampered-Deepfake}
}
```

## License

This project is for research purposes only. Ensure ethical use and obtain proper consent for all audio data.

## Acknowledgments

- **Whisper**: OpenAI's robust speech recognition
- **Montreal Forced Aligner**: Precise phoneme-level alignment
- **librosa**: Comprehensive audio analysis
- **PyDub**: Simple audio manipulation
- **spaCy**: Industrial-strength NLP

---

**Contact**: [Your Email]
**Project**: Tampered Deepfake Detection Research
**Institution**: AVBHAC Lab, Clarkson University(NY)
