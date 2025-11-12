# Production Deployment Guide

## Quick Start

```bash
# 1. Run automated setup
./setup.sh

# 2. Verify installation
python -c "import whisper, librosa, spacy; print('✓ All imports successful')"

# 3. Run complete pipeline
python run_pipeline.py --all
```

## System Requirements

### Hardware (Recommended)

- **CPU**: 8+ cores (12+ recommended for MFA)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (for Whisper large models)
- **RAM**: 16GB minimum, 32GB+ recommended
- **Storage**: 50GB+ available space

### Tested Configuration

- **CPU**: AMD Ryzen 9 5900X (24 threads, 12 cores)
- **GPU**: NVIDIA RTX 4080 16GB
- **RAM**: 62GB
- **OS**: Ubuntu 22.04 / Linux 6.8.0

### Software

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- ffmpeg (required for audio processing)
- Montreal Forced Aligner (optional but recommended)

## Installation Steps

### 1. Clone/Setup Project

```bash
cd /path/to/Tampered_Deepfake
```

### 2. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg python3-pip git
```

**macOS:**
```bash
brew install ffmpeg python@3.11
```

### 3. Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Install spaCy model (required for full NLP)
python -m spacy download en_core_web_sm
```

### 4. Install Montreal Forced Aligner (Optional)

**For precise phoneme-level timestamps:**

```bash
# Using conda
conda create -n mfa -c conda-forge montreal-forced-aligner
conda activate mfa

# Download models
mfa model download acoustic english
mfa model download dictionary english_us_arpa
```

**Note**: MFA is optional. The pipeline works without it using Whisper segment timestamps.

### 5. Verify Installation

```bash
# Run automated setup (does all above steps)
./setup.sh

# Or manually test
python -c "
import whisper
import librosa
import pydub
import spacy
nlp = spacy.load('en_core_web_sm')
print('✓ All modules loaded successfully')
"
```

## Data Preparation

### Directory Structure

```
data/
└── sub096/
    ├── original/
    │   ├── df_sub096.m4a
    │   ├── df_sub096_LP1.m4a
    │   └── df_sub096_LP2.m4a
    └── deepfake/
        ├── df_sub096/
        │   ├── result_1.wav
        │   ├── result_2.wav
        │   └── result_3.wav
        ├── df_sub096_LP1/
        └── df_sub096_LP2/
```

### Key Requirements

1. **Original Audio**: Place in `data/{subject_id}/original/`
   - Format: m4a, wav, mp3 (will be converted)
   - Quality: High quality recordings preferred

2. **Deepfake Audio**: Place in `data/{subject_id}/deepfake/{variant}/`
   - Format: WAV, 16kHz mono preferred
   - Content: Should contain vocabulary from original scripts for substitution
   - **Important**: Longer deepfake clips with diverse vocabulary increase tampering success

## Running the Pipeline

### Full Pipeline (All Steps)

```bash
python run_pipeline.py --all
```

This runs:
1. Preprocessing (m4a → 16kHz mono WAV)
2. Analysis (Whisper ASR + MFA alignment)
3. Candidate Identification (NLP + difficulty scoring)
4. Audio Tampering (prosody-aware splicing)

### Individual Steps

```bash
# Step 1: Preprocessing only
python run_pipeline.py --step 1

# Step 2: Analysis (specify Whisper model)
python run_pipeline.py --step 2 --whisper_model large-v3

# Step 3: Candidate identification
python run_pipeline.py --step 3

# Step 4: Audio tampering (limit tampers per file)
python run_pipeline.py --step 4 --max_tampers 10
```

### Custom Paths

```bash
# Preprocessing
python src/01_preprocess.py \
  --input_dir data/sub096/original \
  --output_dir output/preprocessed

# Analysis
python src/02_analyze.py \
  --input_dir output/preprocessed \
  --output_dir output/transcripts \
  --whisper_model large-v3 \
  --mfa_jobs 24

# Identification
python src/03_identify.py \
  --audio_dir output/preprocessed \
  --transcript_dir output/transcripts \
  --output_dir output/metadata

# Tampering
python src/04_tamper.py \
  --candidate_dir output/metadata \
  --original_audio_dir output/preprocessed \
  --deepfake_base_dir data \
  --output_dir output \
  --max_tampers_per_file 10 \
  --log_level INFO
```

## Output Structure

```
output/
├── preprocessed/          # Normalized 16kHz mono WAV files
│   ├── df_sub096.wav
│   ├── df_sub096_LP1.wav
│   └── ...
├── transcripts/           # Whisper JSON transcripts
│   ├── df_sub096.json
│   └── ...
├── tampered/              # Final tampered audio files
│   ├── df_sub096_T001_M_SUB_few.wav
│   ├── df_sub096_T002_H_DEL_always.wav
│   └── ...
└── metadata/              # Candidate and tamper metadata
    ├── df_sub096_candidates.json
    ├── df_sub096_T001_M_SUB_few.json
    └── ...
```

## Performance Optimization

### GPU Utilization

**Whisper Model Selection:**
- `tiny`: Fastest, less accurate (~1-2 sec/file)
- `base`: Good balance (~3-5 sec/file)
- `small`: Better accuracy (~8-10 sec/file)
- `medium`: High accuracy (~15-20 sec/file)
- `large-v3`: Best accuracy (~30-40 sec/file)

```bash
# For production quality
python run_pipeline.py --step 2 --whisper_model large-v3

# For testing/iteration
python run_pipeline.py --step 2 --whisper_model tiny
```

### CPU Parallelization

**MFA Jobs** (if installed):
```bash
# Use all CPU cores
python run_pipeline.py --step 2 --mfa_jobs $(nproc)

# Or specify
python src/02_analyze.py --mfa_jobs 24
```

### Batch Processing

For large datasets:
```bash
# Process in batches
for subject in sub001 sub002 sub003; do
  python run_pipeline.py --all --input_dir data/$subject
done
```

## Troubleshooting

### Common Issues

**1. ffmpeg not found**
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Verify
which ffmpeg
```

**2. CUDA/GPU not detected**
```bash
# Check GPU
nvidia-smi

# Install CUDA toolkit
# Follow: https://developer.nvidia.com/cuda-downloads

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

**3. spaCy model not found**
```bash
python -m spacy download en_core_web_sm

# Verify
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('OK')"
```

**4. MFA not found (optional)**
```bash
# Install via conda
conda install -c conda-forge montreal-forced-aligner

# Verify
mfa version

# Pipeline works without MFA using Whisper timestamps
```

**5. Triton/word_timestamps error**
```
AttributeError: Cannot set attribute 'src' directly
```
**Solution**: word_timestamps are disabled in the code due to triton compatibility. MFA provides better phoneme-level timestamps anyway.

**6. No tampered files generated**

Check:
1. Deepfake audio contains required vocabulary
2. Candidates were identified (check `output/metadata/*.json`)
3. Review logs for specific errors

```bash
python src/04_tamper.py --log_level DEBUG
```

## Performance Benchmarks

**Dataset**: 12 audio files (~20 minutes total)

| Step | Time | Hardware |
|------|------|----------|
| Preprocessing | ~30s | CPU |
| Whisper (tiny) | ~20s | RTX 4080 |
| Whisper (large-v3) | ~2 min | RTX 4080 |
| MFA (optional) | ~3 min | 24 CPU threads |
| Candidate ID | ~1 min | CPU |
| Tampering (10/file) | ~5 min | CPU |
| **Total (tiny model)** | ~7 min | - |
| **Total (large-v3)** | ~12 min | - |

## Production Best Practices

### 1. Use Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Log Everything

```bash
python run_pipeline.py --all 2>&1 | tee pipeline_$(date +%Y%m%d_%H%M%S).log
```

### 3. Backup Outputs

```bash
# After successful run
tar -czf outputs_$(date +%Y%m%d).tar.gz output/
```

### 4. Version Control Data

Track which data version produced which outputs:
```bash
echo "Pipeline run: $(date)" > output/VERSION.txt
echo "Data: data/sub096/" >> output/VERSION.txt
echo "Whisper: large-v3" >> output/VERSION.txt
```

### 5. Monitor Resource Usage

```bash
# During processing
watch -n 1 'nvidia-smi; echo ""; free -h'
```

## Scaling to Multiple Subjects

```bash
#!/bin/bash
# Process all subjects

for subject_dir in data/sub*/; do
    subject=$(basename $subject_dir)
    echo "Processing $subject..."

    python run_pipeline.py --all \
        --input_dir $subject_dir/original \
        --output_dir output/$subject \
        2>&1 | tee logs/${subject}_$(date +%Y%m%d).log
done

echo "All subjects processed!"
```

## Data Quality Guidelines

### Original Audio

- **Duration**: 3-10 minutes ideal
- **Content**: Rich vocabulary, varied sentence structures
- **Quality**: Clear speech, minimal background noise
- **Format**: Any (m4a, wav, mp3) - will be normalized

### Deepfake Audio

- **Duration**: Longer is better (5+ minutes recommended)
- **Content**: Should overlap with original vocabulary
- **Coverage**: Read same scripts as originals for best results
- **Quality**: 16kHz mono WAV preferred

### Optimal Setup

1. Record original speaker reading scripts (df_script.pdf, LP_script.pdf)
2. Generate deepfakes of same speaker reading SAME scripts
3. This ensures vocabulary overlap for word substitution

## Support & Maintenance

### Logs Location

- Pipeline logs: Current directory or `2>&1 | tee`
- Individual scripts: `output/logs/` (if created)
- Errors: stderr output

### Updating Dependencies

```bash
pip install --upgrade -r requirements.txt
python -m spacy download en_core_web_sm --upgrade
```

### Cleaning Up

```bash
# Remove all generated files (keep source data)
rm -rf output/*

# Start fresh
./setup.sh
python run_pipeline.py --all
```

## Citation

If using this pipeline in research:

```bibtex
@software{tampered_deepfake_pipeline_2025,
  title={Tampered Deepfake Audio Generation Pipeline},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/tampered-deepfake},
  note={Research pipeline for generating prosody-aware tampered audio}
}
```

## License & Ethics

**Research Use Only**: This tool is intended for academic research in deepfake detection.

**Ethical Guidelines**:
- Obtain informed consent for all audio recordings
- Use only for research purposes
- Do not distribute tampered audio outside research context
- Follow institutional review board (IRB) guidelines
- Consider societal impact of research

## Contact

For questions or issues:
- Create an issue on GitHub
- Email: [your-email@institution.edu]
- Documentation: See README.md and CLAUDE.md

---

**Last Updated**: 2025-10-26
**Pipeline Version**: 1.0.0
**Tested On**: Ubuntu 22.04, Python 3.11, RTX 4080
