#!/usr/bin/env python3
"""
Test script for LLM-based tampering candidate identification.

This demonstrates the new LLM analyzer on actual Subject 96 transcript data.
"""

import json
from pathlib import Path
from src.utils.llm_analyzer import LLMAnalyzer

def main():
    # Load a sample transcript
    transcript_path = Path("output/transcripts/df_sub096_LP1_result_1.json")

    if not transcript_path.exists():
        print(f"ERROR: Transcript not found at {transcript_path}")
        print("Please run Stage 2 (02_analyze.py) first to generate transcripts.")
        return

    with open(transcript_path, 'r') as f:
        transcript_data = json.load(f)

    text = transcript_data['text']

    print("="*80)
    print("LLM ANALYZER TEST")
    print("="*80)
    print(f"\nTranscript: {transcript_path.name}")
    print(f"Text ({len(text)} chars):")
    print(f"  {text}")
    print()

    # Initialize LLM analyzer
    print("Connecting to LM Studio...")
    try:
        analyzer = LLMAnalyzer()
        print("✓ Connected successfully\n")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return

    # Run analysis
    print("Running LLM analysis (4 strategies)...")
    print("-"*80)

    results = analyzer.analyze_transcript(text)

    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    total_candidates = 0

    for strategy_name, candidates in results.items():
        print(f"\n{strategy_name.upper().replace('_', ' ')}:")
        print(f"  Found: {len(candidates)} candidates")

        if candidates:
            for i, cand in enumerate(candidates, 1):
                print(f"\n  [{i}] {cand['word']} → {cand['replacement']}")
                if 'reason' in cand:
                    print(f"      Reason: {cand['reason']}")
                if 'context' in cand:
                    print(f"      Context: {cand['context']}")
                if 'shift_direction' in cand:
                    print(f"      Shift: {cand['shift_direction']}")

        total_candidates += len(candidates)

    print("\n" + "="*80)
    print(f"TOTAL LLM CANDIDATES: {total_candidates}")
    print("="*80)

    # Compare with spaCy baseline
    candidate_path = Path("output/metadata") / f"{transcript_path.stem}_candidates.json"
    if candidate_path.exists():
        with open(candidate_path, 'r') as f:
            spacy_candidates = json.load(f)

        print(f"\nCOMPARISON:")
        print(f"  spaCy candidates: {len(spacy_candidates)}")
        print(f"  LLM candidates:   {total_candidates}")
        print(f"  Combined total:   {len(spacy_candidates) + total_candidates}")
        print(f"  Increase:         +{total_candidates} ({total_candidates/len(spacy_candidates)*100:.1f}%)")

if __name__ == "__main__":
    main()
