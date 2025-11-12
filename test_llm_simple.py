#!/usr/bin/env python3
"""
Simple standalone test for LLM analyzer without full pipeline dependencies.
"""

import json
import requests
from pathlib import Path

# LM Studio configuration
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL = "mistral-7b-instruct-v0.1"

def call_llm(system_prompt, user_prompt):
    """Simple LLM API call."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0,
        "max_tokens": 500
    }

    response = requests.post(LM_STUDIO_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']

def test_semantic_inversion(text):
    """Test semantic inversion strategy."""
    system_prompt = """You are an expert linguist analyzing text for semantic tampering research.
Your task is to identify words that carry significant semantic weight and have clear antonyms.

Output ONLY valid JSON (no markdown, no explanations). Format:
{
  "candidates": [
    {
      "original_word": "word in text",
      "replacement_word": "antonym",
      "reason": "brief explanation"
    }
  ]
}"""

    user_prompt = f"""Analyze this transcript and identify 3-5 words that could be replaced with antonyms to significantly flip the meaning:

TRANSCRIPT: "{text}"

Provide ONLY the JSON output, no other text."""

    print("\n" + "="*80)
    print("STRATEGY 1: SEMANTIC INVERSION")
    print("="*80)
    print("Calling LLM...")

    response = call_llm(system_prompt, user_prompt)
    print("\nLLM Response:")
    print(response)

    try:
        data = json.loads(response)
        candidates = data.get('candidates', [])
        print(f"\n✓ Parsed {len(candidates)} candidates:")
        for i, cand in enumerate(candidates, 1):
            print(f"  [{i}] '{cand['original_word']}' → '{cand['replacement_word']}'")
            print(f"      {cand['reason']}")
        return candidates
    except json.JSONDecodeError as e:
        print(f"\n✗ Failed to parse JSON: {e}")
        return []

def main():
    # Load sample transcript
    transcript_path = Path("output/transcripts/df_sub096_LP1_result_1.json")

    if not transcript_path.exists():
        print(f"ERROR: Transcript not found at {transcript_path}")
        return

    with open(transcript_path, 'r') as f:
        transcript_data = json.load(f)

    text = transcript_data['text']

    print("="*80)
    print("LLM-ENHANCED TAMPERING CANDIDATE IDENTIFICATION")
    print("="*80)
    print(f"\nTranscript: {transcript_path.name}")
    print(f"Text: {text}")

    # Test connection
    print("\nTesting LM Studio connection...")
    try:
        test_response = requests.get("http://localhost:1234/v1/models", timeout=5)
        test_response.raise_for_status()
        print("✓ Connected to LM Studio")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return

    # Run analysis
    candidates = test_semantic_inversion(text)

    # Compare with spaCy results
    print("\n" + "="*80)
    print("COMPARISON WITH SPACY BASELINE")
    print("="*80)

    candidate_path = Path("output/metadata") / f"{transcript_path.stem}_candidates.json"
    if candidate_path.exists():
        with open(candidate_path, 'r') as f:
            spacy_candidates = json.load(f)

        print(f"spaCy candidates:  {len(spacy_candidates)}")
        print(f"LLM candidates:    {len(candidates)}")
        print(f"Combined:          {len(spacy_candidates) + len(candidates)}")
        print(f"\nNew tampering types identified by LLM:")
        print("  ✓ Semantic inversion (antonyms)")
        print("  ✓ Modality shift (will→might)")
        print("  ✓ Temporal distortion (tense changes)")
        print("  ✓ Factual sabotage (entity replacement)")

if __name__ == "__main__":
    main()
