#!/usr/bin/env python3
"""
Test LLM tampering candidates with sentence coherence validation.

Ensures that suggested word replacements produce grammatically correct
and semantically meaningful sentences.
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
        "max_tokens": 800
    }

    response = requests.post(LM_STUDIO_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']

def identify_and_validate_candidates(text):
    """
    Identify tampering candidates AND validate sentence coherence.

    This uses a two-stage prompt:
    1. Identify replacement candidates
    2. Generate the tampered sentence to verify it makes sense
    """
    system_prompt = """You are an expert linguist analyzing text for semantic tampering research.

Your task:
1. Identify 3-5 words that can be replaced with antonyms or opposites
2. For EACH candidate, generate the full tampered sentence
3. Verify the tampered sentence is grammatically correct and semantically coherent
4. Only include candidates where the result makes sense

Output ONLY valid JSON (no markdown, no explanations). Format:
{
  "candidates": [
    {
      "original_word": "word",
      "replacement_word": "replacement",
      "original_sentence": "full original sentence containing the word",
      "tampered_sentence": "full sentence with replacement applied",
      "is_coherent": true,
      "semantic_impact": "brief description of meaning change"
    }
  ]
}

IMPORTANT:
- Only include candidates where is_coherent=true
- Tampered sentence must be grammatically correct
- Replacement must fit naturally in context"""

    user_prompt = f"""Analyze this transcript. Find words to replace with antonyms/opposites that change meaning while maintaining grammatical correctness:

TRANSCRIPT: "{text}"

Provide ONLY the JSON output with validated candidates."""

    print("\n" + "="*80)
    print("LLM ANALYSIS WITH VALIDATION")
    print("="*80)
    print("Calling LLM with enhanced validation prompt...")

    response = call_llm(system_prompt, user_prompt)
    print("\nLLM Response:")
    print(response)

    try:
        data = json.loads(response)
        candidates = data.get('candidates', [])

        print(f"\n{'='*80}")
        print(f"VALIDATED CANDIDATES: {len(candidates)}")
        print("="*80)

        for i, cand in enumerate(candidates, 1):
            print(f"\n[{i}] WORD REPLACEMENT:")
            print(f"    '{cand['original_word']}' → '{cand['replacement_word']}'")
            print(f"\n    ORIGINAL SENTENCE:")
            print(f"    {cand['original_sentence']}")
            print(f"\n    TAMPERED SENTENCE:")
            print(f"    {cand['tampered_sentence']}")
            print(f"\n    COHERENT: {'✓ YES' if cand['is_coherent'] else '✗ NO'}")
            print(f"    IMPACT: {cand['semantic_impact']}")
            print(f"    {'-'*76}")

        # Filter to only coherent candidates
        valid_candidates = [c for c in candidates if c.get('is_coherent', False)]
        print(f"\n{'='*80}")
        print(f"FINAL COUNT:")
        print(f"  Total suggested: {len(candidates)}")
        print(f"  Validated:       {len(valid_candidates)} ✓")
        print("="*80)

        return valid_candidates

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
    print("LLM TAMPERING IDENTIFICATION WITH SENTENCE VALIDATION")
    print("="*80)
    print(f"\nTranscript: {transcript_path.name}")
    print(f"\nOriginal Text:")
    print(f"  {text}")

    # Test connection
    print("\nTesting LM Studio connection...")
    try:
        test_response = requests.get("http://localhost:1234/v1/models", timeout=5)
        test_response.raise_for_status()
        print("✓ Connected to LM Studio (Mistral-7B)")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return

    # Run analysis with validation
    valid_candidates = identify_and_validate_candidates(text)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nThe LLM identified {len(valid_candidates)} validated tampering candidates")
    print("where replacements produce grammatically correct, coherent sentences.")
    print("\nKey advantages over rule-based NLP:")
    print("  ✓ Context-aware semantic inversions")
    print("  ✓ Validates sentence coherence automatically")
    print("  ✓ Identifies nuanced antonym relationships")
    print("  ✓ Ensures tampered sentences make linguistic sense")

if __name__ == "__main__":
    main()
