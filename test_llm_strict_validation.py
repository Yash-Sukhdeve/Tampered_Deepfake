#!/usr/bin/env python3
"""
Strict validation: LLM must verify replacements create natural, coherent sentences
with significant semantic impact.
"""

import json
import requests
from pathlib import Path

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL = "mistral-7b-instruct-v0.1"

def call_llm(system_prompt, user_prompt):
    """Call LLM API."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0,
        "max_tokens": 1000
    }
    response = requests.post(LM_STUDIO_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']

def strict_validation(text):
    """
    Strict validation with natural language flow check.
    """
    system_prompt = """You are an expert linguist performing STRICT validation for audio tampering research.

TASK: Identify 3-5 word replacements that:
1. Use TRUE ANTONYMS (opposites), not just synonyms
2. Create NATURAL-SOUNDING sentences when you read them aloud
3. Cause SIGNIFICANT meaning change (not subtle)

VALIDATION CRITERIA (must pass ALL):
✓ Replacement is a true antonym/opposite of original
✓ Tampered sentence is grammatically perfect
✓ Tampered sentence sounds natural when read aloud
✓ No awkward repetitions or redundant words
✓ Semantic impact is HIGH (meaning clearly flips)

REJECT candidates that:
✗ Create awkward/unnatural phrasing
✗ Use synonyms instead of antonyms
✗ Have minimal semantic impact
✗ Result in redundant words (e.g., "down...down")

Output ONLY valid JSON:
{
  "candidates": [
    {
      "original_word": "word",
      "replacement_word": "true antonym",
      "original_sentence": "sentence with original word",
      "tampered_sentence": "sentence with replacement",
      "passes_naturalness_test": true,
      "semantic_impact_score": 4.5,
      "impact_explanation": "how meaning changes"
    }
  ]
}

semantic_impact_score: 1-5 scale (1=minimal, 5=complete reversal)
Only include candidates with score >= 4.0"""

    user_prompt = f"""Analyze this transcript with STRICT validation criteria:

TRANSCRIPT: "{text}"

Find words to replace with TRUE ANTONYMS that create natural, high-impact semantic changes.

Provide ONLY JSON output with validated candidates."""

    print("\n" + "="*80)
    print("STRICT VALIDATION MODE")
    print("="*80)
    print("Calling LLM with rigorous validation criteria...")

    response = call_llm(system_prompt, user_prompt)
    print("\nLLM Response:")
    print(response)

    try:
        data = json.loads(response)
        candidates = data.get('candidates', [])

        print(f"\n{'='*80}")
        print(f"STRICTLY VALIDATED CANDIDATES: {len(candidates)}")
        print("="*80)

        for i, cand in enumerate(candidates, 1):
            score = cand.get('semantic_impact_score', 0)
            natural = cand.get('passes_naturalness_test', False)

            print(f"\n[{i}] '{cand['original_word']}' → '{cand['replacement_word']}'")
            print(f"    Semantic Impact: {score}/5.0 {'✓' if score >= 4.0 else '✗'}")
            print(f"    Natural Flow:    {'✓ PASS' if natural else '✗ FAIL'}")

            print(f"\n    ORIGINAL:")
            print(f"    \"{cand['original_sentence']}\"")

            print(f"\n    TAMPERED:")
            print(f"    \"{cand['tampered_sentence']}\"")

            print(f"\n    IMPACT: {cand['impact_explanation']}")
            print(f"    {'-'*76}")

        # Filter to high-quality candidates only
        high_quality = [c for c in candidates
                       if c.get('passes_naturalness_test', False)
                       and c.get('semantic_impact_score', 0) >= 4.0]

        print(f"\n{'='*80}")
        print(f"QUALITY FILTER:")
        print(f"  All suggested:   {len(candidates)}")
        print(f"  High quality:    {len(high_quality)} ✓")
        print(f"  (Impact ≥4.0 AND natural flow)")
        print("="*80)

        return high_quality

    except json.JSONDecodeError as e:
        print(f"\n✗ JSON parse error: {e}")
        print(f"Response was: {response[:300]}...")
        return []

def main():
    transcript_path = Path("output/transcripts/df_sub096_LP1_result_1.json")

    with open(transcript_path, 'r') as f:
        text = json.load(f)['text']

    print("="*80)
    print("STRICT LLM VALIDATION TEST")
    print("="*80)
    print(f"\nOriginal: {text}\n")

    # Test connection
    try:
        requests.get("http://localhost:1234/v1/models", timeout=5).raise_for_status()
        print("✓ Connected to LM Studio\n")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return

    # Run strict validation
    high_quality_candidates = strict_validation(text)

    # Final summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"\nMistral-7B identified {len(high_quality_candidates)} high-quality candidates")
    print("that pass ALL validation criteria:")
    print("  ✓ True antonyms (not synonyms)")
    print("  ✓ Natural-sounding sentences")
    print("  ✓ High semantic impact (≥4.0/5.0)")
    print("  ✓ Grammatically perfect")

    if high_quality_candidates:
        print("\nThese candidates are ready for audio tampering implementation.")
    else:
        print("\nNo candidates met the strict criteria. May need to adjust thresholds")
        print("or try different tampering strategies (modality shift, temporal, etc.)")

if __name__ == "__main__":
    main()
