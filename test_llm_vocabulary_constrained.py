#!/usr/bin/env python3
"""
Vocabulary-constrained tampering: Only suggest replacements using words
available in the deepfake audio corpus.
"""

import json
import requests
from pathlib import Path
from collections import Counter

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
        "max_tokens": 1200
    }
    response = requests.post(LM_STUDIO_URL, json=payload, timeout=180)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']

def extract_deepfake_vocabulary():
    """Extract all words from deepfake transcripts."""
    words = []
    transcript_dir = Path('output/transcripts')

    for transcript_file in transcript_dir.glob('*.json'):
        with open(transcript_file, 'r') as f:
            data = json.load(f)
            text = data.get('text', '').lower()
            # Remove punctuation for cleaner vocabulary
            text = text.replace('.', '').replace(',', '').replace('?', '').replace('!', '')
            words.extend(text.split())

    word_freq = Counter(words)
    return sorted(word_freq.keys()), word_freq

def vocabulary_constrained_analysis(original_text, deepfake_vocab, word_freq):
    """
    Ask LLM to identify tampering candidates using ONLY words available
    in the deepfake corpus.
    """
    # Create a more manageable vocab list for the prompt
    common_words = [w for w, count in word_freq.most_common(200)]

    system_prompt = """You are an expert linguist performing vocabulary-constrained audio tampering analysis.

CRITICAL CONSTRAINT: You can ONLY suggest word replacements using words from the DEEPFAKE VOCABULARY list provided.

Your task:
1. Analyze the original transcript
2. Identify 5-8 words that can be replaced with words from the deepfake vocabulary
3. Ensure replacements create natural, grammatically correct sentences
4. Prioritize high semantic impact (meaning changes significantly)

Replacement types to consider:
- Antonyms: up↔down, back↔forward, white↔red, near↔far
- Negations: add "not" or "no" to reverse meaning
- Modality: will↔might, can↔could
- Directional: up/down, back/forward, near/far, in/out

Output ONLY valid JSON:
{
  "candidates": [
    {
      "original_word": "word in original",
      "replacement_word": "word from deepfake vocab",
      "original_context": "sentence containing original word",
      "tampered_sentence": "sentence with replacement",
      "replacement_type": "antonym|negation|modality|directional",
      "semantic_impact": 4.5,
      "is_natural": true,
      "deepfake_word_frequency": 15
    }
  ]
}

RULES:
- replacement_word MUST exist in the deepfake vocabulary
- Tampered sentence must sound natural
- semantic_impact: 1-5 scale (minimum 3.0)
- Only include candidates with is_natural=true"""

    # Limit vocab shown to avoid token overflow
    vocab_sample = ', '.join(common_words[:150])

    user_prompt = f"""ORIGINAL TRANSCRIPT:
"{original_text}"

DEEPFAKE VOCABULARY (most common 150 words):
{vocab_sample}

Key antonym pairs available: back/forward, up/down, near/far, white/red, not, no, yes, all, some, few, many

Identify 5-8 word replacements that:
- Use ONLY words from the deepfake vocabulary above
- Create natural-sounding tampered sentences
- Have high semantic impact (≥3.0)

Provide ONLY JSON output."""

    print("\n" + "="*80)
    print("VOCABULARY-CONSTRAINED ANALYSIS")
    print("="*80)
    print(f"Available deepfake vocabulary: {len(deepfake_vocab)} unique words")
    print(f"Showing LLM top 150 most common words")
    print("\nCalling Mistral...")

    response = call_llm(system_prompt, user_prompt)
    print("\nLLM Response:")
    print(response)

    try:
        data = json.loads(response)
        candidates = data.get('candidates', [])

        print(f"\n{'='*80}")
        print(f"CANDIDATES FOUND: {len(candidates)}")
        print("="*80)

        validated = []
        for i, cand in enumerate(candidates, 1):
            replacement = cand['replacement_word'].lower()

            # Verify replacement word exists in deepfake corpus
            in_corpus = replacement in deepfake_vocab
            impact = cand.get('semantic_impact', 0)
            natural = cand.get('is_natural', False)

            status = "✓ VALID" if (in_corpus and natural and impact >= 3.0) else "✗ INVALID"

            print(f"\n[{i}] '{cand['original_word']}' → '{cand['replacement_word']}'")
            print(f"    Type: {cand['replacement_type']}")
            print(f"    Impact: {impact}/5.0")
            print(f"    Natural: {'✓' if natural else '✗'}")
            print(f"    In deepfake corpus: {'✓' if in_corpus else '✗ NOT FOUND'}")
            print(f"    Deepfake occurrences: {word_freq.get(replacement, 0)}x")
            print(f"    Status: {status}")

            print(f"\n    ORIGINAL: {cand['original_context']}")
            print(f"    TAMPERED: {cand['tampered_sentence']}")
            print(f"    {'-'*76}")

            if in_corpus and natural and impact >= 3.0:
                validated.append(cand)

        print(f"\n{'='*80}")
        print(f"VALIDATION RESULTS:")
        print(f"  Suggested:   {len(candidates)}")
        print(f"  Validated:   {len(validated)} ✓")
        print(f"  (In corpus AND natural AND impact ≥3.0)")
        print("="*80)

        return validated

    except json.JSONDecodeError as e:
        print(f"\n✗ JSON parse error: {e}")
        return []

def main():
    # Load original transcript
    transcript_path = Path("output/transcripts/df_sub096_LP1_result_1.json")

    with open(transcript_path, 'r') as f:
        original_text = json.load(f)['text']

    print("="*80)
    print("VOCABULARY-CONSTRAINED TAMPERING ANALYSIS")
    print("="*80)
    print("\nOriginal transcript:")
    print(f"  {original_text}")

    # Extract deepfake vocabulary
    print("\nExtracting deepfake vocabulary...")
    deepfake_vocab, word_freq = extract_deepfake_vocabulary()
    print(f"✓ Found {len(deepfake_vocab)} unique words in deepfake corpus")

    # Key words for tampering
    key_words = ['not', 'no', 'yes', 'back', 'forward', 'up', 'down', 'white', 'red',
                 'near', 'far', 'all', 'some', 'many', 'few']
    available_key_words = [w for w in key_words if w in deepfake_vocab]
    print(f"\nKey tampering words available: {available_key_words}")
    print(f"  'not': {word_freq.get('not', 0)} occurrences")
    print(f"  'no': {word_freq.get('no', 0)} occurrences")

    # Test connection
    try:
        requests.get("http://localhost:1234/v1/models", timeout=5).raise_for_status()
        print("\n✓ Connected to LM Studio\n")
    except Exception as e:
        print(f"\n✗ Connection failed: {e}")
        return

    # Run constrained analysis
    validated = vocabulary_constrained_analysis(original_text, deepfake_vocab, word_freq)

    # Summary
    print("\n" + "="*80)
    print("IMPLEMENTABLE TAMPERING CANDIDATES")
    print("="*80)
    print(f"\nMistral identified {len(validated)} candidates that:")
    print("  ✓ Use words available in your deepfake audio")
    print("  ✓ Create natural-sounding sentences")
    print("  ✓ Have high semantic impact (≥3.0/5.0)")
    print("\nThese candidates can be IMMEDIATELY implemented in Stage 4!")

    if validated:
        print("\nReady for audio tampering:")
        for i, cand in enumerate(validated, 1):
            freq = word_freq.get(cand['replacement_word'].lower(), 0)
            print(f"  {i}. {cand['original_word']} → {cand['replacement_word']} ({freq} instances)")

if __name__ == "__main__":
    main()
