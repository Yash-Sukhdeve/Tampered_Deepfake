"""
LLM-based Semantic Tampering Analyzer

This module uses a local LLM (via LM Studio) to identify semantic tampering
candidates that go beyond rule-based NLP (spaCy) approaches.

Implements four high-impact tampering strategies:
1. Semantic Inversion - Replace words with antonyms
2. Modality & Certainty Shift - Change modal verbs (will/might/could)
3. Temporal Distortion - Alter verb tenses
4. Factual Sabotage - Replace entities (dates, numbers, names)

Author: Enhanced for Tampered_Deepfake research project
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LM Studio Configuration
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
DEFAULT_MODEL = "mistral-7b-instruct-v0.1"
DEFAULT_TEMPERATURE = 0  # Deterministic for reproducibility
DEFAULT_MAX_TOKENS = 500

class LLMAnalyzer:
    """Client for analyzing transcripts using local LLM via LM Studio."""

    def __init__(
        self,
        base_url: str = LM_STUDIO_BASE_URL,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS
    ):
        """
        Initialize LLM analyzer.

        Args:
            base_url: LM Studio API base URL
            model: Model identifier
            temperature: Sampling temperature (0 = deterministic)
            max_tokens: Maximum tokens in response
        """
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.session = requests.Session()

        # Test connection
        self._test_connection()

    def _test_connection(self) -> bool:
        """Test connection to LM Studio server."""
        try:
            response = self.session.get(f"{self.base_url}/models", timeout=5)
            response.raise_for_status()
            models = response.json()
            logger.info(f"Connected to LM Studio. Available models: {[m['id'] for m in models['data']]}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to LM Studio at {self.base_url}: {e}")
            raise ConnectionError(f"LM Studio not accessible: {e}")

    def _call_llm(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """
        Make API call to LM Studio.

        Args:
            system_prompt: System instruction
            user_prompt: User query

        Returns:
            LLM response text or None on failure
        """
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }

            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=120  # 2 minute timeout
            )
            response.raise_for_status()
            elapsed = time.time() - start_time

            result = response.json()
            content = result['choices'][0]['message']['content']
            tokens = result['usage']['total_tokens']

            logger.debug(f"LLM call completed in {elapsed:.2f}s ({tokens} tokens)")
            return content

        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return None

    def identify_semantic_inversion_candidates(self, text: str) -> List[Dict[str, Any]]:
        """
        Identify words that can be replaced with antonyms to flip meaning.

        Strategy: Find semantically-weighted words with clear opposites.
        Example: "conclusive" → "inconclusive"

        Args:
            text: Input transcript text

        Returns:
            List of candidate dictionaries
        """
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

        response = self._call_llm(system_prompt, user_prompt)
        if not response:
            return []

        try:
            # Parse LLM response
            data = json.loads(response)
            candidates = []

            for item in data.get('candidates', []):
                candidates.append({
                    'word': item['original_word'],
                    'replacement': item['replacement_word'],
                    'type': 'semantic_inversion',
                    'action': 'substitute',
                    'reason': item.get('reason', ''),
                    'llm_source': 'mistral-7b'
                })

            logger.info(f"Identified {len(candidates)} semantic inversion candidates")
            return candidates

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM JSON response: {e}\nResponse: {response[:200]}")
            return []

    def identify_modality_candidates(self, text: str) -> List[Dict[str, Any]]:
        """
        Identify modal verbs and certainty markers that can be shifted.

        Strategy: Find will/must/can and downgrade to might/could/may.
        Example: "This will solve" → "This might solve"

        Args:
            text: Input transcript text

        Returns:
            List of candidate dictionaries
        """
        system_prompt = """You are an expert linguist analyzing text for semantic tampering research.
Your task is to identify modal verbs and certainty markers (will, must, can, definitely, certainly) that could be weakened or strengthened.

Output ONLY valid JSON (no markdown, no explanations). Format:
{
  "candidates": [
    {
      "original_phrase": "phrase with modal",
      "original_word": "will",
      "replacement_word": "might",
      "certainty_shift": "high_to_low"
    }
  ]
}"""

        user_prompt = f"""Analyze this transcript and identify 2-4 modal verbs or certainty markers that could be altered:

TRANSCRIPT: "{text}"

Provide ONLY the JSON output, no other text."""

        response = self._call_llm(system_prompt, user_prompt)
        if not response:
            return []

        try:
            data = json.loads(response)
            candidates = []

            for item in data.get('candidates', []):
                candidates.append({
                    'word': item['original_word'],
                    'replacement': item['replacement_word'],
                    'type': 'modality_shift',
                    'action': 'substitute',
                    'context': item.get('original_phrase', ''),
                    'shift_direction': item.get('certainty_shift', ''),
                    'llm_source': 'mistral-7b'
                })

            logger.info(f"Identified {len(candidates)} modality shift candidates")
            return candidates

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM JSON response: {e}")
            return []

    def identify_temporal_distortion_candidates(self, text: str) -> List[Dict[str, Any]]:
        """
        Identify verbs whose tense can be changed to distort timeline.

        Strategy: Change past → future, or completed → ongoing.
        Example: "completed the task" → "will complete the task"

        Args:
            text: Input transcript text

        Returns:
            List of candidate dictionaries
        """
        system_prompt = """You are an expert linguist analyzing text for semantic tampering research.
Your task is to identify verb phrases where changing the tense would significantly alter the meaning.

Output ONLY valid JSON (no markdown, no explanations). Format:
{
  "candidates": [
    {
      "original_phrase": "completed the task",
      "original_verb": "completed",
      "replacement_phrase": "will complete the task",
      "tense_shift": "past_to_future"
    }
  ]
}"""

        user_prompt = f"""Analyze this transcript and identify 2-3 verb phrases where tense changes would alter meaning:

TRANSCRIPT: "{text}"

Provide ONLY the JSON output, no other text."""

        response = self._call_llm(system_prompt, user_prompt)
        if not response:
            return []

        try:
            data = json.loads(response)
            candidates = []

            for item in data.get('candidates', []):
                candidates.append({
                    'word': item['original_verb'],
                    'replacement': item.get('replacement_phrase', ''),
                    'type': 'temporal_distortion',
                    'action': 'substitute',
                    'context': item.get('original_phrase', ''),
                    'tense_shift': item.get('tense_shift', ''),
                    'llm_source': 'mistral-7b'
                })

            logger.info(f"Identified {len(candidates)} temporal distortion candidates")
            return candidates

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM JSON response: {e}")
            return []

    def identify_factual_sabotage_candidates(self, text: str) -> List[Dict[str, Any]]:
        """
        Identify entities (numbers, dates, names) that could be altered.

        Strategy: Replace specific facts with plausible alternatives.
        Example: "Tuesday at 3PM" → "Friday at 10AM"

        Args:
            text: Input transcript text

        Returns:
            List of candidate dictionaries
        """
        system_prompt = """You are an expert linguist analyzing text for semantic tampering research.
Your task is to identify specific factual entities (numbers, dates, times, names, locations) that could be replaced with plausible but incorrect alternatives.

Output ONLY valid JSON (no markdown, no explanations). Format:
{
  "candidates": [
    {
      "original_entity": "Tuesday",
      "entity_type": "day",
      "replacement_entity": "Friday",
      "reason": "plausible alternative day"
    }
  ]
}"""

        user_prompt = f"""Analyze this transcript and identify 2-4 factual entities that could be sabotaged:

TRANSCRIPT: "{text}"

Provide ONLY the JSON output, no other text."""

        response = self._call_llm(system_prompt, user_prompt)
        if not response:
            return []

        try:
            data = json.loads(response)
            candidates = []

            for item in data.get('candidates', []):
                candidates.append({
                    'word': item['original_entity'],
                    'replacement': item['replacement_entity'],
                    'type': 'factual_sabotage',
                    'action': 'substitute',
                    'entity_type': item.get('entity_type', ''),
                    'reason': item.get('reason', ''),
                    'llm_source': 'mistral-7b'
                })

            logger.info(f"Identified {len(candidates)} factual sabotage candidates")
            return candidates

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM JSON response: {e}")
            return []

    def analyze_transcript(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run all LLM-based tampering analyses on a transcript.

        Args:
            text: Input transcript text

        Returns:
            Dictionary with candidate lists for each strategy
        """
        logger.info(f"Starting LLM analysis on transcript ({len(text)} chars)")
        start_time = time.time()

        results = {
            'semantic_inversion': self.identify_semantic_inversion_candidates(text),
            'modality_shift': self.identify_modality_candidates(text),
            'temporal_distortion': self.identify_temporal_distortion_candidates(text),
            'factual_sabotage': self.identify_factual_sabotage_candidates(text)
        }

        elapsed = time.time() - start_time
        total_candidates = sum(len(v) for v in results.values())

        logger.info(f"LLM analysis complete: {total_candidates} candidates in {elapsed:.2f}s")
        return results
