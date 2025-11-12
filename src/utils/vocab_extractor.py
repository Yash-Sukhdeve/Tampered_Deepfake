"""
Deepfake Vocabulary Extractor

Extracts and caches vocabulary from deepfake audio transcripts for
vocabulary-constrained LLM tampering candidate identification.

Author: Enhanced for Tampered_Deepfake research project
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import Counter
import pickle

logger = logging.getLogger(__name__)

# Global cache for vocabulary (to avoid repeated extraction)
_VOCAB_CACHE = None

class VocabularyCache:
    """Cached vocabulary data."""
    def __init__(self, unique_words: Set[str], word_frequencies: Dict[str, int]):
        self.unique_words = unique_words
        self.word_frequencies = word_frequencies
        self.total_words = sum(word_frequencies.values())

    def __len__(self):
        return len(self.unique_words)

    def contains(self, word: str) -> bool:
        """Check if word exists in vocabulary."""
        return word.lower() in self.unique_words

    def get_frequency(self, word: str) -> int:
        """Get frequency count for a word."""
        return self.word_frequencies.get(word.lower(), 0)

    def get_high_frequency_words(self, top_n: int = 150) -> List[str]:
        """Get top N most frequent words."""
        return [word for word, _ in Counter(self.word_frequencies).most_common(top_n)]


def clean_word(word: str) -> str:
    """
    Clean punctuation from word while preserving contractions.

    Args:
        word: Raw word string

    Returns:
        Cleaned word in lowercase

    Examples:
        "gone." → "gone"
        "don't" → "don't" (preserved)
        "3PM," → "3PM"
    """
    # Remove trailing punctuation
    word = word.strip()
    while word and word[-1] in '.,!?;:':
        word = word[:-1]

    # Remove leading punctuation (except for quotes in contractions)
    while word and word[0] in '.,!?;:':
        word = word[1:]

    return word.lower()


def extract_deepfake_vocabulary(
    transcript_dir: Path,
    use_cache: bool = True,
    cache_file: Path = None
) -> VocabularyCache:
    """
    Extract all unique words from deepfake transcripts.

    Args:
        transcript_dir: Directory containing Whisper transcript JSON files
        use_cache: Whether to use global cache (default: True)
        cache_file: Optional path to save/load pickled cache

    Returns:
        VocabularyCache object with unique words and frequencies

    Example:
        vocab = extract_deepfake_vocabulary(Path('output/transcripts'))
        print(f"Found {len(vocab)} unique words")
        print(f"'not' appears {vocab.get_frequency('not')} times")
    """
    global _VOCAB_CACHE

    # Return cached if available
    if use_cache and _VOCAB_CACHE is not None:
        logger.debug("Using cached vocabulary")
        return _VOCAB_CACHE

    # Try to load from file cache
    if cache_file and cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                _VOCAB_CACHE = pickle.load(f)
            logger.info(f"Loaded vocabulary cache from {cache_file}")
            return _VOCAB_CACHE
        except Exception as e:
            logger.warning(f"Failed to load cache file: {e}")

    # Extract vocabulary from transcripts
    logger.info(f"Extracting vocabulary from {transcript_dir}")

    all_words = []
    transcript_files = list(transcript_dir.glob("*.json"))

    if not transcript_files:
        raise FileNotFoundError(f"No transcript JSON files found in {transcript_dir}")

    for transcript_file in transcript_files:
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Get text from Whisper output
            text = data.get('text', '')

            # Split and clean words
            words = [clean_word(w) for w in text.split()]
            words = [w for w in words if w]  # Remove empty strings

            all_words.extend(words)

        except Exception as e:
            logger.warning(f"Failed to process {transcript_file.name}: {e}")

    # Calculate statistics
    word_frequencies = Counter(all_words)
    unique_words = set(all_words)

    # Create cache object
    _VOCAB_CACHE = VocabularyCache(unique_words, dict(word_frequencies))

    logger.info(f"Extracted vocabulary: {len(unique_words)} unique words, "
                f"{len(all_words)} total words")

    # Save to file cache if requested
    if cache_file:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(_VOCAB_CACHE, f)
            logger.info(f"Saved vocabulary cache to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save cache file: {e}")

    return _VOCAB_CACHE


def validate_word_in_corpus(word: str, vocab: VocabularyCache) -> bool:
    """
    Validate if a word exists in the deepfake corpus.

    Args:
        word: Word to validate
        vocab: VocabularyCache object

    Returns:
        True if word exists in corpus, False otherwise
    """
    return vocab.contains(word)


def get_vocabulary_summary(vocab: VocabularyCache) -> Dict:
    """
    Get summary statistics about vocabulary.

    Args:
        vocab: VocabularyCache object

    Returns:
        Dictionary with statistics
    """
    top_words = vocab.get_high_frequency_words(top_n=20)

    return {
        'unique_words': len(vocab),
        'total_words': vocab.total_words,
        'top_20_words': [
            {
                'word': word,
                'frequency': vocab.get_frequency(word)
            }
            for word in top_words
        ],
        'key_tampering_words': {
            'not': vocab.get_frequency('not'),
            'no': vocab.get_frequency('no'),
            'yes': vocab.get_frequency('yes'),
            'all': vocab.get_frequency('all'),
            'some': vocab.get_frequency('some'),
            'many': vocab.get_frequency('many'),
            'few': vocab.get_frequency('few'),
            'will': vocab.get_frequency('will'),
            'might': vocab.get_frequency('might'),
            'could': vocab.get_frequency('could'),
            'up': vocab.get_frequency('up'),
            'down': vocab.get_frequency('down'),
            'back': vocab.get_frequency('back'),
            'forward': vocab.get_frequency('forward')
        }
    }


def clear_cache():
    """Clear the global vocabulary cache."""
    global _VOCAB_CACHE
    _VOCAB_CACHE = None
    logger.debug("Vocabulary cache cleared")


if __name__ == "__main__":
    # Test vocabulary extraction
    logging.basicConfig(level=logging.INFO)

    transcript_dir = Path("output/transcripts")
    if not transcript_dir.exists():
        print(f"Error: {transcript_dir} not found")
        exit(1)

    print("="*80)
    print("VOCABULARY EXTRACTOR TEST")
    print("="*80)

    # Extract vocabulary
    vocab = extract_deepfake_vocabulary(transcript_dir)

    # Display summary
    summary = get_vocabulary_summary(vocab)
    print(f"\nUnique words: {summary['unique_words']}")
    print(f"Total words: {summary['total_words']}")

    print("\nTop 20 most frequent words:")
    for item in summary['top_20_words']:
        print(f"  {item['word']:15s} x{item['frequency']}")

    print("\nKey tampering words:")
    for word, freq in summary['key_tampering_words'].items():
        status = "✓" if freq > 0 else "✗"
        print(f"  {status} {word:10s} x{freq}")

    print("\n" + "="*80)
    print("VALIDATION TESTS")
    print("="*80)

    # Test validation
    test_words = ['not', 'hello', 'world', 'the', 'zzz', 'up', 'down']
    for word in test_words:
        in_corpus = validate_word_in_corpus(word, vocab)
        freq = vocab.get_frequency(word)
        status = "✓ FOUND" if in_corpus else "✗ NOT FOUND"
        print(f"{word:10s} {status:15s} (frequency: {freq})")
