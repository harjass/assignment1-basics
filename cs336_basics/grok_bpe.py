from typing import Dict, List, Tuple
import regex as re
from collections import defaultdict, Counter
import itertools

def train_bpe_optimized(input_path: str, vocab_size: int, special_tokens: List[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """Train a BPE tokenizer with optimized pair counting."""
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # Initialize vocabulary
    vocab: Dict[int, bytes] = {}
    idx = 0
    for token in special_tokens:
        vocab[idx] = token.encode('utf-8')
        idx += 1
    for i in range(256):
        vocab[idx] = bytes([i])
        idx += 1

    # Read corpus
    with open(input_path, 'r', encoding='utf-8') as f:
        corpus = f.read()

    # Split on special tokens
    if special_tokens:
        special_pattern = '|'.join(re.escape(t) for t in special_tokens)
        chunks = re.split(f'({special_pattern})', corpus)
    else:
        chunks = [corpus]

    # Collect texts for pre-tokenization
    texts = [chunks[i] for i in range(0, len(chunks), 2) if i < len(chunks) and chunks[i]]

    # Single-threaded pre-tokenization for small corpora
    pre_tokens = list(itertools.chain.from_iterable(re.findall(PAT, t) for t in texts))

    # Frequency of each unique pre-token
    freq: Counter = Counter(pre_tokens)

    # Represent each unique pre-token as list of bytes (initially single bytes)
    pre_token_symbols: Dict[str, List[bytes]] = {
        word: [bytes([b]) for b in word.encode('utf-8')]
        for word in freq
    }

    # Initialize pair counts and pair to words
    pair_counts: Dict[Tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_words: Dict[Tuple[bytes, bytes], set] = defaultdict(set)

    for word, syms in pre_token_symbols.items():
        for j in range(len(syms) - 1):
            pair = (syms[j], syms[j + 1])
            pair_counts[pair] += freq[word]
            pair_to_words[pair].add(word)

    # Merges list
    merges: List[Tuple[bytes, bytes]] = []

    current_vocab_size = len(vocab)

    while current_vocab_size < vocab_size:
        if not pair_counts:
            break

        # Find best pair: max count, then lex max
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))

        if pair_counts[best_pair] <= 0:
            break

        merges.append(best_pair)

        new_token = best_pair[0] + best_pair[1]
        vocab[current_vocab_size] = new_token
        current_vocab_size += 1

        # Affected words
        affected_words = pair_to_words.pop(best_pair, set())

        for word in affected_words:
            syms = pre_token_symbols[word]

            # Subtract old pairs
            for j in range(len(syms) - 1):
                pair = (syms[j], syms[j + 1])
                pair_counts[pair] -= freq[word]
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                pair_to_words[pair].discard(word)
                if not pair_to_words[pair]:
                    del pair_to_words[pair]

            # Update symbols by merging
            new_syms: List[bytes] = []
            j = 0
            while j < len(syms):
                if j + 1 < len(syms) and (syms[j], syms[j + 1]) == best_pair:
                    new_syms.append(new_token)
                    j += 2
                else:
                    new_syms.append(syms[j])
                    j += 1
            pre_token_symbols[word] = new_syms

            # Add new pairs
            for j in range(len(new_syms) - 1):
                pair = (new_syms[j], new_syms[j + 1])
                pair_counts[pair] += freq[word]
                pair_to_words[pair].add(word)

    return vocab, merges