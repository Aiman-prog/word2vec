import numpy as np
from collections import Counter


def build_vocab(corpus, min_count=5):
    """
    Tokenize corpus and build vocabulary mappings.
    Words appearing fewer than min_count times are discarded. This removes
    noise from rare words and significantly shrinks the vocab on large corpora.

    Args:
        corpus: a string containing the entire text corpus
        min_count: discard words with frequency below this threshold

    Returns:
        word2idx: dict mapping word -> integer index
        idx2word: dict mapping integer index -> word
        word_counts: dict mapping word -> frequency count (filtered vocab only)
        total_count: int, total number of tokens in the filtered vocabulary
    """
    tokens = corpus.lower().split()

    word_counts = Counter(tokens)

    word_counts = {w: c for w, c in word_counts.items() if c >= min_count}

    unique_words = sorted(word_counts.keys())

    word2idx = {word: i for i, word in enumerate(unique_words)}
    idx2word = {i: word for i, word in enumerate(unique_words)}

    total_count = sum(word_counts.values())
    return word2idx, idx2word, word_counts, total_count


def subsample_tokens(token_ids, keep_probs):
    """
    Randomly discard frequent words before generating training pairs.
    The discard probability for word w is: 1 - sqrt(t / freq(w))
    where freq(w) = count(w) / total_tokens.

    Args:
        token_ids: int32 numpy array of token indices (already filtered by min_count)
        keep_probs: float32 numpy array of shape (vocab_size,), keep probability per word ID
                    — built once in the caller so no dict lookup happens per token

    Returns:
        subsampled: int32 numpy array with frequent words randomly removed
    """
    mask = np.random.rand(len(token_ids)) < keep_probs[token_ids]
    return token_ids[mask]


def generate_training_pairs(token_ids, window_size, random_window=False):
    """
    Generate (center_word, context_word) pairs using a sliding window.

    For each token in the corpus, pair it with every token within
    `window_size` positions to the left and right.

    When random_window=True, the effective radius is sampled independently
    for each center word from Uniform(1, window_size) (Mikolov 2013 §3.1).
    This gives closer context words higher expected weight.

    Args:
        token_ids: int32 numpy array of token indices (already filtered and subsampled)
        window_size: maximum context window radius
        random_window: if True, sample a random radius per center word

    Returns:
        pairs: numpy array of shape (N, 2) with (center_idx, context_idx) rows
    """
    n = len(token_ids)
    chunks = []

    # Pre-sample one radius per position (Mikolov 2013 §3.1)
    if random_window and n > 0:
        radii = np.random.randint(1, window_size + 1, size=n)

    for offset in range(1, window_size + 1):
        left_c  = token_ids[offset:]   # center sits to the right of context
        left_x  = token_ids[:-offset]  # context
        right_c = token_ids[:-offset]  # center sits to the left of context
        right_x = token_ids[offset:]   # context

        if random_window and n > 0:
            # Include this offset only where the center word's sampled radius >= offset.
            # Left pairs:  center is at corpus position [offset, n),  radii index starts at offset
            # Right pairs: center is at corpus position [0, n-offset), radii index starts at 0
            lmask = radii[offset:]  >= offset
            rmask = radii[:-offset] >= offset
            left_c,  left_x  = left_c[lmask],  left_x[lmask]
            right_c, right_x = right_c[rmask], right_x[rmask]

        if len(left_c) > 0:
            chunks.append(np.column_stack([left_c,  left_x]))
        if len(right_c) > 0:
            chunks.append(np.column_stack([right_c, right_x]))

    if not chunks:
        return np.empty((0, 2), dtype=np.int32)
    return np.vstack(chunks).astype(np.int32)


def get_noise_distribution(word_counts, word2idx):
    """
    Compute the noise distribution for negative sampling.
    P(w) = count(w)^0.75 / sum(count(w')^0.75)

    The 3/4 power smooths the distribution so rare words
    get sampled more often than their raw frequency would suggest.

    Args:
        word_counts: dict mapping word -> frequency
        word2idx: dict mapping word -> index

    Returns:
        noise_dist: numpy array of shape (vocab_size,) with sampling probabilities
    """
    words_by_idx = sorted(word2idx, key=word2idx.__getitem__)
    counts = np.array([word_counts[w] for w in words_by_idx], dtype=np.float32)

    noise_dist = np.power(counts, 0.75)

    # Normalize
    noise_dist /= noise_dist.sum()

    return noise_dist
