import numpy as np


def normalize_embeddings(W):
    """
    Return a row-normalized copy of W (each row divided by its L2 norm).
    Pre-computing this once makes every subsequent cosine search a plain matmul.

    Args:
        W: embedding matrix, shape (V, d)

    Returns:
        W_normed: unit-length rows, shape (V, d)
    """
    return W / np.linalg.norm(W, axis=1, keepdims=True)


def cosine_similarity(vec_a, vec_b):
    """
    Compute cosine similarity between two vectors.

    cosine_sim = (a . b) / (||a|| * ||b||)

    Args:
        vec_a: numpy array of shape (d,)
        vec_b: numpy array of shape (d,)

    Returns:
        similarity: scalar in [-1, 1]
    """
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))


def find_nearest(word, word2idx, idx2word, W_normed, top_n=5):
    """
    Find the top_n most similar words to the given word using cosine similarity.

    Args:
        word: string, the query word
        word2idx: dict mapping word -> index
        idx2word: dict mapping index -> word
        W_normed: pre-normalized embedding matrix from normalize_embeddings(), shape (V, d)
        top_n: how many neighbors to return

    Returns:
        neighbors: list of (word, similarity_score) tuples, sorted by similarity descending
    """
    # Row is already unit length — cosine similarity is a plain matmul
    query_norm   = W_normed[word2idx[word]]       # shape (d,)
    similarities = W_normed @ query_norm           # shape (V,)

    # Exclude the query word itself, then take top_n
    similarities[word2idx[word]] = -np.inf

    top_indices = np.argsort(similarities)[::-1][:top_n]
    return [(idx2word[i], float(similarities[i])) for i in top_indices]


def analogy(word_a, word_b, word_c, word2idx, idx2word, W_normed):
    """
    Solve: word_a is to word_b as word_c is to ???

    Compute: vec(word_b) - vec(word_a) + vec(word_c)
    Return the word closest to that result vector (excluding a, b, c).

    Args:
        word_a, word_b, word_c: strings
        word2idx: dict mapping word -> index
        idx2word: dict mapping index -> word
        W_normed: pre-normalized embedding matrix from normalize_embeddings(), shape (V, d)

    Returns:
        result_word: string, the best analogy answer
    """
    # Compute the analogy vector from unit rows — direction is preserved
    vec = W_normed[word2idx[word_b]] - W_normed[word2idx[word_a]] + W_normed[word2idx[word_c]]

    vec_norm     = vec / np.linalg.norm(vec)
    similarities = W_normed @ vec_norm                          # shape (V,)

    # Mask out input words so they can't be returned as the answer
    for w in (word_a, word_b, word_c):
        similarities[word2idx[w]] = -np.inf

    return idx2word[int(np.argmax(similarities))]


def eval_analogies(tests, word2idx, idx2word, W_normed):
    """
    Evaluate analogy accuracy over a list of (a, b, c, expected) tuples.
    Skips any test where a word is absent from the vocabulary.

    Args:
        tests: list of (word_a, word_b, word_c, expected) tuples
        word2idx: dict mapping word -> index
        idx2word: dict mapping index -> word
        W_normed: pre-normalized embedding matrix from normalize_embeddings(), shape (V, d)

    Returns:
        accuracy: float, fraction correct (0.0 if all skipped)
        correct:  int, number of correct predictions
        total:    int, number attempted (OOV tests excluded)
    """
    correct = 0
    total   = 0
    for word_a, word_b, word_c, expected in tests:
        if not all(w in word2idx for w in (word_a, word_b, word_c, expected)):
            continue
        if analogy(word_a, word_b, word_c, word2idx, idx2word, W_normed) == expected:
            correct += 1
        total += 1
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total
