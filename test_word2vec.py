import numpy as np
import pytest
from data import build_vocab, generate_training_pairs, get_noise_distribution
from model import sigmoid, forward_and_backward, sgd_update
from evaluate import cosine_similarity, eval_analogies, normalize_embeddings, analogy

CORPUS = "the cat sat on the mat"


# ============================================================
# Tests for build_vocab
# ============================================================

class TestBuildVocab:
    def test_vocab_size(self):
        word2idx, idx2word, word_counts, _ = build_vocab(CORPUS, min_count=1)
        assert len(word2idx) == 5  # "the" appears twice but is one word

    def test_round_trip(self):
        word2idx, idx2word, word_counts, _ = build_vocab(CORPUS, min_count=1)
        for word, idx in word2idx.items():
            assert idx2word[idx] == word

    def test_word_counts(self):
        word2idx, idx2word, word_counts, _ = build_vocab(CORPUS, min_count=1)
        assert word_counts["the"] == 2
        assert word_counts["cat"] == 1

    def test_empty_corpus(self):
        word2idx, idx2word, word_counts, _ = build_vocab("", min_count=1)
        assert len(word2idx) == 0

    def test_min_count_filters(self):
        word2idx, _, _, _ = build_vocab(CORPUS, min_count=2)
        # Only "the" (freq=2) survives min_count=2
        assert list(word2idx.keys()) == ["the"]


# ============================================================
# Tests for generate_training_pairs
# ============================================================

class TestTrainingPairs:
    def _pair_exists(self, pairs, a, b):
        """Check if (a, b) exists as a row in the (N, 2) numpy array."""
        return np.any((pairs[:, 0] == a) & (pairs[:, 1] == b))

    def _to_ids(self, tokens, word2idx):
        return np.array([word2idx[t] for t in tokens if t in word2idx], dtype=np.int32)

    def test_window_1_contains_neighbors(self):
        word2idx, _, _, _ = build_vocab(CORPUS, min_count=1)
        token_ids = self._to_ids(CORPUS.lower().split(), word2idx)
        pairs = generate_training_pairs(token_ids, window_size=1)
        cat_idx = word2idx["cat"]
        the_idx = word2idx["the"]
        sat_idx = word2idx["sat"]
        # "cat" neighbors with window=1 are "the" and "sat"
        assert self._pair_exists(pairs, cat_idx, the_idx)
        assert self._pair_exists(pairs, cat_idx, sat_idx)

    def test_window_1_excludes_distant(self):
        word2idx, _, _, _ = build_vocab(CORPUS, min_count=1)
        token_ids = self._to_ids(CORPUS.lower().split(), word2idx)
        pairs = generate_training_pairs(token_ids, window_size=1)
        cat_idx = word2idx["cat"]
        mat_idx = word2idx["mat"]
        # "cat" and "mat" are too far apart for window=1
        assert not self._pair_exists(pairs, cat_idx, mat_idx)

    def test_window_0_no_pairs(self):
        word2idx, _, _, _ = build_vocab(CORPUS, min_count=1)
        token_ids = self._to_ids(CORPUS.lower().split(), word2idx)
        pairs = generate_training_pairs(token_ids, window_size=0)
        assert len(pairs) == 0

    def test_single_word_no_pairs(self):
        word2idx, _, _, _ = build_vocab("hello", min_count=1)
        token_ids = self._to_ids(["hello"], word2idx)
        pairs = generate_training_pairs(token_ids, window_size=1)
        assert len(pairs) == 0


# ============================================================
# Tests for negative sampling
# ============================================================

class TestNegativeSampling:
    def test_noise_distribution_sums_to_one(self):
        word2idx, _, word_counts, _ = build_vocab(CORPUS, min_count=1)
        noise_dist = get_noise_distribution(word_counts, word2idx)
        assert abs(noise_dist.sum() - 1.0) < 1e-5  # float32 precision

    def test_noise_distribution_shape(self):
        word2idx, _, word_counts, _ = build_vocab(CORPUS, min_count=1)
        noise_dist = get_noise_distribution(word_counts, word2idx)
        assert noise_dist.shape == (len(word2idx),)

    def test_negative_samples_in_range(self):
        word2idx, _, word_counts, _ = build_vocab(CORPUS, min_count=1)
        noise_dist = get_noise_distribution(word_counts, word2idx)
        neg = np.random.choice(len(word2idx), size=100, p=noise_dist)
        assert np.all(neg >= 0)
        assert np.all(neg < len(word2idx))

    def test_most_frequent_word_sampled_most(self):
        word2idx, _, word_counts, _ = build_vocab(CORPUS, min_count=1)
        noise_dist = get_noise_distribution(word_counts, word2idx)
        neg = np.random.choice(len(word2idx), size=10000, p=noise_dist)
        counts = np.bincount(neg, minlength=len(word2idx))
        # "the" (freq=2) should be sampled most often
        the_idx = word2idx["the"]
        assert counts[the_idx] == counts.max()


# ============================================================
# Tests for sigmoid
# ============================================================

class TestSigmoid:
    def test_sigmoid_zero(self):
        assert sigmoid(0.0) == 0.5

    def test_sigmoid_large_positive(self):
        result = sigmoid(100.0)
        assert np.isfinite(result)
        assert result > 0.99

    def test_sigmoid_large_negative(self):
        result = sigmoid(-100.0)
        assert np.isfinite(result)
        assert result < 0.01


# ============================================================
# Tests for forward_and_backward
# ============================================================

class TestForwardAndBackward:
    def test_loss_is_positive(self):
        np.random.seed(42)
        V, d = 5, 3
        W_in = np.random.randn(V, d) * 0.1
        W_out = np.random.randn(V, d) * 0.1
        loss, _, _, _ = forward_and_backward(
            np.array([0]), np.array([1]), np.array([[2, 3]]), W_in, W_out
        )
        assert loss[0] > 0

    def test_loss_with_random_weights(self):
        """With very small weights, scores ≈ 0, sigmoid ≈ 0.5, loss ≈ -log(0.5) * (1+k)."""
        np.random.seed(42)
        V, d, k = 5, 3, 3
        W_in = np.random.randn(V, d) * 0.01
        W_out = np.random.randn(V, d) * 0.01
        loss, _, _, _ = forward_and_backward(
            np.array([0]), np.array([1]), np.arange(2, 2 + k)[None, :], W_in, W_out
        )
        expected_approx = -np.log(0.5) * (1 + k)  # ≈ -log(0.5) * 4 ≈ 2.77
        assert abs(loss[0] - expected_approx) < 0.5

    def _numerical_gradient(self, f, x, eps=1e-5):
        """Compute numerical gradient of f with respect to each element of x."""
        grad = np.zeros_like(x)
        for i in range(x.size):
            old_val = x.flat[i]
            x.flat[i] = old_val + eps
            loss_plus = f()
            x.flat[i] = old_val - eps
            loss_minus = f()
            x.flat[i] = old_val
            grad.flat[i] = (loss_plus - loss_minus) / (2 * eps)
        return grad

    def test_gradient_check(self):
        """Compare analytical gradients with numerical gradients."""
        np.random.seed(42)
        V, d = 5, 3
        W_in = np.random.randn(V, d)
        W_out = np.random.randn(V, d)
        center_indices  = np.array([0])
        context_indices = np.array([1])
        neg_indices     = np.array([[2, 3]])  # (1, k)

        _, grad_v_c, grad_u_o, grad_u_negs = forward_and_backward(
            center_indices, context_indices, neg_indices, W_in, W_out
        )
        # Gradients are (1, d) / (1, k, d) — squeeze the batch dim for comparison
        grad_v_c    = grad_v_c[0]
        grad_u_o    = grad_u_o[0]
        grad_u_negs = grad_u_negs[0]  # (k, d)

        def loss_fn():
            l, _, _, _ = forward_and_backward(center_indices, context_indices, neg_indices, W_in, W_out)
            return l[0]

        # Check grad_v_c
        num_grad_v_c = self._numerical_gradient(loss_fn, W_in[0])
        rel_error = np.abs(grad_v_c - num_grad_v_c) / (np.abs(grad_v_c) + np.abs(num_grad_v_c) + 1e-8)
        assert np.all(rel_error < 1e-5), f"v_c gradient check failed: max rel error = {rel_error.max()}"

        # Check grad_u_o
        num_grad_u_o = self._numerical_gradient(loss_fn, W_out[1])
        rel_error = np.abs(grad_u_o - num_grad_u_o) / (np.abs(grad_u_o) + np.abs(num_grad_u_o) + 1e-8)
        assert np.all(rel_error < 1e-5), f"u_o gradient check failed: max rel error = {rel_error.max()}"

        # Check each grad_u_neg
        for i, neg_idx in enumerate(neg_indices[0]):
            num_grad_neg = self._numerical_gradient(loss_fn, W_out[neg_idx])
            rel_error = np.abs(grad_u_negs[i] - num_grad_neg) / (np.abs(grad_u_negs[i]) + np.abs(num_grad_neg) + 1e-8)
            assert np.all(rel_error < 1e-5), f"u_neg[{i}] gradient check failed: max rel error = {rel_error.max()}"


# ============================================================
# Tests for SGD update
# ============================================================

class TestSGDUpdate:
    def test_only_touched_rows_change(self):
        np.random.seed(42)
        V, d = 5, 3
        W_in = np.random.randn(V, d)
        W_out = np.random.randn(V, d)
        W_in_orig = W_in.copy()
        W_out_orig = W_out.copy()

        center_indices  = np.array([0])
        context_indices = np.array([1])
        neg_indices     = np.array([[2, 3]])
        _, grad_v_c, grad_u_o, grad_u_negs = forward_and_backward(
            center_indices, context_indices, neg_indices, W_in, W_out
        )
        sgd_update(W_in, W_out, center_indices, context_indices, neg_indices,
                   grad_v_c, grad_u_o, grad_u_negs, lr=np.array([0.1]))

        # Untouched rows should be identical
        assert np.array_equal(W_in[4], W_in_orig[4])  # row 4 not involved
        assert np.array_equal(W_out[4], W_out_orig[4])

    def test_lr_zero_no_change(self):
        np.random.seed(42)
        V, d = 5, 3
        W_in = np.random.randn(V, d)
        W_out = np.random.randn(V, d)
        W_in_orig = W_in.copy()
        W_out_orig = W_out.copy()

        center_indices  = np.array([0])
        context_indices = np.array([1])
        neg_indices     = np.array([[2, 3]])
        _, grad_v_c, grad_u_o, grad_u_negs = forward_and_backward(
            center_indices, context_indices, neg_indices, W_in, W_out
        )
        sgd_update(W_in, W_out, center_indices, context_indices, neg_indices,
                   grad_v_c, grad_u_o, grad_u_negs, lr=np.array([0.0]))

        assert np.array_equal(W_in, W_in_orig)
        assert np.array_equal(W_out, W_out_orig)


# ============================================================
# Tests for eval_analogies
# ============================================================

class TestEvalAnalogies:
    def _make_normed(self):
        """Tiny 5-word vocab with orthonormal embeddings (each word is its own axis)."""
        word2idx = {"cat": 0, "mat": 1, "sat": 2, "on": 3, "the": 4}
        idx2word = {i: w for w, i in word2idx.items()}
        W = np.eye(5, dtype=np.float32)   # orthonormal — each word is its own axis
        W_normed = normalize_embeddings(W)
        return word2idx, idx2word, W_normed

    def test_skips_oov(self):
        word2idx, idx2word, W_normed = self._make_normed()
        tests = [("cat", "mat", "sat", "unknown")]   # "unknown" not in vocab
        acc, correct, total = eval_analogies(tests, word2idx, idx2word, W_normed)
        assert total == 0
        assert acc == 0.0

    def test_counts_correct(self):
        word2idx, idx2word, W_normed = self._make_normed()
        # With orthonormal W the analogy result is deterministic
        result = analogy("cat", "mat", "sat", word2idx, idx2word, W_normed)
        tests = [("cat", "mat", "sat", result)]   # guaranteed correct
        acc, correct, total = eval_analogies(tests, word2idx, idx2word, W_normed)
        assert total == 1
        assert correct == 1
        assert acc == 1.0

    def test_counts_incorrect(self):
        word2idx, idx2word, W_normed = self._make_normed()
        result = analogy("cat", "mat", "sat", word2idx, idx2word, W_normed)
        wrong = next(w for w in word2idx if w != result and w not in ("cat", "mat", "sat"))
        tests = [("cat", "mat", "sat", wrong)]    # guaranteed wrong
        acc, correct, total = eval_analogies(tests, word2idx, idx2word, W_normed)
        assert total == 1
        assert correct == 0
        assert acc == 0.0

    def test_mixed(self):
        word2idx, idx2word, W_normed = self._make_normed()
        result = analogy("cat", "mat", "sat", word2idx, idx2word, W_normed)
        wrong  = next(w for w in word2idx if w != result and w not in ("cat", "mat", "sat"))
        tests  = [
            ("cat", "mat", "sat", result),   # correct
            ("cat", "mat", "sat", wrong),    # incorrect
            ("cat", "mat", "sat", "oov"),    # skipped
        ]
        acc, correct, total = eval_analogies(tests, word2idx, idx2word, W_normed)
        assert total == 2
        assert correct == 1
        assert acc == 0.5


# ============================================================
# Integration test
# ============================================================

class TestIntegration:
    def test_loss_decreases(self):
        """Train on tiny corpus, verify loss goes down."""
        from main import train
        W_in, word2idx, idx2word, loss_history = train(
            CORPUS, embedding_dim=5, window_size=1,
            num_negatives=2, learning_rate=0.1, num_epochs=100, seed=42,
            min_count=1,
        )
        assert W_in is not None
        assert loss_history[-1] < loss_history[0]

    def test_similar_words_cluster(self):
        """After training, 'cat' and 'on' should be more similar than 'cat' and 'mat'.
        Reason: cat and on share 2 context words (the, sat), while cat and mat share only 1 (the).
        """
        from main import train
        W_in, word2idx, idx2word, _ = train(
            CORPUS, embedding_dim=5, window_size=1,
            num_negatives=2, learning_rate=0.1, num_epochs=100, seed=42,
            min_count=1,
        )
        sim_cat_on  = cosine_similarity(W_in[word2idx["cat"]], W_in[word2idx["on"]])
        sim_cat_mat = cosine_similarity(W_in[word2idx["cat"]], W_in[word2idx["mat"]])
        assert sim_cat_on > sim_cat_mat, (
            f"Expected cat-on ({sim_cat_on:.3f}) > cat-mat ({sim_cat_mat:.3f})"
        )
