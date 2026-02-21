import os
import json
import time
import numpy as np
from data import build_vocab, subsample_tokens, generate_training_pairs, get_noise_distribution
from model import forward_and_backward, sgd_update
from evaluate import cosine_similarity, find_nearest, analogy, normalize_embeddings


def save_model(W_in, word2idx, path="model"):
    """
    Save trained embeddings and vocabulary to disk.
    Creates two files: <path>.npy (embeddings) and <path>.json (vocab).
    """
    np.save(f"{path}.npy", W_in)
    with open(f"{path}.json", "w") as f:
        json.dump(word2idx, f)
    print(f"Model saved to {path}.npy and {path}.json")


def load_model(path="model"):
    """
    Load embeddings and vocabulary from disk.
    Returns W_in, word2idx, idx2word — same as train().
    """
    W_in = np.load(f"{path}.npy")
    with open(f"{path}.json", "r") as f:
        word2idx = json.load(f)
    assert W_in.shape[0] == len(word2idx), (
        f"Model mismatch: embeddings have {W_in.shape[0]:,} rows but vocab has {len(word2idx):,} words. "
        "Delete the saved model and retrain."
    )
    idx2word = {i: w for w, i in word2idx.items()}
    print(f"Model loaded from {path}.npy and {path}.json  (vocab={len(word2idx):,}, dim={W_in.shape[1]})")
    return W_in, word2idx, idx2word


def train(corpus, embedding_dim=5, window_size=1, num_negatives=2,
          learning_rate=0.1, num_epochs=100, seed=42, min_count=5, batch_size=256):
    """
    Full training pipeline for skip-gram with negative sampling.

    Steps:
        1. Build vocabulary from corpus
        2. Generate all (center, context) training pairs
        3. Compute noise distribution for negative sampling
        4. Initialize W_in and W_out with small random values
        5. For each epoch:
            - Shuffle training pairs
            - For each pair: forward+backward pass, then SGD update
            - Track and print average loss
        6. Return trained embeddings

    Args:
        corpus: string, the training text
        embedding_dim: int, size of word vectors (d)
        window_size: int, context window radius
        num_negatives: int, number of negative samples per pair (k)
        learning_rate: float, SGD learning rate
        num_epochs: int, number of training passes
        seed: int, random seed for reproducibility

    Returns:
        W_in: trained input embeddings, shape (V, d)
        word2idx: dict mapping word -> index
        idx2word: dict mapping index -> word
        loss_history: list of per-epoch average losses
    """
    np.random.seed(seed)

    # --- Setup: vocabulary, noise distribution ---
    word2idx, idx2word, word_counts, total_count = build_vocab(corpus, min_count=min_count)
    vocab_size = len(word2idx)
    print(f"Vocabulary size: {vocab_size:,}")

    noise_dist = get_noise_distribution(word_counts, word2idx)

    # --- Initialise embeddings with small uniform random values ---
    # Dividing by embedding_dim keeps initial scores (dot products) small,
    # preventing sigmoid from saturating before training even starts
    W_in  = (np.random.rand(vocab_size, embedding_dim) - 0.5) / embedding_dim
    W_out = (np.random.rand(vocab_size, embedding_dim) - 0.5) / embedding_dim

    # Pre-tokenize corpus to int IDs once — avoids re-splitting 17M tokens every epoch
    raw_token_ids = np.array(
        [word2idx[t] for t in corpus.lower().split() if t in word2idx], dtype=np.int32
    )

    # Pre-build keep_probs indexed by word ID — replaces per-token dict lookup in subsampling
    keep_probs = np.array(
        [min(1.0, np.sqrt(1e-5 * total_count / word_counts[idx2word[i]]))
         for i in range(vocab_size)],
        dtype=np.float32
    )

    # Compute total_steps once from unsubsampled pairs — stable LR denominator
    # regardless of how subsampling varies num_pairs each epoch.
    total_steps = num_epochs * len(generate_training_pairs(raw_token_ids, window_size))
    global_step = 0

    loss_history = []

    # --- Training loop ---
    for epoch in range(num_epochs):
        # Subsample frequent words each epoch — new random draws every epoch
        # so the model sees different subsets and learns more robustly.
        # Subsampling is only meaningful on large corpora; t=1e-5 would discard
        # nearly everything on a tiny corpus, so we skip it when total_count < 10000.
        if total_count >= 10000:
            token_ids = subsample_tokens(raw_token_ids, keep_probs)
        else:
            token_ids = raw_token_ids

        # Regenerate pairs from the (possibly subsampled) token sequence
        pairs = generate_training_pairs(token_ids, window_size)
        np.random.shuffle(pairs)

        total_loss = 0.0
        num_pairs  = len(pairs)

        # Pre-compute LR schedule for this epoch using the stable global step counter
        lr_schedule = learning_rate * (1.0 - np.arange(global_step, global_step + num_pairs) / total_steps)
        lr_schedule = np.maximum(lr_schedule, learning_rate * 0.0001)  # floor
        global_step += num_pairs

        # Print progress every ~1% of batches (at least every batch for tiny corpora)
        num_batches    = (num_pairs + batch_size - 1) // batch_size
        report_every   = max(1, num_batches // 100)
        pairs_done     = 0
        epoch_start    = time.time()

        # Process pairs in batches — reduces Python function-call overhead ~250×
        # vs calling forward_and_backward once per pair.
        # Negatives are sampled per batch (256 × k × 4 bytes) rather than all at once
        # (num_pairs × k × 4 bytes ≈ up to 3 GB on text8).
        for b in range(0, num_pairs, batch_size):
            sl  = slice(b, min(b + batch_size, num_pairs))
            bsz = sl.stop - sl.start

            neg_samples = np.random.choice(vocab_size, size=(bsz, num_negatives), p=noise_dist)

            loss_batch, grad_v_c, grad_u_o, grad_u_negs = forward_and_backward(
                pairs[sl, 0], pairs[sl, 1], neg_samples, W_in, W_out
            )
            sgd_update(W_in, W_out, pairs[sl, 0], pairs[sl, 1], neg_samples,
                       grad_v_c, grad_u_o, grad_u_negs, lr_schedule[sl])

            total_loss += loss_batch.sum()
            pairs_done += bsz

            batch_idx = b // batch_size
            if batch_idx % report_every == 0:
                pct    = int(100 * pairs_done / num_pairs)
                filled = pct // 5
                bar    = "=" * filled + ">" + " " * (20 - filled)
                print(f"\r  Epoch {epoch + 1}/{num_epochs}  [{bar} {pct:3d}%]"
                      f"  loss={total_loss / pairs_done:.4f}", end="", flush=True)

        elapsed = time.time() - epoch_start
        mins, secs = divmod(int(elapsed), 60)
        time_str   = f"{mins}m {secs:02d}s" if mins else f"{secs}s"

        avg_loss = total_loss / num_pairs
        loss_history.append(avg_loss)
        print(f"\rEpoch {epoch + 1}/{num_epochs}  avg_loss={avg_loss:.4f}  ({time_str})          ")

    return W_in, word2idx, idx2word, loss_history


if __name__ == "__main__":
    # Path to the text8 dataset.
    # Download with: curl -O http://mattmahoney.net/dc/text8.zip && unzip text8.zip
    CORPUS_FILE = "text8"
    MODEL_PATH  = "model/model"  # saves model/model.npy + model/model.json

    os.makedirs("model", exist_ok=True)

    if os.path.exists(f"{MODEL_PATH}.npy"):
        # Skip training and load the saved model directly
        W_in, word2idx, idx2word = load_model(MODEL_PATH)
    else:
        print(f"Loading corpus from '{CORPUS_FILE}'...")
        with open(CORPUS_FILE, "r") as f:
            corpus = f.read()

        W_in, word2idx, idx2word, _ = train(
            corpus,
            embedding_dim=100,
            window_size=5,
            num_negatives=5,
            learning_rate=0.025,
            num_epochs=5,
        )
        save_model(W_in, word2idx, MODEL_PATH)

    W_normed = normalize_embeddings(W_in)

    print("\n--- Nearest neighbours ---")
    for query in ["king", "woman", "computer"]:
        if query in word2idx:
            print(f"\n{query}:")
            for word, score in find_nearest(query, word2idx, idx2word, W_normed):
                print(f"  {word:<15} {score:.4f}")

    print("\n--- Analogies (a : b :: c : ?) ---")
    tests = [
        ("man",   "king",   "woman"),   # → queen
        ("france","paris",  "england"), # → london
        ("good",  "better", "bad"),     # → worse
    ]
    for a, b, c in tests:
        if all(w in word2idx for w in (a, b, c)):
            result = analogy(a, b, c, word2idx, idx2word, W_normed)
            print(f"  {a} : {b}  ::  {c} : {result}")
