import os
import json
import time
import numpy as np
from tqdm.auto import tqdm
from data import build_vocab, subsample_tokens, generate_training_pairs, get_noise_distribution
from model import forward_and_backward, sgd_update
from evaluate import find_nearest, analogy, normalize_embeddings, eval_analogies, load_analogy_dataset


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
          learning_rate=0.1, num_epochs=100, seed=42, min_count=5, batch_size=256,
          random_window=False):
    """
    Full training pipeline for skip-gram with negative sampling.

    Steps:
        1. Build vocabulary (min_count filter) and noise distribution
        2. Initialise W_in and W_out with small random values
        3. Pre-tokenize corpus to integer IDs; build per-word keep probabilities
           for frequent-word subsampling (Mikolov t=1e-5 formula)
        4. For each epoch:
            a. Subsample tokens (stochastically drop frequent words)
            b. Regenerate (center, context) pairs from the subsampled sequence
            c. Shuffle pairs, then process in mini-batches:
               - Draw k negatives per pair from the noise distribution
               - Forward pass: compute SGNS loss
               - Backward pass: compute gradients via chain rule
               - SGD update with per-step linearly decayed learning rate
            d. Log average loss for the epoch
        5. Return trained W_in embeddings

    Args:
        corpus: string, the training text
        embedding_dim: int, size of word vectors (d)
        window_size: int, context window radius
        num_negatives: int, number of negative samples per pair (k)
        learning_rate: float, SGD learning rate
        num_epochs: int, number of training passes
        seed: int, random seed for reproducibility
        random_window: bool, sample per-center window radius (Mikolov 2013 §3.1)

    Returns:
        W_in: trained input embeddings, shape (V, d)
        word2idx: dict mapping word -> index
        idx2word: dict mapping index -> word
        loss_history: list of per-epoch average losses
    """
    np.random.seed(seed)

    # Setup: vocabulary, noise distribution 
    word2idx, idx2word, word_counts, total_count = build_vocab(corpus, min_count=min_count)
    vocab_size = len(word2idx)
    print(f"Vocabulary size: {vocab_size:,}")

    noise_dist = get_noise_distribution(word_counts, word2idx)

    # Initialise embeddings with small uniform random values 
    W_in  = (np.random.rand(vocab_size, embedding_dim) - 0.5) / embedding_dim
    W_out = (np.random.rand(vocab_size, embedding_dim) - 0.5) / embedding_dim

    # Pre-tokenize corpus to int IDs 
    raw_token_ids = np.array(
        [word2idx[t] for t in corpus.lower().split() if t in word2idx], dtype=np.int32
    )

    # Pre-build keep_probs indexed by word ID.
    # Formula: min(1, sqrt(t * N / f))  where t=1e-5,
    # N=total token count, f=word frequency. Words above the threshold are
    # kept with probability < 1; very frequent words are discarded most of the time.
    keep_probs = np.array(
        [min(1.0, np.sqrt(1e-5 * total_count / word_counts[idx2word[i]]))
         for i in range(vocab_size)],
        dtype=np.float32
    )

    # Pre-compute a stable epoch budget from the unsubsampled corpus (matches Mikolov:
    # LR is tied to tokens visited including subsampled-away ones, not pairs generated).
    full_pairs_per_epoch = len(generate_training_pairs(raw_token_ids, window_size,
                                                       random_window=random_window))
    total_steps = num_epochs * full_pairs_per_epoch
    global_step = 0

    loss_history = []

    for epoch in range(num_epochs):
        # Subsampling is only meaningful on large corpora; skip it for tiny datasets.
        if total_count >= 10000:
            token_ids = subsample_tokens(raw_token_ids, keep_probs)
        else:
            token_ids = raw_token_ids

        # Regenerate pairs from the (possibly subsampled) token sequence
        pairs = generate_training_pairs(token_ids, window_size, random_window=random_window)
        np.random.shuffle(pairs)

        total_loss = 0.0
        num_pairs  = len(pairs)

        # LR sweeps the full epoch's budget even if subsampling left fewer pairs.
        # np.linspace distributes the budget evenly across however many pairs exist this epoch,
        # matching Mikolov: LR progress counts all tokens including subsampled-away ones.
        lr_schedule = learning_rate * (
            1.0 - np.linspace(global_step, global_step + full_pairs_per_epoch,
                              num=num_pairs, endpoint=False) / total_steps
        )
        lr_schedule = np.maximum(lr_schedule, learning_rate * 0.0001)
        global_step += full_pairs_per_epoch  # advance by full epoch budget, not num_pairs

        pairs_done = 0
        epoch_start = time.time()

        pbar = tqdm(range(0, num_pairs, batch_size),
                    desc=f"Epoch {epoch + 1}/{num_epochs}",
                    unit="batch", leave=False)

        for b in pbar:
            sl  = slice(b, min(b + batch_size, num_pairs))

            # Get actual number of pairs in this slice (might be less than batch_size at epoch end).
            bsz = sl.stop - sl.start

            neg_samples = np.random.choice(vocab_size, size=(bsz, num_negatives), p=noise_dist)

            loss_batch, grad_v_c, grad_u_o, grad_u_negs = forward_and_backward(
                pairs[sl, 0], pairs[sl, 1], neg_samples, W_in, W_out
            )
            sgd_update(W_in, W_out, pairs[sl, 0], pairs[sl, 1], neg_samples,
                       grad_v_c, grad_u_o, grad_u_negs, lr_schedule[sl])

            total_loss += loss_batch.sum()
            pairs_done += bsz
            pbar.set_postfix(loss=f"{total_loss / pairs_done:.4f}")

        elapsed = time.time() - epoch_start
        mins, secs = divmod(int(elapsed), 60)
        time_str   = f"{mins}m {secs:02d}s" if mins else f"{secs}s"

        if num_pairs == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}  (no pairs after subsampling — skipped)")
            loss_history.append(0.0)
            continue

        avg_loss = total_loss / num_pairs
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}  avg_loss={avg_loss:.4f}  ({time_str})")

    return W_in, word2idx, idx2word, loss_history


if __name__ == "__main__":
    CORPUS_FILE = "text8"
    MODEL_PATH  = "model/model"  

    os.makedirs("model", exist_ok=True)

    if os.path.exists(f"{MODEL_PATH}.npy"):
        W_in, word2idx, idx2word = load_model(MODEL_PATH)
    else:
        print(f"Loading corpus from '{CORPUS_FILE}'...")
        with open(CORPUS_FILE, "r") as f:
            corpus = f.read()

        W_in, word2idx, idx2word, _ = train(
            corpus,
            embedding_dim=300,
            window_size=5,
            num_negatives=5,
            learning_rate=0.025,
            num_epochs=15,
            random_window=True,
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
        ("man",   "king",   "woman"),
        ("france","paris",  "england"),
        ("good",  "better", "bad"),
    ]
    for a, b, c in tests:
        if all(w in word2idx for w in (a, b, c)):
            result = analogy(a, b, c, word2idx, idx2word, W_normed)
            print(f"  {a} : {b}  ::  {c} : {result}")

    ANALOGY_FILE = "questions-words.txt"
    if os.path.exists(ANALOGY_FILE):
        ANALOGY_TESTS = load_analogy_dataset(ANALOGY_FILE)
        print(f"\n--- Analogy accuracy ({len(ANALOGY_TESTS):,} questions, Google Analogy dataset) ---")
    else:
        print("\n--- Analogy accuracy (16-question fallback; download questions-words.txt for full benchmark) ---")
        ANALOGY_TESTS = [
            # semantic: capital-country
            ("berlin", "germany", "paris",   "france"),
            ("berlin", "germany", "rome",    "italy"),
            ("berlin", "germany", "madrid",  "spain"),
            ("berlin", "germany", "athens",  "greece"),
            ("berlin", "germany", "tokyo",   "japan"),
            # semantic: gender
            ("man",    "king",    "woman",   "queen"),
            ("man",    "actor",   "woman",   "actress"),
            ("man",    "father",  "woman",   "mother"),
            ("man",    "husband", "woman",   "wife"),
            ("man",    "son",     "woman",   "daughter"),
            # syntactic: comparative
            ("good",   "better",  "bad",     "worse"),
            ("good",   "better",  "big",     "bigger"),
            ("good",   "better",  "fast",    "faster"),
            # syntactic: past tense
            ("walk",   "walked",  "run",     "ran"),
            ("walk",   "walked",  "go",      "went"),
            ("look",   "looked",  "play",    "played"),
        ]
    acc, correct, total = eval_analogies(ANALOGY_TESTS, word2idx, idx2word, W_normed)
    print(f"  {correct}/{total} correct  ({acc * 100:.1f}%)")
