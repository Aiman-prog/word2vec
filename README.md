# Word2Vec from Scratch

Skip-gram word2vec with negative sampling, implemented in pure NumPy.

## Implementation highlights

- **Batched forward/backward pass** — processes pairs in mini-batches using vectorised NumPy ops (`einsum`) instead of looping over individual pairs, giving ~250× less Python overhead
- **Linear learning rate decay** — LR anneals from `lr` to `lr × 0.0001` over training, computed from a stable denominator (unsubsampled pair count) so the schedule doesn't drift when subsampling varies epoch to epoch
- **Batch-level negative sampling** — negatives are drawn per batch (256 × k × 4 bytes ≈ 5 KB) rather than pre-allocating the entire epoch upfront (up to 3 GB on text8)
- **Frequent-word subsampling** — Mikolov's `t=1e-5` formula discards high-frequency tokens with probability proportional to their excess frequency, improving representation of rare words; skipped automatically on small corpora
- **Pre-normalised embeddings** — `normalize_embeddings()` is called once after training to scale all word vectors to unit length, simplifying cosine similarity into a basic dot product; this removes the overhead of recomputing vector lengths on every query and allows nearest-neighbour and analogy searches to run as fast matrix multiplications
- **Gradient check** — `test_gradient_check` verifies analytical gradients against numerical finite differences (central differences, ε=1e-5) to confirm the chain rule derivation is correct

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Download the text8 corpus and place it in the project directory:

```bash
curl -O http://mattmahoney.net/dc/text8.zip && unzip text8.zip
```

## Run

```bash
python3 main.py
```

Trains skip-gram with negative sampling on text8 and prints nearest neighbours
for a few query words when done.

## Smoke test (no dataset required)

Trains on a 6-word corpus to verify the pipeline works end-to-end:

```bash
python3 smoke_test.py
```

## Tests

```bash
python3 -m pytest test_word2vec.py -v
```

## File structure

| File | Description |
|------|-------------|
| `data.py` | Vocabulary building, subsampling, training pair generation, noise distribution |
| `model.py` | Forward pass, SGNS loss, gradients (`forward_and_backward`), SGD update |
| `evaluate.py` | Cosine similarity, nearest neighbours, analogies, analogy accuracy (`eval_analogies`) |
| `main.py` | Training loop, model save/load |
| `demo.ipynb` | Interactive demo: smoke test, full text8 training, loss curve, nearest neighbours, analogies |
| `test_word2vec.py` | 27 unit tests including gradient check |

## Hyperparameters (text8)

| Parameter | Value |
|-----------|-------|
| Embedding dim | 100 |
| Window size | 5 |
| Negative samples | 5 |
| Learning rate | 0.025 |
| Epochs | 5 |
