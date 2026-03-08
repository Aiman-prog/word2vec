# Word2Vec from Scratch

Skip-gram word2vec with negative sampling, implemented in pure NumPy — no ML framework.
Trained on the text8 corpus (17 M tokens, 71 k-word vocabulary) in ~15 epochs.

---

## Results

Trained model output on text8 using `main.py` defaults (embedding dim=300, window=5, 5 negatives, 15 epochs, batch size=256):

**Nearest neighbours** (cosine similarity):

| Query | Top-5 neighbours |
|-------|-----------------|
| `computer` | computers, hardware, machines, computing, applications |
| `france` | french, paris, vexin, belgium, maintenon |
| `science` | scientific, fiction, computer, contributions, research |

**Analogies** (a : b :: c : ?):

```
man   : king   ::  woman   : queen    ✓
france: paris  ::  england : london   ✓
walk  : walked ::  run     : ran      ✓
```

**Analogy accuracy**: 27.3% overall on the full Google Analogy dataset (~19.5 k questions, 14 categories). Semantic categories reach 29.5% (capital-common-countries: 65.4%, nationality-adjective: 81.9%); syntactic categories average 25.7%, limited by text8's relatively small size (~17 M tokens).

---

## Architecture

```
Corpus ──▶ [data.py]  build_vocab          → word2idx, word_counts
                      subsample_tokens     → filtered token IDs (Mikolov t=1e-5)
                      generate_training_pairs → (center, context) pairs
                      get_noise_distribution  → unigram^0.75 noise dist

        ──▶ [main.py] train loop           batched SGD, linear LR decay
                        │ forward_and_backward ← [model.py]
                        │ sgd_update           ← [model.py]
                        ▼
                      W_in, W_out  (V × d weight matrices)

        ──▶ [evaluate.py] normalize_embeddings → unit-norm W_in
                          find_nearest         → cosine-similarity lookup
                          analogy              → 3CosAdd arithmetic
                          eval_analogies       → accuracy over test suite
```

---

## Design decisions

- **Batched forward/backward pass** — processes pairs in mini-batches using vectorised NumPy ops (`einsum`) instead of looping over individual pairs. Trades memory for speed; a single matrix multiply replaces thousands of dot products.

- **Linear learning rate decay** — LR anneals from `lr` down to `lr × 0.0001` over the full training run. The schedule is computed from a stable denominator (unsubsampled pair count) so it doesn't drift when subsampling varies epoch to epoch.

- **Batch-level negative sampling** — negatives are drawn per batch rather than pre-allocating the entire epoch upfront. Keeps memory usage proportional to batch size rather than total training pairs.

- **Frequent-word subsampling** — Mikolov's `t=1e-5` formula gives each token a keep probability of `min(1, sqrt(t·N/f))`, where `f` is the word's frequency. High-frequency words get keep probabilities well below 1 (e.g. "the" is discarded most of the time), which improves representation of rare words. Skipped automatically on small corpora (< 10 k tokens) where it would be harmful.

- **Two weight matrices (W_in / W_out)** — each word has two vectors: `W_in` is used when the word is a center word, `W_out` when it is a context word. Both capture semantic information. Only `W_in` is kept after training because that is what Mikolov's original implementation returned.

- **Pre-normalised embeddings** — `normalize_embeddings()` is called once after training to scale all word vectors to unit length. This turns cosine similarity into a plain dot product, so nearest-neighbour and analogy searches run as fast matrix multiplications with no per-query normalisation overhead.

- **Gradient check** — the chain-rule derivation was verified by hand first (see §Gradient derivation below), then confirmed computationally: `test_gradient_check` compares analytical gradients against central finite differences (ε=1e-5) to ensure the implementation matches the derivation exactly.

---

## Gradient derivation

The derivation below was worked out by hand before writing the backward pass — a useful check that the chain rule was applied correctly before trusting the gradient check.

The SGNS loss for a single (center $v_c$, context $u_o$) pair with $k$ negative samples $u_1 \ldots u_k$:

$$L = -\log \sigma(v_c \cdot u_o) - \sum_{i=1}^{k} \log \sigma(-v_c \cdot u_i)$$

**Gradient w.r.t. the positive context vector $u_o$**

$$\frac{\partial L}{\partial u_o}
= -\frac{1}{\sigma(v_c \cdot u_o)} \cdot \sigma(v_c \cdot u_o)(1 - \sigma(v_c \cdot u_o)) \cdot v_c
= (\sigma(v_c \cdot u_o) - 1)\, v_c$$

**Gradient w.r.t. each negative context vector $u_i$**

$$\frac{\partial L}{\partial u_i}
= -\frac{1}{\sigma(-v_c \cdot u_i)} \cdot \sigma(-v_c \cdot u_i)(1 - \sigma(-v_c \cdot u_i)) \cdot (-v_c)
= \sigma(v_c \cdot u_i)\, v_c$$

where the last step uses the identity $1 - \sigma(-x) = \sigma(x)$.

**Gradient w.r.t. the center vector $v_c$** (sum over all output vectors via chain rule)

$$\frac{\partial L}{\partial v_c}
= (\sigma(v_c \cdot u_o) - 1)\, u_o + \sum_{i=1}^{k} \sigma(v_c \cdot u_i)\, u_i$$

These three expressions map directly to `grad_u_o`, `grad_u_negs`, and `grad_v_c` in `model.py:63-71`.
The gradient check in `test_word2vec.py` verifies all three against central finite differences ($\varepsilon = 10^{-5}$).

---

## Alternatives considered

**CBOW** — predicts the center word from averaged context vectors; faster to train but produces weaker embeddings for rare words because the averaging destroys distributional signal.

**Hierarchical softmax** — replaces negative sampling with a Huffman tree over the vocabulary, reducing per-step cost to O(log V) without sampling noise. More complex to implement; negative sampling is faster in practice for large vocabularies.

**GloVe** — builds a sparse global co-occurrence matrix in one corpus pass, then factorises it over many training epochs. Captures corpus-level statistics directly, at the cost of materialising and storing the co-occurrence counts before any gradient step.

---

## Usage

> **Quickstart**: open `demo.ipynb` for an end-to-end interactive walkthrough — smoke test, full training, visualisations, and evaluation all in one place. For the mathematical background (gradient derivation, loss function, why negative sampling), see `walkthrough.md`.

### Prerequisites

```bash
pip install -r requirements.txt
```

### Dataset

Download the text8 corpus and place it in the project directory:

```bash
curl -O http://mattmahoney.net/dc/text8.zip && unzip text8.zip
```

### Train

```bash
python3 main.py
```

Trains skip-gram with negative sampling on text8 and prints nearest neighbours and analogy results when done. Key hyperparameters (set in `main.py`):

| Parameter | Value |
|-----------|-------|
| Embedding dim | 300 |
| Window size | 5 |
| Negative samples | 5 |
| Learning rate | 0.025 |
| Epochs | 15 |
| Min count | 5 |
| Batch size | 256 |

### Smoke test (no dataset required)

Trains on a 6-word corpus to verify the full pipeline end-to-end:

```bash
python3 smoke_test.py
```

### Unit tests

```bash
python3 -m pytest test_word2vec.py -v
```

---

## File structure

| File | Description |
|------|-------------|
| `data.py` | Vocabulary building, subsampling, training pair generation, noise distribution |
| `model.py` | Forward pass, SGNS loss, gradients (`forward_and_backward`), SGD update |
| `evaluate.py` | Cosine similarity, nearest neighbours, analogies, analogy accuracy (`eval_analogies`) |
| `main.py` | Training loop, model save/load |
| `demo.ipynb` | Interactive walkthrough: smoke test on a 6-word corpus, full text8 training with loss and LR-decay curves, embedding visualisations (heatmaps, PCA), nearest neighbours, vector arithmetic, and analogy accuracy. **Start here to see the model train and evaluate end-to-end.** |
| `walkthrough.md` | Theory companion: distributional hypothesis, skip-gram task, the embedding matrix, why full softmax is too slow, negative sampling, the SGNS loss, step-by-step gradient derivation, SGD update, and why the embeddings work. **Read this for the mathematical reasoning behind the implementation.** |
| `test_word2vec.py` | 30 unit tests including gradient check |
