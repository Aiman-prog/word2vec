# Code Explanation
---

## 1. One-hot vector

The naive approach is a **one-hot vector**: a vector of zeros with a single 1 at the word's position.

```
Vocabulary: [cat, sat, mat, the, on]

"cat" → [1, 0, 0, 0, 0]
"mat" → [0, 0, 1, 0, 0]
```

Two problems:
- **No similarity**: the dot product of any two one-hot vectors is 0. The representation says "cat" and "mat" are exactly as different as "cat" and "the", which is clearly wrong.
- **High dimensionality**: a 71,000-word vocabulary means 71,000-dimensional vectors. Almost all zeros.

Word2Vec's solution: train a lookup table that maps each word to a dense vector of maybe 100 numbers, where **similar words end up with similar vectors**. These are called **word embeddings**.

After training:
```
"cat"    → [0.21, -0.54, 0.03, ...]   # 100 numbers
"kitten" → [0.19, -0.51, 0.07, ...]   # similar!
"car"    → [-0.44, 0.12, 0.88, ...]   # different
```

The key insight: words that appear in similar contexts tend to have similar meanings, so training on context prediction naturally produces meaningful vectors.

---

## 2. The skip-gram task

Skip-gram is a self-supervised task: given a **center word**, predict its **context words**.

```
Sentence:  "the cat sat on the mat"
                    ↑
              center word: "sat"
              window = 1

Context words to predict: "cat", "on"
```

We slide this window across the entire corpus, generating millions of (center, context) pairs.

**Example — "the cat sat on the mat", window=2:**

| Center | Context words within window | Pairs generated |
|--------|-----------------------------|-----------------|
| the    | cat, sat                    | (the,cat) (the,sat) |
| cat    | the, sat, on                | (cat,the) (cat,sat) (cat,on) |
| **sat**| **the, cat, on, the**       | **(sat,the) (sat,cat) (sat,on) (sat,the)** |
| on     | cat, sat, the, mat          | (on,cat) (on,sat) (on,the) (on,mat) |
| the    | sat, on, mat                | (the,sat) (the,on) (the,mat) |
| mat    | on, the                     | (mat,on) (mat,the) |

18 pairs from 6 tokens. On text8 (~17M tokens, window=5) this produces tens of millions of pairs per epoch.

Why does this work? Words that appear in similar contexts tend to mean similar things. By training to predict context, the model is forced to encode meaning.

**→ Code:** `data.py` — `generate_training_pairs()`. Rather than looping over every token, it loops over each offset `1..window_size` and collects pairs via array slicing:

```python
for offset in range(1, window_size + 1):
    left_c  = token_ids[offset:]   # centers (right of context)
    left_x  = token_ids[:-offset]  # contexts
    right_c = token_ids[:-offset]  # centers (left of context)
    right_x = token_ids[offset:]   # contexts
```

With `random_window=True` (used at training time in `main.py`), each center word gets a random radius sampled from `Uniform(1, window_size)` — closer neighbours appear in more pairs on average.

---

## 3. The lookup table (embedding matrix)

The model has one learnable matrix: `W_in` of shape `(V, d)`, where `V` is the vocabulary size and `d` is the embedding dimension (e.g. 300).

Row `i` of `W_in` is the embedding for word `i`. The "forward pass" for a center word is just a row lookup — no matrix multiply, no activation function:

```python
v_c = W_in[center_index]   # shape: (d,) — one row out of V rows
```

**How the vector gets built:**

1. `W_in` starts as random noise — every word gets a random d-dimensional vector (initialised in `main.py` as `(rand - 0.5) / d`)
2. For each training pair `(sat, cat)`, the gradient tells us: *push `W_in["sat"]` and `W_out["cat"]` closer together*
3. For each negative pair `(sat, king)`: *push them apart*
4. After millions of such nudges, words that share many context words end up with similar rows in `W_in` — not because we told the model what "similar" means, but because the loss function forced it

The vector for "sat" is never explicitly constructed — it *emerges* from thousands of small gradient updates, each one adjusting that single row by a tiny amount.

**Two matrices: input vectors and output vectors**

Following Mikolov's terminology, every word has two separate vector representations:

- `W_in` holds the **input vector** for each word, used when the word is the center (the word doing the predicting)
- `W_out` holds the **output vector** for each word, used when the word is a context or negative target (the word being predicted toward)

Both are updated during training. Only `W_in` is kept afterwards. It is the representation shaped by the prediction task, and is what we query when looking up nearest neighbours or solving analogies.

---

## 4. Why not use the full softmax?

The "natural" objective would be: for each center word, predict a probability distribution over the entire vocabulary.

$$P(\text{context} = w \mid \text{center}) = \frac{e^{v_c \cdot u_w}}{\sum_{w'=1}^{V} e^{v_c \cdot u_{w'}}}$$

The denominator sums over all V words. With millions of pairs, this is too slow.

---

## 5. Negative sampling 

Instead of predicting which word is the context, turn it into a binary question: **"Is this a real (center, context) pair, or a fake one?"**

- The **real pair** (center, context) should get score → 1
- A few **fake pairs** (center, random word) should get score → 0

The random words drawn for fake pairs are called **negative samples**.

```
Real:  ("sat", "cat")  → label 1
Fake:  ("sat", "king") → label 0
Fake:  ("sat", "the")  → label 0
Fake:  ("sat", "car")  → label 0
```

This turns the problem into a binary classification problem. In this implementation, `k=5` negatives per real pair.

The score for a pair is the dot product of their embeddings: $v_c \cdot u_w$. Higher means more likely to be real.

**→ Code:** Negatives are sampled once per batch in `main.py`:

```python
neg_samples = np.random.choice(vocab_size, size=(bsz, num_negatives), p=noise_dist)
```

`noise_dist` is built in `data.py` — `get_noise_distribution()` — using `P(w) ∝ count(w)^0.75`. The 0.75 exponent smooths the distribution so rare words get sampled more often than their raw frequency would suggest, giving them more gradient signal during training.

---

## 6. The loss function

We want dot products to be large for real pairs and small for fake ones. We use sigmoid to squash scores into (0, 1), then take the log-likelihood:

$$L = -\log \sigma(v_c \cdot u_o) - \sum_{i=1}^{k} \log \sigma(-v_c \cdot u_i)$$

where $\sigma(x) = \frac{1}{1+e^{-x}}$.

Breaking it down:
- $-\log \sigma(v_c \cdot u_o)$: minimised when $v_c \cdot u_o$ is large (real pair gets high score). 
- $-\log \sigma(-v_c \cdot u_i)$: minimised when $v_c \cdot u_i$ is small (fake pair gets low score). 

```python
# In model.py:
s_pos  = np.sum(V_batch * U_batch, axis=1)          # dot products, real pairs
s_negs = np.einsum('bd,bkd->bk', V_batch, U_neg)    # dot products, fake pairs

loss = -np.log(sigmoid(s_pos)) - np.sum(np.log(sigmoid(-s_negs)), axis=1)
```

---

## 7. Gradient derivation — step by step

This is where calculus comes in. To update the embeddings via gradient descent, we need $\frac{\partial L}{\partial v_c}$, $\frac{\partial L}{\partial u_o}$, and $\frac{\partial L}{\partial u_i}$.

One useful fact first — the derivative of sigmoid:

$$\frac{d}{dx}\sigma(x) = \sigma(x)(1 - \sigma(x))$$

**Gradient w.r.t. the context vector $u_o$ (real pair)**

Apply chain rule to $-\log \sigma(v_c \cdot u_o)$:

$$\frac{\partial L}{\partial u_o}
= -\frac{1}{\sigma(v_c \cdot u_o)} \cdot \sigma(v_c \cdot u_o)(1 - \sigma(v_c \cdot u_o)) \cdot v_c$$

The $\sigma$ terms cancel:

$$= -(1 - \sigma(v_c \cdot u_o))\, v_c = (\sigma(v_c \cdot u_o) - 1)\, v_c$$

Intuition: if the dot product is already high (model is confident), $\sigma \approx 1$, gradient $\approx 0$ — no update needed. If the dot product is low (model is wrong), $\sigma \approx 0$, gradient $\approx -v_c$ — big update to push $u_o$ toward $v_c$.

**Gradient w.r.t. each negative context vector $u_i$**

Apply chain rule to $-\log \sigma(-v_c \cdot u_i)$. Note the inner $-1$ from the chain rule:

$$\frac{\partial L}{\partial u_i}
= -\frac{1}{\sigma(-v_c \cdot u_i)} \cdot \sigma(-v_c \cdot u_i)(1 - \sigma(-v_c \cdot u_i)) \cdot (-v_c)$$

Using $1 - \sigma(-x) = \sigma(x)$:

$$= \sigma(v_c \cdot u_i)\, v_c$$

Intuition: if the dot product is high (model wrongly thinks this is a real pair), $\sigma \approx 1$, big update to push $u_i$ away from $v_c$.

**Gradient w.r.t. the center vector $v_c$**

$v_c$ appears in all terms of $L$, so sum the contributions:

$$\frac{\partial L}{\partial v_c}
= (\sigma(v_c \cdot u_o) - 1)\, u_o + \sum_{i=1}^{k} \sigma(v_c \cdot u_i)\, u_i$$

```python
# In model.py — these three lines implement the three expressions above:
grad_u_o    = (sig_pos - 1)[:, None] * V_batch
grad_u_negs = sig_negs[:, :, None]   * V_batch[:, None, :]
grad_v_c    = (sig_pos - 1)[:, None] * U_batch \
            + np.einsum('bk,bkd->bd', sig_negs, U_neg)
```

---

## 8. SGD update

Once we have gradients, we subtract them (scaled by learning rate) from the relevant rows:

```python
W_in[center_index]   -= lr * grad_v_c
W_out[context_index] -= lr * grad_u_o
W_out[neg_indices]   -= lr * grad_u_negs
```

Notice that only the rows involved in this batch are updated — the rest of the matrix stays unchanged. This is what makes training feasible: each step touches `1 + 1 + k` rows out of 71,000.

The implementation uses `np.add.at` instead of direct indexing, because the same row can appear multiple times in a batch (e.g. the word "the" might be a center word twice) — `np.add.at` accumulates both gradient contributions correctly.

---

## 9. The training loop

```
for each epoch:
    subsample tokens (drop frequent words like "the")
    generate (center, context) pairs
    shuffle pairs
    for each mini-batch of 256 pairs:
        draw 5 negative samples per pair
        forward pass → compute loss
        backward pass → compute gradients
        SGD update → adjust W_in and W_out
```

**Why subsample frequent words?** Words like "the" appear so often that without subsampling, most training pairs involve them — and "the" co-occurs with everything, so it doesn't carry useful signal. Mikolov's formula keeps each token with probability `min(1, sqrt(t·N/f))`. A word appearing 1% of the time gets discarded ~90% of the time.

**Why decay the learning rate?** Early in training the model makes big mistakes, so big steps are helpful. Later, when embeddings are close to their final values, big steps cause oscillation. Linear decay from 0.025 to 0.0000025 over 15 epochs keeps training stable.

---

## 10. Why do the embeddings work?

After training, words that appear in similar contexts have vectors that point in similar directions. You can verify this with:

**Nearest neighbours** — find the words whose vectors have the highest cosine similarity to a query:
```
computer → computers, hardware, machines, computing, applications
france   → french, paris, vexin, belgium, maintenon
```

**Analogy arithmetic** — semantic relationships are encoded as vector differences:
```
vec("paris") - vec("france") ≈ vec("london") - vec("england")

So: vec("france") + (vec("paris") - vec("france")) + vec("england") ≈ vec("london")
    → france : paris :: england : london  
```

This works because the model is forced to encode meaning in a consistent direction: the "capital city" relationship pushes the vector in the same direction regardless of which country you start from.

---

