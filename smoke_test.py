"""
Quick sanity check on a tiny corpus — no dataset download required.
Verifies that the training loop runs and embeddings capture context similarity.

Run with: python3 smoke_test.py
"""

from main import train
from evaluate import cosine_similarity

corpus = "the cat sat on the mat"

W_in, word2idx, idx2word = train(
    corpus,
    embedding_dim=5,
    window_size=1,
    num_negatives=2,
    learning_rate=0.1,
    num_epochs=100,
    min_count=1,
)

print("\n--- Cosine similarities after training ---")
cat = W_in[word2idx["cat"]]
mat = W_in[word2idx["mat"]]
on  = W_in[word2idx["on"]]

sim_cat_on  = cosine_similarity(cat, on)
sim_cat_mat = cosine_similarity(cat, mat)

# "cat" and "on" share 2 context words (the, sat)
# "cat" and "mat" share only 1 context word (the)
# so cat<->on should be more similar than cat<->mat
print(f"cat <-> on  : {sim_cat_on:.4f}  (share 'the' + 'sat', should be higher)")
print(f"cat <-> mat : {sim_cat_mat:.4f}  (share 'the' only)")
print(f"cat-on > cat-mat: {sim_cat_on > sim_cat_mat}")
