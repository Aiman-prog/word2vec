import numpy as np


def sigmoid(x):
    """
    Numerically stable sigmoid function.
    Clip input to [-10, 10] to prevent overflow in exp().

    Args:
        x: scalar or numpy array

    Returns:
        sigmoid(x): same shape as input, values in (0, 1)
    """
    x = np.clip(x, -10, 10)
    return 1.0 / (1.0 + np.exp(-x))


def forward_and_backward(center_indices, context_indices, neg_indices, W_in, W_out):
    """
    Compute loss and gradients for a batch of (center, context) pairs in a single pass.

    Loss (SGNS objective, per pair):
        L = -log(sigmoid(s_pos)) - sum_k[ log(sigmoid(-s_neg)) ]

    Args:
        center_indices:  int array of shape (B,), center word indices
        context_indices: int array of shape (B,), positive context word indices
        neg_indices:     int array of shape (B, k), negative sample indices
        W_in:  input (center) embedding matrix,  shape (V, d)
        W_out: output (context) embedding matrix, shape (V, d)

    Returns:
        loss:        per-pair losses,                  shape (B,)
        grad_v_c:    gradient for center vectors,      shape (B, d)
        grad_u_o:    gradient for context vectors,     shape (B, d)
        grad_u_negs: gradient for negative vectors,    shape (B, k, d)
    """
    # Forward pass: embedding lookup
    V_batch = W_in[center_indices]    # (B, d)
    U_batch = W_out[context_indices]  # (B, d)
    U_neg   = W_out[neg_indices]      # (B, k, d)

    # compute dot product scores for positive and negative pairs
    s_pos  = np.sum(V_batch * U_batch, axis=1)            # (B,)
    s_negs = np.einsum('bd,bkd->bk', V_batch, U_neg)      # (B, k)

    # apply sigmoid to scores to get probabilities
    sig_pos  = sigmoid(s_pos)    # (B,)
    sig_negs = sigmoid(s_negs)   # (B, k)

    # --- Calculate loss (per pair) ---
    loss = -np.log(sig_pos) - np.sum(np.log(sigmoid(-s_negs)), axis=1)  # (B,)

    # --- Calculate gradients (via chain rule on the loss) ---
    grad_u_o    = (sig_pos - 1)[:, None] * V_batch                        # (B, d)
    grad_u_negs = sig_negs[:, :, None] * V_batch[:, None, :]              # (B, k, d)
    grad_v_c    = (sig_pos - 1)[:, None] * U_batch \
                + np.einsum('bk,bkd->bd', sig_negs, U_neg)                # (B, d)

    return loss, grad_v_c, grad_u_o, grad_u_negs


def sgd_update(W_in, W_out, center_indices, context_indices, neg_indices,
               grad_v_c, grad_u_o, grad_u_negs, lr):
    """
    Update only the embedding rows involved in this batch.
    Modifies W_in and W_out in-place.
    
    Args:
        W_in, W_out: embedding matrices (modified in-place)
        center_indices:  int array of shape (B,)
        context_indices: int array of shape (B,)
        neg_indices:     int array of shape (B, k)
        grad_v_c:    shape (B, d)
        grad_u_o:    shape (B, d)
        grad_u_negs: shape (B, k, d)
        lr: learning rate array of shape (B,)
    """
    W_in[center_indices]   -= lr[:, None]       * grad_v_c      # (B, d)
    W_out[context_indices] -= lr[:, None]       * grad_u_o      # (B, d)
    W_out[neg_indices]     -= lr[:, None, None] * grad_u_negs   # (B, k, d)
