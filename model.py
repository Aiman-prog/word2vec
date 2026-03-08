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
    V_batch = W_in[center_indices]   
    U_batch = W_out[context_indices]  
    U_neg   = W_out[neg_indices]      

    # compute dot product scores for positive and negative pairs
    s_pos  = np.sum(V_batch * U_batch, axis=1)           
    # equivalent: (V_batch[:, None, :] * U_neg).sum(axis=-1)
    # or as a loop:
    #   for b in range(B):
    #       for j in range(k):
    #           s_negs[b, j] = np.dot(V_batch[b], U_neg[b, j])
    # For each pair in the batch we have k negative samples, so U_neg is (B, k, d) and the result is (B, k).
    # V_batch has shape (B, d), U_neg has shape (B, k, d).
    s_negs = np.einsum('bd,bkd->bk', V_batch, U_neg)

    # apply sigmoid to scores to get probabilities
    sig_pos  = sigmoid(s_pos)   
    sig_negs = sigmoid(s_negs)   

    # calculate loss (per pair) 
    loss = -np.log(sig_pos) - np.sum(np.log(sigmoid(-s_negs)), axis=1)  

    # calculate gradients (via chain rule on the loss) 
    grad_u_o    = (sig_pos - 1)[:, None] * V_batch                       
    grad_u_negs = sig_negs[:, :, None] * V_batch[:, None, :]              
    # equivalent: (sig_negs[:, :, None] * U_neg).sum(axis=1)
    # or as a loop:
    #   for b in range(B):
    #       for j in range(k):
    #           grad_v_c[b] += sig_negs[b, j] * U_neg[b, j]
    grad_v_c    = (sig_pos - 1)[:, None] * U_batch \
                + np.einsum('bk,bkd->bd', sig_negs, U_neg)

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
    np.add.at(W_in,  center_indices,  -(lr[:, None]       * grad_v_c))
    np.add.at(W_out, context_indices, -(lr[:, None]       * grad_u_o))
    np.add.at(W_out, neg_indices,     -(lr[:, None, None] * grad_u_negs))
