import torch.nn as nn
import torch
from torch_judge import check

def causal_attention(Q, K, V):
    """
    Q: (B, S_q, d_q)
    K: (B, S_k, d_k) --> d_k = d_q required
    V: (B, S_k, d_v)

    return: (B, S_q, d_v)
    """
    B, S_k, d_k = K.shape
    S_q = Q.shape[1]
    raw_weights = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)

    mask = torch.tril(torch.ones((B, S_q, S_k))) == 0

    raw_weights = torch.masked_fill(raw_weights, mask, float("-inf"))
    att_weights = torch.softmax(raw_weights, dim = -1)

    return att_weights @ V


def main():
    check('causal_attention')

if __name__ == "__main__":
    main()