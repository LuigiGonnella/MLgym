import torch.nn as nn
import torch
from torch_judge import check

def sliding_window_attention(Q, K, V, window_size):
    #we want that each query i gives attention to each key j where |i - j| < window_size

    _, S_k, d_k = K.shape
    S_q = Q.shape[1]

    raw_weights = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)

    q_idx = torch.arange(S_q).unsqueeze(1) #(S_q, 1)
    k_idx = torch.arange(S_k).unsqueeze(0) #(1, S_k)

    mask = torch.abs(q_idx - k_idx) > window_size #(S_q, S_k)

    raw_weights = torch.masked_fill(raw_weights, mask, float("-inf"))

    att_weights = torch.softmax(raw_weights, dim = -1)

    return att_weights @ V


def main():
    check('sliding_window')

if __name__ == "__main__":
    main()