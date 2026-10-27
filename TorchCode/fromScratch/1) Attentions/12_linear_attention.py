import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_judge import check

def linear_attention(Q, K, V):
    #from O(S^2 * D) to O(S * D^2)

    def kernel(x):
        return F.elu(x) + 1

    num = kernel(Q) @ (kernel(K).transpose(-2, -1) @ V) #(B, S_q, d_v)
    den = kernel(Q) @ torch.sum(kernel(K), dim = 1).unsqueeze(-1) #(B, S_q, d_q) @ (B, d_k, 1) --> (B, S_q, 1)

    return num / (den + 1e-6) #(B, S_q, d_v)
    


def main():
    check('linear_attention')

if __name__ == "__main__":
    main()