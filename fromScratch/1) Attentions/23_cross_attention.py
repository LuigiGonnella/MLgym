import torch.nn as nn
import torch
from torch_judge import check

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0

        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, xq, xv): #xq from DECODER, xv from ENCODER

        B, S_q, _ = xq.shape
        S_k = xv.shape[1]

        Q = self.W_q(xq).reshape((B, S_q, self.num_heads, self.head_dim)).transpose(1, 2)
        K = self.W_k(xv).reshape((B, S_k, self.num_heads, self.head_dim)).transpose(1, 2)
        V = self.W_v(xv).reshape((B, S_k, self.num_heads, self.head_dim)).transpose(1, 2)

        d_k = K.shape[-1]
        raw_weights = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)
        att_weights = torch.softmax(raw_weights, dim = -1)

        hidden = att_weights @ V #(B, num_heads, S_q, head_dim)
        hidden = hidden.transpose(1, 2).reshape((B, S_q, self.d_model))

        return self.W_o(hidden)




def main():
    check('cross_attention')

if __name__ == "__main__":
    main()