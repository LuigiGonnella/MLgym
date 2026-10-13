import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_judge import check

class KVCacheAttention(nn.Module):
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

    def forward(self, x, cache = None):
        #cache : (past_K, past_V), each is (B, num_heads, S, head_dim)

        B, S, _ = x.shape
        Q = self.W_q(x).reshape((B, S, self.num_heads, self.head_dim)).transpose(1, 2) #(B, num_heads, S, head_dim)
        K = self.W_k(x).reshape((B, S, self.num_heads, self.head_dim)).transpose(1, 2) #(B, num_heads, S, head_dim)
        V = self.W_v(x).reshape((B, S, self.num_heads, self.head_dim)).transpose(1, 2) #(B, num_heads, S, head_dim)

        if cache is not None:
            K = torch.cat([cache[0], K], dim = 2)
            V = torch.cat([cache[1], V], dim = 2)

        d_k = K.shape[-1]
        raw_weights = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5) #(B, S_q, S_k)

        S_k = K.shape[2]
        S_q = Q.shape[2]
        past_len = S_k - S_q

        q_idx = torch.arange(past_len, S_k).unsqueeze(1) #(S_q, 1)
        k_idx = torch.arange(S_k).unsqueeze(0) #(1, S_k)

        mask = q_idx < k_idx #true when q is looking into the future, so we will mask them --> (S_q, S_k)

        raw_weights = torch.masked_fill(raw_weights, mask, float("-inf"))

        att_weights = torch.softmax(raw_weights, dim = -1) #(B, num_heads, S_q, S_k)

        hidden = att_weights @ V #(B, num_heads, S_q, head_dim)
        hidden = hidden.transpose(1, 2).reshape((B, S, self.d_model))

        return (self.W_o(hidden), (K, V))



def main():
    check('kv_cache')

if __name__ == "__main__":
    main()