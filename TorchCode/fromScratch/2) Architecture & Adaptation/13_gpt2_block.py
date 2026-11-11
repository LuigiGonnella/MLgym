import torch
import torch.nn as nn
from torch_judge import check

class GPT2Block(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.mlp = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        
        xl = self.ln1(x)

        def causal_att(x):
            B, S_q, _ = x.shape

            Q = self.W_q(x).reshape((B, S_q, self.num_heads, self.head_dim)).transpose(1, 2) #(B, num_heads, S_q, head_dim)
            K = self.W_k(x).reshape((B, S_q, self.num_heads, self.head_dim)).transpose(1, 2) #(B, num_heads, S_q, head_dim) in causal att prefill S_q == S_k
            V = self.W_v(x).reshape((B, S_q, self.num_heads, self.head_dim)).transpose(1, 2) #(B, num_heads, S_q, head_dim)

            d_k = K.shape[-1]
            raw_weights = (Q @ K.transpose(-1, -2) / (d_k ** 0.5))

            mask = torch.tril(torch.ones((S_q, S_q))) == 0
            raw_weights = torch.masked_fill(raw_weights, mask, float("-inf"))

            att_weights = torch.softmax(raw_weights, dim = -1)
            hidden = att_weights @ V #(B, num_heads, S_q, head_dim)
            hidden = hidden.transpose(1, 2).reshape((B, S_q, self.d_model)) #(B, S_q, d_model)

            return self.W_o(hidden)

        xa = x + causal_att(xl)

        xal = self.ln2(xa)

        return xa + self.mlp(xal)

def main():
    check('gpt2_block')

if __name__ == "__main__":
    main()
