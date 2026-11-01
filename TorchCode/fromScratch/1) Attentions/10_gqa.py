import torch.nn as nn
import torch
from torch_judge import check

class GroupQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.head_dim = d_model // num_heads
        self.q_per_kv = num_heads // num_kv_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, self.head_dim * num_kv_heads)
        self.W_v = nn.Linear(d_model, self.head_dim * num_kv_heads)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):

        B, S, _ = x.shape
        Q = self.W_q(x).reshape((B, S, self.num_heads, self.head_dim)).transpose(1, 2) #(B, num_heads, S, head_dim)
        K = self.W_k(x).reshape((B, S, self.num_kv_heads, self.head_dim)).transpose(1, 2) #(B, num_kv_heads, S, head_dim)
        V = self.W_v(x).reshape((B, S, self.num_kv_heads, self.head_dim)).transpose(1, 2) #(B, num_kv_heads, S, head_dim)

        K = torch.repeat_interleave(K, self.q_per_kv, dim = 1) #(B, num_heads, S, head_dim)
        V = torch.repeat_interleave(V, self.q_per_kv, dim = 1) #(B, num_heads, S, head_dim)

        d_k = K.shape[-1]

        raw_weights = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5) #(B, num_heads, S, S)
        att_weights = torch.softmax(raw_weights, dim = -1) #(B, num_heads, S, S)

        hidden = att_weights @ V #(B, num_heads, S, head_dim)
        hidden = hidden.transpose(1, 2).reshape((B, S, self.d_model)) #(B, S, d_model)

        return self.W_o(hidden)  #(B, S, d_model)
        

def main():
    check('gqa')

if __name__ == "__main__":
    main()

#### AGAIN ####
class GroupQueryAttention(nn.Module):

    def __init__(self, d_model, num_heads, num_kv_heads):
        """
        we have more Q than KV, so a single KV will be shared by more (n_q / n_kv) Qs 
        """
        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.kv_per_q = num_heads // num_kv_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.head_dim)

        self.W_o = nn.Linear(d_model, d_model)


    def forward(self, x):
        """
        x: (B, seq_len, d_model)
        """

        B, seq_len, _ = x.shape

        Q = self.W_q(x).reshape((B, seq_len, self.num_heads, self.head_dim)).transpose(1, 2) #(B, num_heads, seq_len, head_dim)
        K = self.W_k(x).reshape((B, seq_len, self.num_kv_heads, self.head_dim)).transpose(1, 2) #(B, num_kv_heads, seq_len, head_dim)
        V = self.W_v(x).reshape((B, seq_len, self.num_kv_heads, self.head_dim)).transpose(1, 2) #(B, num_kv_heads, seq_len, head_dim)

        K = torch.repeat_interleave(K, self.kv_per_q, dim = 1) #(B, num_heads, seq_len, head_dim)
        V = torch.repeat_interleave(V, self.kv_per_q, dim = 1) #(B, num_heads, seq_len, head_dim)

        d_k = K.shape[-1]
        raw_weights = (Q @ K.transpose(-1, -2)) / (d_k ** 0.5)

        att_weights = torch.softmax(raw_weights, dim = -1)

        hidden = (att_weights @ V).transpose(1, 2).reshape((B, seq_len, self.d_model))

        return self.W_o(hidden)