import torch.nn as nn
import torch
from torch_judge import check

class MultiHeadAttention(nn.Module):
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

    def forward(self, Q, K, V):

        B, S_k, _ = K.shape
        S_q = Q.shape[1]
        

        Qheads = self.W_q(Q).reshape((B, S_q, self.num_heads, self.head_dim)).transpose(1, 2) #(B, num_heads, S_q, head_dim)
        Kheads = self.W_k(K).reshape((B, S_k, self.num_heads, self.head_dim)).transpose(1, 2) #(B, num_heads, S_k, head_dim)
        Vheads = self.W_v(V).reshape((B, S_k, self.num_heads, self.head_dim)).transpose(1, 2) #(B, num_heads, S_k, head_dim) 

        d_k = Kheads.shape[-1]

        raw_weights = (Qheads @ Kheads.transpose(-2, -1)) / (d_k ** 0.5) #(B, num_heads, S_q, S_k)
        att_weights = torch.softmax(raw_weights, dim = -1) #same

        hidden = att_weights @ Vheads #(B, num_heads, S_q, head_dim)
        hidden = hidden.transpose(1, 2).reshape((B, S_q, self.d_model)) #num_heads * head_dim = d_model

        return self.W_o(hidden)

def main():
    check('mha')

if __name__ == "__main__":
    main()