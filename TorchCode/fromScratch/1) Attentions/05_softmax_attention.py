import torch
from torch_judge import check

def scaled_dot_product_attention(
    Q: torch.Tensor,  # (batch, seq_q, d_k)
    K: torch.Tensor,  # (batch, seq_k, d_k)
    V: torch.Tensor,  # (batch, seq_k, d_v)
) -> torch.Tensor:   # (batch, seq_q, d_v)

    d_k = K.shape[-1]
    raw_weights = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5) #not probs --> (batch, seq_q, seq_k)
    att_weights = torch.softmax(raw_weights, dim = -1) #probs --> (batch, seq_q, seq_k)

    return att_weights @ V #weighted sum --> (batch, seq_q, d_v)


def main():
    check('attention')

if __name__ == "__main__":
    main()