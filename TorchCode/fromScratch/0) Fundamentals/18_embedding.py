import torch
from torch_judge import check
import torch.nn as nn


class MyEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        data = torch.randn((num_embeddings, embedding_dim))
        self.weight = nn.Parameter(data, requires_grad=True) 

    def forward(self, indices):
        return self.weight[indices]


def main():
    check("embedding")

if __name__ == "__main__":
    main()