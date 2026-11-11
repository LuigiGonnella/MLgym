from torch_judge import check
import torch.nn as nn
import torch

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        self.linear = nn.Linear(in_features, out_features, bias=True).requires_grad_(False)
        deltaA = (torch.randn((rank, in_features)) * ((2 / in_features) ** 0.5))
        self.lora_A = nn.Parameter(deltaA, requires_grad=True)
        deltaB = torch.zeros((out_features, rank))
        self.lora_B = nn.Parameter(deltaB, requires_grad=True)

    def forward(self, x):

        x_hidden = self.linear(x)
        lora = self.alpha / self.rank * (x @ self.lora_A.T) @ self.lora_B.T

        return x_hidden + lora



def main():
    check('lora')

if __name__ == "__main__":
    main()
