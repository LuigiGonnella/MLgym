import torch
from torch_judge import check

class SimpleLinear:
    def __init__(self, in_features: int, out_features: int):

        #xavier init
        self.weight = (torch.randn((out_features, in_features)) * (1 / (in_features ** 2))).requires_grad_()

        self.bias = torch.randn((out_features,), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = x @ self.weight.T + self.bias

        return out


def main():
    check("linear")

if __name__ == "__main__":
    main()