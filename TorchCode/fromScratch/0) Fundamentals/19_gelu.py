import torch
from torch_judge import check
import torch.nn as nn


def my_gelu(x: torch.Tensor) -> torch.Tensor:

    cdf = 0.5 * (1 + torch.erf(x / (2 ** 0.5)))
    return x * cdf


def main():
    check("gelu")

if __name__ == "__main__":
    main()