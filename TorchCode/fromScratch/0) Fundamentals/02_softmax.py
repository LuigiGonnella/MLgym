
import torch
from torch_judge import check

def my_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    x: (B, d_in)

    return (B, d_in)
    """

    xmax = torch.max(x, dim=dim, keepdim=True).values

    xshifted = x - xmax
    expshifted = torch.exp(xshifted)
    sumexp = torch.sum(expshifted, dim=dim, keepdim=True)

    return expshifted / sumexp


def main():
    check("softmax")

if __name__ == "__main__":
    main()


#### AGAIN ####
def my_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    x: (B, d)
    return: (B, d)
    """

    maxel = torch.max(x, dim = dim, keepdim=True).values #(B, 1)
    shifted = x - maxel #(B, d)

    num = torch.exp(shifted) #(B, d)
    den = torch.sum(shifted, dim = dim, keepdim=True) #(B, 1)

    return num / den
