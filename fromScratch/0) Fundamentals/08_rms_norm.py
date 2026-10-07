import torch
from torch_judge import check


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps= 1e-5
) -> torch.Tensor:
    """
    x: (B, d_in)
    gamma: (d_in, )

    return: (B, d_in)
    """


    squared_mean = torch.mean(torch.square(x), dim = -1, keepdim=True)
    rms = torch.sqrt(squared_mean + eps)

    xnorm = x / rms

    return xnorm * weight

def main():
    check("rmsnorm")

if __name__ == "__main__":
    main()