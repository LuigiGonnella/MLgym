import torch

from torch_judge import check

def my_layer_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps = 1e-5) -> torch.Tensor:
    """
    x: (B, d_in)
    gamma: (d_in,)
    beta: (d_in,)

    return: (B, d_in)
    """

    mu = torch.mean(x, dim = -1, keepdim=True)
    var = torch.var(x, dim=  -1, keepdim=True, unbiased = False)

    xshifted = x - mu
    xnorm = xshifted / (torch.sqrt(var + eps))

    return xnorm * gamma + beta

def main():
    check("layernorm")

if __name__ == "__main__":
    main()