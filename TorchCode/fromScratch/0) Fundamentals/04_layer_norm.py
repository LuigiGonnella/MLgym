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


#### AGAIN ####
def my_layer_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps = 1e-5) -> torch.Tensor:
    """
    x: (B, d)
    gamma: (d, )
    beta: (d, )
    out: (B, d)
    """

    mean = torch.mean(x, dim = -1, keepdim=True)
    var = torch.var(x, dim = -1, keepdim=True, unbiased=False) #exp. value of the estimator is not equal to the var, to be equal we should compute the var as the mean with (n - 1) as denominator

    x_norm = (x - mean) / ((var + eps) ** 0.5)

    return gamma * x_norm + beta


