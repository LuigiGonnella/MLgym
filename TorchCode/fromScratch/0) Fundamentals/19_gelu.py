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



#### AGAIN ####

def my_gelu(x: torch.Tensor) -> torch.Tensor:
    """
    GeLU activation is similiar to ReLU but it is defined as the cumulative density function of a Normal variable time the input w.r.t. input x --> x * P[z <= x]
    So for ~ 0 values, the activation is ~ 0, even if for small values it is higher than ReLU (= 0) since h cdf will be pretty high ( ~ 0.5)
    For << 0 values, the activation goes nearer and nearer to zero
    For >> 0 values the activation goes neare and nearer to 1, so x * cdf(x) ~ x, even if slower than ReLU ( = x)
    The cdf of a normal is approximated by the formula 0.5 * (1 + erf(x / sqrt(2)))
    """

    cdf = 0.5 * (1 + torch.erf((x / (2 ** 0.5))))

    return x * cdf


