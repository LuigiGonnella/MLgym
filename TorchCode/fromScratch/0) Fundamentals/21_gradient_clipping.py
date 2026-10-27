import torch
from torch_judge import check

def clip_grad_norm(parameters: torch.Tensor, max_norm: float) -> torch.Tensor:
    """
    parameters: (np, ...)

    return: (np, ...)
    """

    norm = torch.sqrt(
            sum(torch.sum(p.grad ** 2) for p in parameters) #different p can have different shapes, e.g. w: (out_f, in_f) and b: (out_f, )
            #so we need to compute the total norm as the sum over both dim (inner sum) and params (outer sum, python sum since we have a generator inside and not a tensor)
        )

    if norm > max_norm:
        scale = max_norm / norm
        for p in parameters:
            p.grad.mul_(scale) #in-place version of mul

    return norm

def main():
    check("gradient_clipping")

if __name__ == "__main__":
    main()
