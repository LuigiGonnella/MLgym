import torch
from torch_judge import check

def kaiming_init(weight: torch.Tensor) -> torch.Tensor:
    d_in = weight.shape[-1]
    weight[:] = (torch.randn_like(weight) * ((2 / d_in) ** 0.5)).requires_grad_()

    return weight

def main():
    check("weight_init")

if __name__ == "__main__":
    main()