import torch
from torch_judge import check
import torch.nn as nn

class MyDropout(nn.Module):
    def __init__(self, p=0.5):
        """
        p: probability of having zero
        """

        super().__init__()
        if not 0 <= p < 1:
            raise ValueError("p must satisfy 0 <= p < 1")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.training:
            mask = torch.rand_like(x) <= self.p
            
            x = x.masked_fill(mask, 0.0) * (1 / (1 - self.p))
        
            return x

        return x



def main():
    check("dropout")

if __name__ == "__main__":
    main()