# Implement the **ReLU** (Rectified Linear Unit) activation function from scratch.

# $$\text{ReLU}(x) = \max(0, x)$$
import torch
from torch_judge import check

def relu(x: torch.Tensor) -> torch.Tensor:

    return torch.where(condition = x > 0, input = x, other = 0)




def main():
    check("relu")

if __name__ == "__main__":
    main()