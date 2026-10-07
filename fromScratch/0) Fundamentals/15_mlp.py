import torch
from torch_judge import check
import torch.nn as nn

class SwiGLUMLP(nn.Module):

    @staticmethod
    def _silu(x: torch.Tensor) -> torch.Tensor:
        return x / (1 + torch.exp(-x))


    def __init__(self, d_model, d_ff):
        super().__init__()

        self.gate_proj = nn.Linear(d_model, d_ff)
        self.up_proj = nn.Linear(d_model, d_ff)
        self.down_proj = nn.Linear(d_ff, d_model)


    def forward(self, x):

        return self.down_proj(self._silu(self.gate_proj(x)) * self.up_proj(x))

def main():
    check("mlp")

if __name__ == "__main__":
    main()
        