import torch
from torch import nn
from torch_judge import check

class Int8Linear(nn.Module):
  def __init__(self, weight: torch.Tensor, bias: torch.Tensor = None): 
    """
    weigth: (out_feat, in_feat)
    bias: ()

    maxabs : 127 = x_old : x_new 
      --> x_new = x_old * 127 / maxabs --> x_new = x_old / scale --> scale = maxabs / 127
      --> x_old = x_new.float() * self.scale
    """
    super().__init__()

    self.scale = torch.max(torch.abs(weight), keepdim=True, dim = -1).values / 127.0
    weight_int8 = torch.round(weight / self.scale).to(torch.int8)    
    self.register_buffer("weight_int8", weight_int8) #no parameter, not trainable
    self.bias = bias
    
      
  def forward(self, x: torch.Tensor) -> torch.Tensor: 
    #dequantization
    weight = self.weight_int8.float() * self.scale
   
    
    return x @ weight.T + self.bias if self.bias is not None else x @ weight.T 


def main():
  # 🧪 Debug
  w = torch.randn(8, 4)
  q = Int8Linear(w)
  x = torch.randn(2, 4)
  print('Output:', q(x).shape)
  print('dtype:', q.weight_int8.dtype)
  print('Max quant error:', (w - q.weight_int8.float() * q.scale).abs().max().item())
  check("int8_quantization")

if __name__ == "__main__":
    main()