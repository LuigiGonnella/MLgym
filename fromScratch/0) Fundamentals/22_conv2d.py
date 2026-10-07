import torch
from torch_judge import check
import torch.nn.functional as F

def my_conv2d(x, weight, bias=None, stride=1, padding=0) -> torch.Tensor:
    """
    features <--> x: (B, Cin, H, W)
    kernel <--> weight: (Cout, Cin, kH, kW)

    return: (B, Cout, newH, newW)
    """

    B, Cin, H, W = x.shape
    Cout, _, kH, kW = weight.shape

    newH = (H - kH + 2 * padding) // stride + 1
    newW = (W - kW + 2 * padding) // stride + 1

    if padding:
        x = F.pad(x, (padding, padding, padding, padding), 'constant', 0.0) #pad tuple as (i, j, k, z)
        #pads by 'padding' 'elements, in last dim (first 2 tuple elements indicates last dim) 
        # on both directions (i, j) are the two directions
        #same in before last dim (k, z) are the two directions of before last dim
        #puts `0.0` as filling number

    out = torch.empty((B, Cout, newH, newW), dtype=x.dtype, device=x.device)

    #we need to multiply kernel for x, element by element for each batch and each Cout kernel --> sum that multiplication for each combination of batch and Cout

    weight = weight.unsqueeze(0) #(1, Cout, Cin, kH, kW)

    for i in range(newH):
        for j in range(newW):
            startH = i * stride
            startW = j * stride

            #get patch window
            patch = x[:, :, startH:startH + kH, startW:startW + kW] #(B, Cin, kH, kW)
            patch = patch.unsqueeze(1) #(B, 1, Cin, kH, kW)

            #multiply element-wise with kernel unsqeezed over batches --> (B, Cout, Cin, kH, kW) and sum over Cin, H, W to obtain the single value over (B, Cout)

            values = torch.sum(patch * weight, dim=(2, 3, 4)) #(B, Cout)
            out[:, :, i, j] = values

    if bias is not None:
        out += bias.view(1, -1, 1, 1) #(1, Cout, 1, 1) --> one bias per kernel broadcasted over each batch and each pixel

    return out


def main():
    check("conv2d")

if __name__ == "__main__":
    main()
