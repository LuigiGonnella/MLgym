import torch
from torch_judge import check
import torch.nn as nn



class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim)


    def forward(self, x):
        """
        x: (B, in_channels, img_size, img_size) --> (B, n_patches, in_channels * patch_size * patch_size)
        """

        B = x.shape[0]

        x = x.reshape((B, self.num_patches, self.in_channels * self.patch_size * self.patch_size))
        emb = self.proj(x)

        return emb

def main():
    check('vit_patch')

if __name__ == "__main__":
    main()

