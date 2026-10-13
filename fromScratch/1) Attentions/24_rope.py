import torch.nn as nn
import torch
from torch_judge import check

def apply_rope(q, k):
    """
    q: (B, S, d)
    k: (B, S, d)
    """
    _, S, d = q.shape
    halfd = d // 2

    pos = torch.arange(S).unsqueeze(1) #(S, 1)
    i = torch.arange(halfd).unsqueeze(0) #(1, d//2)

    theta = pos / (10000 ** (2 * i / d)) #(S, d//2)
    theta = torch.cat([theta, theta], dim = 1) #(S, d)

    cost = torch.cos(theta)
    sint = torch.sin(theta)

    q1 = q[:, :, :halfd] 
    q2 = q[:, :, halfd:]

    k1 = k[:, :, :halfd] 
    k2 = k[:, :, halfd:]

    q_rot = torch.cat([-q2, q1], dim = -1)
    k_rot = torch.cat([-k2, k1], dim = -1)

    q_rope = q * cost + q_rot * sint
    k_rope = k * cost + k_rot * sint

    return (q_rope, k_rope)


def main():
    check('rope')

if __name__ == "__main__":
    main()