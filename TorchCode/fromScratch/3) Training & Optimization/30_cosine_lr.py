import torch
from torch_judge import check
import math

def cosine_lr_schedule(step, total_steps, warmup_steps, max_lr, min_lr=0.0):

    if step < warmup_steps:
        return max_lr * (step / warmup_steps)

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

def main():
    check('cosine_lr')

if __name__ == "__main__":
    main()
