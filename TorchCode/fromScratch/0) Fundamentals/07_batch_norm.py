import torch
from torch_judge import check

def my_batch_norm(
    x,
    gamma,
    beta,
    running_mean,
    running_var,
    eps=1e-5,
    momentum=0.1,
    training=True,
) -> torch.Tensor:
    """
    x: (B, d_in)
    gamma: (d_in, )
    beta: (d_in, )

    return: (B, d_in)
    """

    batch_mean = torch.mean(x, dim = 0, keepdim=True)
    batch_var = torch.var(x, dim = 0, keepdim=True, unbiased = False)

    #inference time - use only running stats collected on training set
    correct_mean = running_mean
    correct_var = running_var

    if training: #collect running stats and use batch stats to normalize
        running_mean[:] = (1 - momentum) * running_mean + momentum * batch_mean
        running_var[:] = (1 - momentum) * running_var + momentum * batch_var

        correct_mean =batch_mean
        correct_var = batch_var


    xnorm = (x - correct_mean) / torch.sqrt(correct_var + eps)

    return xnorm * gamma + beta


def main():
    check("batchnorm")

if __name__ == "__main__":
    main()