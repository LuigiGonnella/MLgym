import torch
from torch_judge import check


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits: (B, nc)
    targets: (B, )
    """

    B = logits.shape[0]
    logit_targets = logits[torch.arange(B), targets] #(B, )
    max_logits = torch.max(logits, dim = -1, keepdim=True).values #(B, 1)
    logits_shifted = logits - max_logits #(B, nc)
    logsumexp_shifted = torch.log(torch.sum(torch.exp(logits_shifted), dim = -1, keepdim=True)) #(B, 1)
    logsumexp = max_logits + logsumexp_shifted #(B, 1)

    ce_batches =  -(logit_targets - logsumexp.squeeze(1)) #(B, ) and (B, )

    return torch.mean(ce_batches)

def main():
    check("cross_entropy")

if __name__ == "__main__":
    main()