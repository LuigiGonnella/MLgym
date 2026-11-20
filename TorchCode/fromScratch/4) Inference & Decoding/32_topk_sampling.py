import torch
from torch_judge import check

def sample_top_k_top_p(logits, top_k = 0, top_p = 1.0, temperature = 1.0) -> int:
    """
    logits: (V,) but in general (B, V)
    """
    if temperature == 0.0: #avoid zero-division
        return torch.argmax(logits, dim = -1)

    sorted_logits, sorted_indices = torch.sort(logits / temperature, dim = -1, descending=True)

    if top_k > 0:
        sorted_logits[..., top_k:] = float("-inf") #truncate logits, maintain only top_k

    if top_p < 1.0:
        probs = torch.softmax(sorted_logits, dim=-1)
        cum_sum = torch.cumsum(probs, dim = -1)

        remove = cum_sum > top_p
        #maintain first token that reaches/pass top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False

        sorted_logits[..., remove] = float("-inf") #truncate logits, eliminate tokens that exceed top_p (except for the first)

    probs = torch.softmax(sorted_logits, dim=-1)
    chosen_idx = torch.multinomial(probs, num_samples=1)

    #return torch.gather(sorted_indices, dim = -1, index = chosen_idx) --> general case sorted-indices of shape (B, V)
    return sorted_indices[chosen_idx].item()

def main():
    check('topk_sampling')

if __name__ == "__main__":
    main()