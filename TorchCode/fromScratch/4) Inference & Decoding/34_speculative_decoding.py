import torch
from torch_judge import check

def speculative_decode(target_probs, draft_probs, draft_tokens):
    """
    target_probs: (K, V)
    draft_probs: (K, V)
    draft_tokens: (K,)
    """
    K = draft_tokens.shape[-1]
    target_tok_probs = target_probs[torch.arange(K), draft_tokens] #(K,)
    draft_tok_probs = draft_probs[torch.arange(K), draft_tokens] #(K,)
    ratios = target_tok_probs / draft_tok_probs #(K,)

    tokens = []
    for i in range(K):
        accept = torch.rand(()).item() < ratios[i].item()

        if not accept: #get one of the tokens with 100% acceptance probability (target_p >= draft_p)
            new_probs = target_probs[i] - draft_probs[i]
            residual = torch.maximum(new_probs, torch.zeros_like(new_probs)) #(V,)
            residual /= torch.sum(residual, dim = -1, keepdim=True) 

            chosen_tok = torch.multinomial(residual, num_samples=1) #index == tok, searche dover all vocabulary
            tokens.append(chosen_tok.item())
            break

        tokens.append(draft_tokens[i].item())

    return tokens


def main():
    check('speculative_decoding')

if __name__ == "__main__":
    main()