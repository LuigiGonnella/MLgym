import torch
import torch.nn as nn
from torch_judge import check


class MixtureOfExperts(nn.Module):

    def __init__(self, d_model, d_ff, num_experts, top_k = 2):
        super().__init__()

        self.d_model = d_model

        self.router = nn.Linear(d_model, num_experts)
        self.experts = [nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)) for _ in range(num_experts)]
        self.experts = nn.ModuleList(self.experts)

        self.top_k = top_k

    def forward(self, x):
        """
        x: (B, n_toks, d_model)

        we want to perform a weighted sum for each token of each expert's output where the weight is the router's output
        """
        B, n_toks, _ = x.shape
        tot_toks = B * n_toks
        tokens = x.reshape((tot_toks, self.d_model))

        logits = self.router(tokens) #(tot_toks, num_experts)
        top_logits, top_indices = torch.topk(logits, k = self.top_k, dim = -1) #(tot_toks, top_k)
        top_weights = torch.softmax(top_logits, dim = -1) #(tot_toks, top_k)

        out = torch.zeros((tot_toks, self.d_model))
        for index, expert in enumerate(self.experts):

            tok_ids, exp_ids = torch.where(top_indices == index) #each of shape (n_selected, ) with n_selected <= tot_toks

            if tok_ids.shape[0] == 0: #no token selected this expert
                continue

            exp_weights = top_weights[tok_ids, exp_ids].unsqueeze(1) #(n_selected,1)
            toks = tokens[tok_ids, :] #(n_selected, d_model)

            outs = exp_weights * expert(toks) #(n_selected, d_model)
            out[tok_ids, :] += outs

        return out.reshape((B, n_toks, -1))



def main():
    check('moe')

if __name__ == "__main__":
    main()