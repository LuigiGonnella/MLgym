import torch
from torch_judge import check

def flash_attention(
    Q: torch.Tensor,  # (batch, seq_q, d_k)
    K: torch.Tensor,  # (batch, seq_k, d_k)
    V: torch.Tensor,  # (batch, seq_k, d_v)
    block_size=32, 
) -> torch.Tensor:   # (batch, seq_q, d_v)

    B, S_q, _ = Q.shape
    S_k = K.shape[1]

    Qblocks = [Q[:, i: i + block_size, :] for i in range(0, S_q, block_size)] #last block can have LESS than `block_size` tokens
    Kblocks = [K[:, i: i + block_size, :] for i in range(0, S_k, block_size)] #last block can have LESS than `block_size` tokens
    Vblocks = [V[:, i: i + block_size, :] for i in range(0, S_k, block_size)] #last block can have LESS than `block_size` tokens

    nk = len(Kblocks)

    Oblocks = [] #then we will cat

    for Qi in Qblocks:
        q_len = Qi.shape[1]

        Oi = torch.zeros_like(Qi) #(B, block_size, d_q)
        mi = torch.full((B, q_len, 1), float("-inf")) #max logit per row
        li = torch.zeros((B, q_len, 1)) #sum of exp(logit - max)

        for j in range(nk):
            Kj, Vj = Kblocks[j], Vblocks[j]
            d_k = Kj.shape[-1]

            raw_weights = (Qi @ Kj.transpose(-2, -1)) / (d_k ** 0.5) #(B, q_len, k_len)

            mij = torch.max(raw_weights, dim = -1, keepdim=True).values #(B, q_len, 1)
            fij = torch.exp(raw_weights - mij) #(B, q_len, k_len)
            lij = torch.sum(fij, dim = -1, keepdim=True) #(B, q_len, 1)

            mi_new = torch.maximum(mij, mi)

            old_scale = torch.exp(mi - mi_new) 
            new_scale = torch.exp(mij - mi_new) 

            li_new = old_scale * li + new_scale * lij

            Oi = (old_scale * li * Oi + new_scale * (fij @ Vj)) /  li_new

            li = li_new
            mi = mi_new

        Oblocks.append(Oi)
    
    return torch.cat(Oblocks, dim = 1) #cat along sequence dim



def main():
    check('flash_attention')

if __name__ == "__main__":
    main()