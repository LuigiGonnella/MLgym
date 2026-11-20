import torch
from torch_judge import check

def beam_search(log_prob_fn, start_token, max_len, beam_width, eos_token) -> list(int):

    beams = [(0.0, [start_token])]

    for _ in range(max_len - 1):
        candidates = []

        for prob, seq in beams:

            if seq[-1] == eos_token:
                continue #do not extend ended sequence

            logprobs = log_prob_fn(seq)
            topk = min(beam_width, logprobs.shape[-1])

            best_probs, best_toks = torch.topk(logprobs, topk, dim = -1)

            for p, tok in zip(best_probs, best_toks):
                candidates.append((prob + p, seq + [tok.item()]))

        candidates.sort(reverse=True)
        beams = candidates[:beam_width].copy()

        if beams[0][1][-1] == eos_token:
            break #end search if best sequence ended

    return beams[0][1] #return best sequence
    
    # 🧪 Debug
def simple_fn(tokens):
    lp = torch.full((5,), -10.0)
    lp[min(len(tokens), 4)] = 0.0
    return lp


def main():
    seq = beam_search(simple_fn, start_token=0, max_len=5, beam_width=2, eos_token=4)
    print('Sequence:', seq)
    check('beam_search')

if __name__ == "__main__":
    main()

