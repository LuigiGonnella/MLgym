import torch
from torch_judge import check

# 1) Initialize:
#   self.merges = [] --> list of pairs, useful because dureing the "encode" we will follow this order of merges
#   self.single_chars = set() --> set of single symbols, useful to understand if a text is tokenizable or not ('unrecognized symbol')

# 2) Training:
#   - given a corpus we will decompose it in single chars, with the last char of each word followed by the terminator "</w>", filling self.single_chars
#   - once we did so, we wil run `num_merges` iters where we will merge the most_frequent consecutive pair, excluding the merges ACROSS DIFFERENT WORDS filling self.num_merges

# 3) Encoding:
#   - we obtain the set of single symbols of the given text, if this - self.single_chars is not empty --> unrecognized symbol
#   - Otherwise, following self.merges order, we try to merge that pair on the text in single chars, updating the text itself
#   - At the end, we return the final text, with all the found merges
from collections import Counter

class SimpleBPE:
  def __init__(self) -> None: 
    self.merges = []
    self.single_chars = set()

  def _merge(self, text, pair) -> list[str]:
    i = 0
    n = len(text)
    res = []
    while i < n - 1:
      if (text[i], text[i + 1]) == pair:
        new_tok = text[i] + text[i + 1]
        res.append(new_tok)
        i += 2
      else:
        res.append(text[i])
        i += 1

    if i < n:
      res.append(text[i])

    return res

  def train(self, corpus: list[str], num_merges: int) -> None: 
    tokens = []
    for word in corpus:
      tokens.extend([c for c in word])
      tokens.append("</w>")

    self.single_chars = set(tokens)

    if len(tokens) < 2:
      return # no merge can happen

    for _ in range(num_merges):
        
      pairs = [(a, b) for a, b in zip(tokens[:-1], tokens[1:]) if not a.endswith("</w>")]

      if not pairs:
        return 

      freq = Counter(pairs)
      most_freq = min(pairs, key=lambda x: (-freq[x], x))
      

      self.merges.append(most_freq)
      tokens = self._merge(tokens, most_freq)
      
  def encode(self, text: str) -> list[str]: 
    unknown = set([c for word in text.split() for c in word]) - self.single_chars

    if unknown:
      raise ValueError("Unknown symbol detected, cannot encode this text.")

    code = []
    for word in text.split():
      tokens = list(word) + ["</w>"]

      for pair in self.merges:
        tokens = self._merge(tokens, pair)

      code.extend(tokens)

    return code


def main():
    # 🧪 Debug
  bpe = SimpleBPE()
  bpe.train(['low', 'low', 'low', 'lower', 'newest', 'widest'], num_merges=10)
  print('Merges:', bpe.merges[:5])
  print('Encode:', bpe.encode('low lower'))
  check("bpe")

if __name__ == "__main__":
    main()