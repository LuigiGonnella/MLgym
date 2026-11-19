import torch
from torch_judge import check

class MyAdam:
    def __init__(self, params, lr =1e-3, betas = (0.9, 0.999), eps = 1e-8):
        self.params = params #list of matrices
        self.lr = lr
        self.m = [torch.zeros_like(p) for p in params] #first momentum
        self.v = [torch.zeros_like(p) for p in params] #second momentum
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.t = [0] * len(params)

    def step(self):
        with torch.no_grad():

            for i, p in enumerate(self.params):

                if p.grad is None:
                    continue

                grad = p.grad
                self.t[i] += 1

                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad.square()

                m_hat = self.m[i] / (1 - self.beta1 ** self.t[i])
                v_hat = self.v[i] / (1 - self.beta2 ** self.t[i])

                p.add_(-self.lr * m_hat / ((v_hat ** 0.5) + self.eps))


    def zero_grad(self):

        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

def main():
    check('adam')

if __name__ == "__main__":
    main()
