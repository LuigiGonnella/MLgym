import torch
import torch.nn as nn
from torch_judge import check 

class LinearRegression:
    def closed_form(self, X: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        $$\theta = (X_{aug}^T X_{aug})^{-1} X_{aug}^T y$$
        """

        X_aug = torch.cat([X, torch.ones((X.shape[0], 1))], dim=1) #add ones so the multiplication with theta (containing bias as last dim) will be x1w1 + ... + xdwd + 1*b
        theta = torch.linalg.solve(X_aug.T @ X_aug, X_aug.T @ y)

        w = theta[:-1]
        b = theta[-1]

        return (w, b)

    def gradient_descent(self, X, y, lr = 0.01, steps = 1000) -> tuple[torch.Tensor, torch.Tensor]:
        d_in = X.shape[1]
        B = X.shape[0]
        w = torch.randn((1, d_in)) * (1 / (d_in ** 2)) #(1, d_in)
        b = torch.randn(()) #()
        y = y.unsqueeze(1) #(B, 1)

        for _ in range(steps):

            pred = X @ w.T + b #(B, 1)
            err = pred - y #(B, 1)

            loss = torch.mean(err ** 2) #()

            dw = (2 / B) * err.T @ X #(1, d_in)
            db = (2 / B) * torch.sum(err) #()

            w -= lr * dw
            b -= lr * db

        return (w, b)

    def nn_linear(self, X, y, lr = 0.01, steps = 1000) -> tuple[torch.Tensor, torch.Tensor]:
        d_in = X.shape[1]
        fcl = nn.Linear(d_in, 1, bias=True)
        y = y.unsqueeze(1) #(B, 1)
        optimizer = torch.optim.SGD(fcl.parameters(), lr=lr)

        for _ in range(steps):
            preds = fcl(X) #(B, 1)

            optimizer.zero_grad()
            loss = torch.mean((preds - y) ** 2)
            loss.backward()
            optimizer.step()

        return (fcl.weight, fcl.bias)

def main():
    check("linear_regression")

if __name__ == "__main__":
    main()