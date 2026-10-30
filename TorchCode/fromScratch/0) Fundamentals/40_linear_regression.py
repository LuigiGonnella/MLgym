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


#### AGAIN ####

class LinearRegression:
    def closed_form(self, X: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        theta = (X @ X.T)^(-1) X.T @ y
        """

        #bias is last dim of final theta, so add all ones as last dim of X, so the y = wixi + ... + b*1
        n_samples = X.shape[1]
        X_aug = torch.cat([X, torch.ones((n_samples, 1))], dim = -1)

        theta = torch.linalg.solve(X_aug @ X_aug.T, X.T @ y)

        return theta[:-1], theta[-1]

    def gradient_descent(self, X, y, lr = 0.01, steps = 1000) -> tuple[torch.Tensor, torch.Tensor]:

        B, d_in = X.shape #(B, d_in)

        y = y.view(-1, 1) #(B, 1)

        W = torch.zeros((1, d_in)) #(1, d_in)
        b = torch.zeros(()) #()

        for _ in range(steps):

            pred = X @ W.T + b #(B, 1)
            err = pred - y #(B, 1)

            loss = torch.mean(err ** 2)

            dW = (2 / B) * err.T @ X #(1, d_in)
            db = (2 / B) * torch.sum(err) #()

            W -= lr * dW
            b -= lr * db

        return W, b
        

    def nn_linear(self, X, y, lr = 0.01, steps = 1000) -> tuple[torch.Tensor, torch.Tensor]:
        
        _, d_in = X.shape #(B, d_in)
        y = y.view(-1, 1) #(B, 1)

        W = nn.Linear(in_features=d_in ,out_features=1, bias=True)
        optimizer = torch.optim.Adam(params=W.parameters())

        for _ in range(steps):

            pred = X @ W.T #(B, 1)
            err = pred - y #(B, 1)

            optimizer.zero_grad()
            loss = torch.mean(err ** 2)
            loss.backward()
            optimizer.step()

        return W.weight, W.bias
        