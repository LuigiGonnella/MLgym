import torch
import numpy as np


def softmax(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: (B, dim)
    """

    max_logit = torch.max(logits, dim = -1, keepdim=True).values
    shifted = logits - max_logit

    probs = torch.exp(shifted) / torch.sum(torch.exp(shifted), dim = -1, keepdim=True)

    return probs

def grad_softmax(logits):
    _, dim = logits.shape

    probs = softmax(logits) #(B, dim)

    J = -probs.unsqueeze(-1) * probs.unsqueeze(1) #(B, dim, dim)
    J[:, torch.arange(dim), torch.arange(dim)] += probs

    return J

    

def softmax_ce(logits, y_true):
    """
    logits: (B, dim)
    y_true: (B, ) correct y only
    """

    #for each sample I have to compute - log(softmax) --> -logit_target + logsumexp
    B = logits.shape[0]

    logit_targets = logits[torch.arange(B), y_true] #(B,)

    #logsumexp is stable if we reduce the exponent by the maximum logit but without changin the final values
    #--> max_logit + logsum(exp(logit - max_logit))
    max_logits = torch.max(logits, dim = -1, keepdim=True).values #(B, 1)
    logits_shifted = logits - max_logits #(B, dim)

    logsum = torch.log(torch.sum(torch.exp(logits_shifted), dim = -1, keepdim=True)) #(B, 1)
    logsumexp = max_logits + logsum #(B, 1)

    loss = torch.mean(-logit_targets + logsumexp.squeeze(1)) #()

    return loss


def grad_softmax_ce(logits, y_true):
    B = logits.shape[0]
    grad = softmax(logits) #(B, dim)

    #grad of this loss w.r.t. each logit is
    grad[torch.arange(B), y_true] -= 1 #grad is pj - yj with yj = 1 (OHE) on true label and 0 otherwise
    grad *= (1 / B)

    return grad

def numerical_grad_softmax_ce(x, y_true, h=1e-5):
    B, D = x.shape
    grad = torch.zeros(B, D, dtype=x.dtype, device=x.device)

    for i in range(B): #per sample since we compute the CE loss as the mean
        for j in range(D):
            x_plus = x.clone()
            x_minus = x.clone()

            x_plus[i, j] += h
            x_minus[i, j] -= h

            p_plus = softmax_ce(x_plus, y_true)
            p_minus = softmax_ce(x_minus, y_true)

            grad[i, j] = (p_plus - p_minus) / (2 * h)

    return grad


def numerical_jacobian_softmax(x, h=1e-5):
    B, D = x.shape
    J = torch.zeros(B, D, D, dtype=x.dtype, device=x.device)

    for j in range(D):
        x_plus = x.clone()
        x_minus = x.clone()

        x_plus[:, j] += h
        x_minus[:, j] -= h

        p_plus = softmax(x_plus)
        p_minus = softmax(x_minus)

        J[:, :, j] = (p_plus - p_minus) / (2 * h)

    return J


#TEST gradients of softmax and softmax_ce implementations
def main():
    x = torch.randn(4, 5, dtype=torch.float64)
    y = torch.tensor([0, 3, 2, 4], dtype=torch.long)

    print("TESTING SOFTMAX JACOBIAN")
    J_analytical = grad_softmax(x)
    J_numerical = numerical_jacobian_softmax(x)

    print(torch.allclose(
        J_analytical, J_numerical,
        atol=1e-8, rtol=1e-5
    ))

    print("TESTING CE GRADIENT")
    grad_analytical = grad_softmax_ce(x, y)
    grad_numerical = numerical_grad_softmax_ce(x, y)

    print(torch.allclose(
        grad_analytical, grad_numerical,
        atol=1e-8, rtol=1e-5
    ))

    # Additional check against PyTorch autograd
    # My analytical Jacobian: (B, D, D)
    x_ref = x.clone().requires_grad_(True)
    J_analytical = grad_softmax(x_ref)

    # PyTorch full Jacobian: (B, D, B, D)
    J_full = torch.autograd.functional.jacobian(softmax, x)

    # Extract per-sample Jacobians: (B, D, D)
    idx = torch.arange(x.shape[0])
    J_autograd = J_full[idx, :, idx, :]

    print(torch.allclose(
        J_analytical, J_autograd,
        atol=1e-8, rtol=1e-5
    ))
    
    x_ref = x.clone().requires_grad_(True)
    loss_ref = torch.nn.functional.cross_entropy(x_ref, y)
    loss_ref.backward()

    print("TESTING CE AGAINST AUTOGRAD")
    print(torch.allclose(
        grad_analytical, x_ref.grad,
        atol=1e-8, rtol=1e-5
    ))

if __name__ == "__main__":
    main()
    