import torch
from torch_judge import check

def accumulated_step(model, optimizer, loss_fn, micro_batches) -> float:

    tot_loss = 0.0
    optimizer.zero_grad()
    for batch in micro_batches:
        inputs, targets = batch

        preds = model(inputs)
        loss = loss_fn(preds, targets) / len(micro_batches)

        loss.backward() 
        tot_loss += loss.item()
    
    optimizer.step()

    return  tot_loss


def main():
    check("gradient_accumulation")

if __name__ == "__main__":
    main()