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


#### AGAIN ####

def accumulated_step(model, optimizer, loss_fn, micro_batches) -> float:
    """
    The idea is to perform more small gradient calculation, using less memory, to then accumulate them and perform an update step equivalent to the big one
    
    Loss big batch = (1 / n) * sum(loss_i)
    Loss many mini_batches = (1 / n_m) * sum(loss_mi) with loss_mi = (1 / num_samples) * sum(loss_i) --> mean over samples in each mini batch AND then mean over the number of mini batches

    --> they are equal if each mini_batch has the ssame dimensionality, if not, just sum the dim of each mini_batch to obtain tot_samples and then divide tot_sum_losses / tot_samples
    """


    loss = 0.0
    optimizer.zero_grad() #make sure no gradient still memorized
    for batch in micro_batches:

        samples, targets = batch

        pred = model(samples)
        loss_batch = loss_fn(pred, targets) #contains mean over samples in mini batch

        loss_batch.backward() #one small gradient computation, accumulated over all microbatches

        loss += (loss_batch.item() / len()) #compute second mean over mini batches, so the final loss is equal to the mean over total samples

    optimizer.step() #update after gradient cumulation

    return loss #should be equal to big step loss
