import torch
import torch.nn as nn

config = {
    'dims' : [8, 128, 128, 16, 2],
    'lr' : 1e-3
}


def build_model(config):
    dims, lr = config['dims'], config['lr']

    layers = []

    for i in range(len(dims) - 1):
        layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]

    layers.append(nn.Linear(dims[-2], dims[-1]))

    model = nn.Sequential(layers)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    return model, optimizer



def batch_norm(x,
    gamma, #(d_in)
    beta, #(d_in)
    running_mean, #(B, d_in)
    running_var, #(B, d_in)
    eps=1e-5,
    momentum=0.1,
    training=True,
) -> torch.Tensor:
    
    correct_mean = running_mean
    correct_var = running_var
    
    if training:

        with torch.no_grad():

            batch_mean = torch.mean(x, dim = 0, keepdim=True)
            batch_var = torch.var(x, dim = 0, keepdim=True, unbiased=False)

            running_mean[:] = (1 - momentum) * running_mean + momentum * batch_mean
            running_var[:] = (1 - momentum) * running_var + momentum * batch_var

        correct_mean = batch_mean
        correct_var = batch_var
        
    x_norm = gamma * [(x - correct_mean) / torch.sqrt((correct_var + eps))] + beta

    return x_norm

def training_loop(model, optimizer, dataloader, config):
    epochs = config['epochs']
    loss_fn = config['loss_fn']
    save_every = config['save_every']

    B, d_in = next(iter(dataloader))[0].shape

    step = 0
    for _ in range(epochs):
        running_mean = torch.zeros((B, d_in))
        running_var = torch.zeros((B, d_in))

        for batch in dataloader:
            step += 1
            samples, targets = batch
            
            samples = batch_norm(samples, torch.randn((d_in, ), requires_grad=True), torch.randn((d_in, ), requires_grad=True), running_mean, running_var)

            optimizer.zero_grad()

            pred = model(samples)
            loss = loss_fn(pred, targets)

            loss.backward()
            optimizer.step()

            if step % save_every == 0:
                checkpoint = {
                    'model_state_dict' : model.state_dict(),
                    'optimizer_state_dict' : optimizer.state_dict(),
                    'normalization' : {
                        'running_mean' : running_mean,
                        'running_var' : running_var
                    },
                    'step': step,
                    'config' : config
                }

                torch.save(checkpoint, f"checkpoint_{step}.pth")

def load_model(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location='cuda',
        weights_only='True'
    )

    config = checkpoint['config']

    #architecture
    model, optimizer = build_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])

    #load weights
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model,optimizer

    #than with no_grad() we can geenrate preds, after having normalized with training = False with batch_norm

