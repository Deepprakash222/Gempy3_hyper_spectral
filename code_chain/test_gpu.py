import pyro
import pyro.distributions as dist
import torch
from pyro.infer import MCMC, NUTS
import time

# Enable MPS for PyTorch
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device =torch.device("cpu")

# Define a simple Bayesian linear regression model
def model(x, y=None):
    # Priors for weights and bias
    w = pyro.sample("w", dist.Normal(torch.zeros(1, device=device), torch.ones(1, device=device)))
    b = pyro.sample("b", dist.Normal(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)))

    # Likelihood
    mean = w * x + b
    with pyro.plate("data", x.shape[0]):
        pyro.sample("obs", dist.Normal(mean, torch.tensor(1.0, device=device)), obs=y)

# Generate some synthetic data
def main():
    torch.manual_seed(0)
    x = torch.linspace(0, 1, 10, device=device)
    true_w = 2.0
    true_b = -0.5
    y = true_w * x + true_b + 0.1 * torch.randn(x.size(), device=device)

    # Define the NUTS sampler
    nuts_kernel = NUTS(model)

    # Run MCMC
    
    mcmc = MCMC(nuts_kernel, num_samples=500, warmup_steps=20000, num_chains=7)
    mcmc.run(x, y)

    # Summarize results
    mcmc.summary()
    
    # samples
    samples = mcmc.get_samples(group_by_chain=True)
    print(type(samples))
    for keys, values in samples.items():
        print(keys)
        print(samples[keys].shape)
if __name__ =="__main__":
    start_time= time.time()
    main()
    end_time = time.time()
    print("eclapsed time :", end_time - start_time)