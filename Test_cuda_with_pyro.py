import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import arviz as az
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def main():
    # Check if GPU is available
    
    
    # Generate synthetic data
    def generate_data(N=100):
        X = torch.randn(N, 1).to(device)  # Random input data
        true_weights = torch.tensor([3.0]).to(device)  # True slope
        true_bias = torch.tensor(1.0).to(device)  # True intercept
        noise = 0.1 * torch.randn(N).to(device)  # Gaussian noise
        y = X @ true_weights + true_bias + noise  # Linear relation with noise
        return X, y
    X, y = generate_data()
    
    # Define a simple Bayesian Linear Regression model
    def model(X, y):

        # Priors for the parameters
        weight = pyro.sample("weight", dist.Normal(0., 10.))
        bias = pyro.sample("bias", dist.Normal(0., 10.))
        #print(X.shape)
        # Linear model
        y_hat = X * weight + bias

        # Likelihood
        sigma = pyro.sample("sigma", dist.Uniform(0., 1.))

        #print(sigma)
        with pyro.plate("data", len(X)):
            pyro.sample("obs", dist.Normal(y_hat, sigma), obs=y)
        
    # Inference with NUTS on GPU
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples=200, warmup_steps=100, num_chains=4)
    mcmc.run(X,y,)

    # Summarize the results
    mcmc.summary()

    # Extract the samples
    samples = mcmc.get_samples(group_by_chain=True)
    #print(samples)
    
if __name__ == "__main__":
    
    # Your main script code starts here
    print("Script started...")
    
    # Record the start time
    start_time = datetime.now()
    # 

    
    # Run the main function
    #main()
    if device=="cuda":
        main()
    # Record the end time
    end_time = datetime.now()

    # Your main script code ends here
    print("Script ended...")
    
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    
    print(f"Elapsed time: {elapsed_time}")