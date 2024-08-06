import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import arviz as az
import pandas as pd
import os
import torch
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, Predictive, EmpiricalMarginal
from pyro.infer.autoguide import init_to_mean, init_to_median, init_to_value
from pyro.infer.inspect import get_dependencies
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete

import gempy as gp
import gempy_engine
import gempy_viewer as gpv
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_probability.plot_posterior import default_red, default_blue, PlotPosterior

import scipy.io
from scipy.stats import zscore
from sklearn.manifold import TSNE

from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import KMeans


def model_test(obs_data,interpolation_input_,geo_model_test,mean_init,cov_init,factor,num_layers=4):
        """
        This Pyro model represents the probabilistic aspects of the geological model.
        It defines a prior distribution for the top layer's location and
        computes the thickness of the geological layer as an observed variable.
        obs_data: represents the observed data features reduced from 204 to 10 using PCA
        interpolation_input_: represents the dictionary of random variables for surface parameters
        num_layers: represents the number of layers we want to include in the model
        
        """

        Random_variable ={}



        # Define prior for the top layer's location
        # prior_mean_surface_1 = sp_coords_copy_test[11, 2]
        # prior_mean_surface_2 = sp_coords_copy_test[14, 2]
        # prior_mean_surface_3 = sp_coords_copy_test[5, 2]
        # prior_mean_surface_4 = sp_coords_copy_test[0, 2]

        # mu_surface_1 = pyro.sample('mu_1', dist.Normal(prior_mean_surface_1, torch.tensor(0.2, dtype=torch.float64)))
        # mu_surface_2 = pyro.sample('mu_2', dist.Normal(prior_mean_surface_2, torch.tensor(0.2, dtype=torch.float64)))
        # mu_surface_3 = pyro.sample('mu_3', dist.Normal(prior_mean_surface_3, torch.tensor(0.2, dtype=torch.float64)))
        # mu_surface_4 = pyro.sample('mu_4', dist.Normal(prior_mean_surface_4, torch.tensor(0.2, dtype=torch.float64)))
        
        
        interpolation_input = geo_model_test.interpolation_input
        # Create a random variable based on the provided dictionary used to modify input data of gempy
        counter=1
        for interpolation_input_data in interpolation_input_[:num_layers]:
            
            # Check if user wants to create random variable based on modifying the surface points of gempy
            if interpolation_input_data["update"]=="interface_data":
                # Check what kind of distribution is needed
                if interpolation_input_data["prior_distribution"]=="normal":
                    mean = interpolation_input_data["normal"]["mean"]
                    #print(mean)
                    std  = interpolation_input_data["normal"]["std"]
                    Random_variable["mu_"+ str(counter)] = pyro.sample("mu_"+ str(counter), dist.Normal(mean, std))
                    #print(counter)
                    #counter=counter+1
                    #print(counter)
                elif interpolation_input_data["prior_distribution"]=="uniform":
                    min = interpolation_input_data["uniform"]["min"]
                    max = interpolation_input_data["uniform"]["min"]
                    Random_variable["mu_"+ str(interpolation_input_data['id'])] = pyro.sample("mu_"+ str(interpolation_input_data['id']), dist.Uniform(min, max))
                    #print(counter)
                    #counter=counter+1
                    
                else:
                    print("We have to include the distribution")
            
            
                # Check which co-ordinates direction we wants to allow and modify the surface point data
                if interpolation_input_data["direction"]=="X":
                    interpolation_input.surface_points.sp_coords = torch.index_put(
                        interpolation_input.surface_points.sp_coords,
                        (torch.tensor([interpolation_input_data["id"]]), torch.tensor([0])),
                        Random_variable["mu_"+ str(counter)])
                elif interpolation_input_data["direction"]=="Y":
                    interpolation_input.surface_points.sp_coords = torch.index_put(
                        interpolation_input.surface_points.sp_coords,
                        (torch.tensor([interpolation_input_data["id"]]), torch.tensor([1])),
                        Random_variable["mu_"+ str(counter)])
                elif interpolation_input_data["direction"]=="Z":
                    interpolation_input.surface_points.sp_coords = torch.index_put(
                        interpolation_input.surface_points.sp_coords,
                        (torch.tensor([interpolation_input_data["id"]]), torch.tensor([2])),
                        Random_variable["mu_"+ str(counter)])
                    
                else:
                    print("Wrong direction")
            #print(counter)
            counter=counter+1

        pyro.sample('mu_1 < 0', dist.Delta(torch.tensor(1.0, dtype=torch.float64)), obs=(Random_variable["mu_1"] < 3.7))
        pyro.sample('mu_1 > mu_2', dist.Delta(torch.tensor(1.0, dtype=torch.float64)), obs=(Random_variable["mu_1"] > Random_variable["mu_2"]))
        pyro.sample('mu_2 > mu_3', dist.Delta(torch.tensor(1.0, dtype=torch.float64)), obs=(Random_variable["mu_2"] > Random_variable["mu_3"]))
        pyro.sample('mu_3 < mu_4', dist.Delta(torch.tensor(1.0, dtype=torch.float64)), obs=(Random_variable["mu_3"] > Random_variable["mu_4"]))
        #pyro.sample('mu_3 > -61.5', dist.Delta(torch.tensor(1.0, dtype=torch.float64)), obs=(mu_surface_3 > 0.625))
        #pyro.sample('mu_4 < -61.5', dist.Delta(torch.tensor(1.0, dtype=torch.float64)), obs=(mu_surface_4 < 0.625))
        pyro.sample('mu_4 > -83', dist.Delta(torch.tensor(1.0, dtype=torch.float64)), obs=(Random_variable["mu_4"] > - 0.2 ))
        # Update the model with the new top layer's location
        interpolation_input = geo_model_test.interpolation_input
        
        # interpolation_input.surface_points.sp_coords = torch.index_put(
        #     interpolation_input.surface_points.sp_coords,
        #     (torch.tensor([11]), torch.tensor([2])),
        #     mu_surface_1
        # )
        # interpolation_input.surface_points.sp_coords = torch.index_put(
        #     interpolation_input.surface_points.sp_coords,
        #     (torch.tensor([14]), torch.tensor([2])),
        #     mu_surface_2
        # )
        
        # interpolation_input.surface_points.sp_coords = torch.index_put(
        #     interpolation_input.surface_points.sp_coords,
        #     (torch.tensor([5]), torch.tensor([2])),
        #     mu_surface_3
        # )
        # interpolation_input.surface_points.sp_coords = torch.index_put(
        #     interpolation_input.surface_points.sp_coords,
        #     (torch.tensor([0]), torch.tensor([2])),
        #     mu_surface_4
        # )
        
        
        #print("interpolation_input",interpolation_input.surface_points.sp_coords)
        
        # # Compute the geological model
        geo_model_test.solutions = gempy_engine.compute_model(
            interpolation_input=interpolation_input,
            options=geo_model_test.interpolation_options,
            data_descriptor=geo_model_test.input_data_descriptor,
            geophysics_input=geo_model_test.geophysics_input,
        )
        
        # Compute and observe the thickness of the geological layer
        
        custom_grid_values = geo_model_test.solutions.octrees_output[0].last_output_center.custom_grid_values
        #accuracy_intermediate = torch.sum(torch.round(custom_grid_values) == y_obs_label) / y_obs_label.shape[0]
        #store_accuracy.append(accuracy_intermediate)
        
        lambda_ = 15.0
        loc_mean = torch.tensor(mean_init,dtype=torch.float64)
        loc_cov =  torch.tensor(cov_init, dtype=torch.float64)
        #class_label = F.softmax(-lambda_* (torch.tensor([1,2,3,4,5,6], dtype=torch.float64) - custom_grid_values.reshape(-1,1))**2, dim=1)
        class_label = torch.mean(F.softmax(-lambda_* (torch.tensor([1,2,3,4,5,6], dtype=torch.float64) - custom_grid_values.reshape(-1,1))**2, dim=1),dim=0)
        
        sample = []
        for i in range(loc_mean.shape[0]):
            # Check if loc_mean[i] and loc_cov[i] are scalars or 1D tensors of length 1
            if torch.numel(loc_mean[i]) == 1 and torch.numel(loc_cov[i]) == 1:
                # Use univariate Normal distribution
                mu = loc_mean[i].item() if torch.is_tensor(loc_mean[i]) else loc_mean[i]
                sigma = torch.sqrt(loc_cov[i]).item() if torch.is_tensor(loc_cov[i]) else torch.sqrt(loc_cov[i])
                sample_data = pyro.sample(f"sample_data{i+1}", dist.Normal(loc=mu, scale=sigma))
                sample.append(sample_data)
            else:
                # Use MultivariateNormal distribution
                #print("pointer reached else multivariate case")
                sample_data = pyro.sample(f"sample_data{i+1}", dist.MultivariateNormal(loc=loc_mean[i], covariance_matrix=loc_cov[i]))
                sample.append(sample_data)

        sample_tensor = torch.stack(sample, dim=0)
    
        #cov_likelihood = 5.0 * torch.eye(loc_cov[0].shape[0], dtype=torch.float64)
        
        with pyro.plate('N='+str(obs_data.shape[0]), obs_data.shape[0]):
            assignment = pyro.sample("assignment", dist.Categorical(class_label))
            # Check the dimensionality of the data
            if obs_data.dim() == 1 or (obs_data.dim() == 2 and obs_data.shape[1] == 1):
                # Univariate case
                loc = sample_tensor[assignment].squeeze()
                scale = torch.sqrt(factor * loc_cov[assignment]).squeeze()
                obs = pyro.sample("obs", dist.Normal(loc=loc, scale=scale), obs=obs_data)
            else:
                # Multivariate case
                loc = sample_tensor[assignment]
                covariance_matrix = factor * loc_cov[assignment]
                obs = pyro.sample("obs", dist.MultivariateNormal(loc=loc, covariance_matrix=covariance_matrix), obs=obs_data)
            #obs = pyro.sample("obs", dist.MultivariateNormal(loc=sample_tesnor[assignment],covariance_matrix= factor * loc_cov[assignment]), obs=obs_data)
            #obs = pyro.sample("obs", dist.MultivariateNormal(loc=sample_tesnor[assignment],covariance_matrix=cov_likelihood), obs=obs_data)
        #return geo_model_test.interpolation_input
