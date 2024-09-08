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


def model_test(obs_data,interpolation_input_,geo_model_test,mean_init,cov_init,factor,num_layers,posterior_condition):
        """
        This Pyro model represents the probabilistic aspects of the geological model.
        It defines a prior distribution for the top layer's location and
        computes the thickness of the geological layer as an observed variable.

        obs_data: represents the observed data features reduced from 204 to 10 using PCA
        interpolation_input_: represents the dictionary of random variables for surface parameters
        geo_model_test : gempy model
        mean_init : initial means from gmm
        cov_init : initial cov from gmm
        num_layers: represents the number of layers we want to include in the model
        posterior_condition (int, optional): posterior condition. Defaults to 2.
        """
 

        Random_variable ={}

        alpha = 1
        cluster=6

        
        interpolation_input = geo_model_test.interpolation_input
        # Create a random variable based on the provided dictionary used to modify input data of gempy
        counter=1
        for interpolation_input_data in interpolation_input_[:num_layers]:
            
            # Check if user wants to create random variable based on modifying the surface points of gempy
            if interpolation_input_data["update"]=="interface_data":
                # Check what kind of distribution is needed
                if interpolation_input_data["prior_distribution"]=="normal":
                    mean = interpolation_input_data["normal"]["mean"]
                    std  = interpolation_input_data["normal"]["std"]
                    Random_variable["mu_"+ str(counter)] = pyro.sample("mu_"+ str(counter), dist.Normal(mean, std))
                    
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
            
            counter=counter+1
        
        #Ensuring layer order

        pyro.sample('mu_1 < 0', dist.Delta(torch.tensor(1.0, dtype=torch.float64)), obs=(Random_variable["mu_1"] < 3.7))
        pyro.sample('mu_1 > mu_2', dist.Delta(torch.tensor(1.0, dtype=torch.float64)), obs=(Random_variable["mu_1"] > Random_variable["mu_2"]))
        pyro.sample('mu_2 > mu_3', dist.Delta(torch.tensor(1.0, dtype=torch.float64)), obs=(Random_variable["mu_2"] > Random_variable["mu_3"]))
        pyro.sample('mu_3 > mu_4', dist.Delta(torch.tensor(1.0, dtype=torch.float64)), obs=(Random_variable["mu_3"] > Random_variable["mu_4"]))
        pyro.sample('mu_4 > -83', dist.Delta(torch.tensor(1.0, dtype=torch.float64)), obs=(Random_variable["mu_4"] > - 0.2 ))
        
        # Update the model with the new top layer's location
        interpolation_input = geo_model_test.interpolation_input
        
        
        # # Compute the geological model
        geo_model_test.solutions = gempy_engine.compute_model(
            interpolation_input=interpolation_input,
            options=geo_model_test.interpolation_options,
            data_descriptor=geo_model_test.input_data_descriptor,
            geophysics_input=geo_model_test.geophysics_input,
        )
        
        # Compute and observe the thickness of the geological layer
        custom_grid_values = geo_model_test.solutions.octrees_output[0].last_output_center.custom_grid_values
            
        if posterior_condition==1:
            ################# Condition 1:  Deterministic for mean and covariance for hsi data ###########################
            
            lambda_ = 5.0
            z_nk = F.softmax(-lambda_* (torch.linspace(1,cluster,cluster, dtype=torch.float64) - custom_grid_values.reshape(-1,1))**2, dim=1)
            
            N_k = torch.sum(z_nk,axis=0)
            N = len(custom_grid_values)
            pi_k = N_k /N
            mean = []
            cov = []
            for i in range(z_nk.shape[1]):
                mean_k = torch.sum( z_nk[:,i][:,None] * obs_data, axis=0)/ N_k[i]
                cov_k = torch.zeros((mean_k.shape[0],mean_k.shape[0]),dtype=torch.float64)
                for j in range(z_nk.shape[0]):
                    cov_k +=  z_nk[j,i]* torch.matmul((obs_data[j,:] - mean_k).reshape((-1,1)) ,(obs_data[j,:] - mean_k).reshape((1,-1)))
                mean.append(mean_k)
                cov_k=cov_k/N_k[i] #+ 1e-3 * torch.diag(torch.ones(cov_k.shape[0],dtype=torch.float64))
                cov.append(cov_k)
            
        
        if posterior_condition==2:
            ################# Condition 2: Deterministic for covariance but a prior on mean ###########################
            lambda_ = 15.0
            loc_mean = torch.tensor(mean_init,dtype=torch.float64)
            loc_cov =  torch.tensor(cov_init, dtype=torch.float64)
            cov_matrix = alpha * torch.eye(loc_mean[0].shape[0],dtype=torch.float64)
            z_nk = F.softmax(-lambda_* (torch.linspace(1,cluster,cluster, dtype=torch.float64) - custom_grid_values.reshape(-1,1))**2, dim=1)
            N_k = torch.sum(z_nk,axis=0)
            N = len(custom_grid_values)
            pi_k = N_k /N
            mean =[]
            cov =[]
            for i in range(loc_mean.shape[0]):
                mean_data = pyro.sample("mean_data"+str(i+1), dist.MultivariateNormal(loc=loc_mean[i],covariance_matrix=cov_matrix))
                mean.append(mean_data)
                
            for i in range(loc_mean.shape[0]):
                cov_k = torch.zeros((loc_mean.shape[1],loc_mean.shape[1]),dtype=torch.float64)
                for j in range(z_nk.shape[0]):
                    cov_k +=  z_nk[j,i]* torch.matmul((obs_data[j,:] - mean[i]).reshape((-1,1)) ,(obs_data[j,:] - mean[i]).reshape((1,-1)))
                cov_k=cov_k/N_k[i] #+ 1e-3 * torch.diag(torch.ones(cov_k.shape[0],dtype=torch.float64))
                cov.append(cov_k)
            
            
        if posterior_condition==3:
            ################# Condition 3:Prior on mean and covariance ###########################
            
            beta  = 1000
            lambda_ = 15.0
            loc_mean = torch.tensor(mean_init,dtype=torch.float64)
            loc_cov =  torch.tensor(cov_init, dtype=torch.float64)
            cov_matrix_mean = alpha * torch.eye(loc_mean[0].shape[0], dtype=torch.float64)
            cov_matrix_cov = beta * torch.eye(loc_mean[0].shape[0], dtype=torch.float64)
            pi_k = torch.mean(F.softmax(-lambda_* (torch.linspace(1,cluster,cluster, dtype=torch.float64) - custom_grid_values.reshape(-1,1))**2, dim=1),dim=0)
            D = loc_mean.shape[1]
            eigen_vector_list , eigen_values_list =[],[]
            for i in range(cov_init.shape[0]):
                eigen_values, eigen_vectors = np.linalg.eig(cov_init[i])
                eigen_values_list.append(eigen_values)
                eigen_vector_list.append(eigen_vectors)
            mean =[]
            cov = []
            for i in range(loc_mean.shape[0]):
                mean_data= pyro.sample("mean_data_"+str(i+1), dist.MultivariateNormal(loc=loc_mean[i],covariance_matrix=cov_matrix_mean))
                mean.append(mean_data)
                eigen_values_init = torch.tensor(eigen_values_list[i],dtype=torch.float64)
                eigen_vectors_data = torch.tensor(eigen_vector_list[i], dtype=torch.float64)
                cov_eigen_values = pyro.sample("cov_eigen_values_"+str(i+1), dist.MultivariateNormal(loc=torch.sqrt(eigen_values_init),covariance_matrix=cov_matrix_cov))
                cov_data = eigen_vectors_data @ torch.diag(cov_eigen_values)**2 @ eigen_vectors_data.T #+ 1e-6 * torch.eye(loc_mean[0].shape[0], dtype=torch.float64)
                cov.append(cov_data)
            
        mean_tensor = torch.stack(mean, dim=0)
        cov_tensor = torch.stack(cov, dim=0)
        
        
        with pyro.plate('N='+str(obs_data.shape[0]), obs_data.shape[0]):
            assignment = pyro.sample("assignment", dist.Categorical(pi_k))
            obs = pyro.sample("obs", dist.MultivariateNormal(loc=mean_tensor[assignment],covariance_matrix = cov_tensor[assignment]), obs=obs_data)
