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

def compute_map(posterior_samples,geo_model_test,normalised_hsi,test_list,y_obs_label, mean_init,cov_init,directory_path,cluster,num_layers,posterior_condition):
    """
    This function computes the maximum a priori based on the posterior samples

    Args:
        posterior_samples : samples generated from mcmc
        geo_model_test : gempy model
        normalised_hsi : normalised hsi data of 204 dimensions as a tensor
        test_list : dictionary of surface points 
        y_obs_label : label data 
        mean_init : initial means from gmm
        cov_init : initial cov from gmm 
        directory_path : path to save file
        num_layers (int, optional): number of layers Defaults to 4.
        posterior_condition (int, optional): posterior condition. Defaults to 2.
    """
    
    
    if(posterior_condition==1):
        print("Posterior 1 reached")
        unnormalise_posterior_value={}
        unnormalise_posterior_value["log_prior_geo_list"]=[]
        unnormalise_posterior_value["log_likelihood_list"]=[]
        unnormalise_posterior_value["log_posterior_list"]=[]
        
        keys_list = list(posterior_samples.keys())
    
        
        prior_mean_surface = [item['normal']['mean'].item() for item in test_list[:num_layers]]
        prior_std_surface =  [item['normal']['std'].item() for item in test_list[:num_layers]]
        store_accuracy=[]
        store_gmm_accuracy = []
        RV_post_mu ={}
        RV_sampledata_post ={} 
        
        for i in range(posterior_samples["mu_1"].shape[0]):
            for j in range(num_layers):  
                RV_post_mu[f"mu_{j+1}"] = posterior_samples[keys_list[j]][i]

            
            # Calculate the log probability of the value
            log_prior_geo = torch.tensor(0.0, dtype=torch.float64)
            for i in range(num_layers):

                log_prior_geo += dist.Normal(prior_mean_surface[i], prior_std_surface[i]).log_prob(RV_post_mu[f"mu_{i+1}"])
            
            # Update the model with the new top layer's location
            interpolation_input = geo_model_test.interpolation_input
            counter1=1
            for interpolation_input_data in test_list[:num_layers]:
                interpolation_input.surface_points.sp_coords = torch.index_put(interpolation_input.surface_points.sp_coords,(torch.tensor([interpolation_input_data["id"]]), torch.tensor([2])), RV_post_mu["mu_"+ str(counter1)])
                counter1=counter1+1
            
            
            
            
            # # Compute the geological model
            geo_model_test.solutions = gempy_engine.compute_model(
                interpolation_input=interpolation_input,
                options=geo_model_test.interpolation_options,
                data_descriptor=geo_model_test.input_data_descriptor,
                geophysics_input=geo_model_test.geophysics_input,
            )
            
            # Compute and observe the thickness of the geological layer
            
            custom_grid_values = geo_model_test.solutions.octrees_output[0].last_output_center.custom_grid_values
            accuracy_intermediate = torch.sum(torch.round(custom_grid_values) == y_obs_label) / y_obs_label.shape[0]
            
            store_accuracy.append(accuracy_intermediate)
            lambda_ = 5.0
            
            z_nk = F.softmax(-lambda_* (torch.tensor([1,2,3,4,5,6], dtype=torch.float64) - custom_grid_values.reshape(-1,1))**2, dim=1)
            
            
            N_k = torch.sum(z_nk,axis=0)
            N = len(custom_grid_values)
            pi_k = N_k /N
            mean = []
            cov = []
            for i in range(z_nk.shape[1]):
                mean_k = torch.sum( z_nk[:,i][:,None] * normalised_hsi, axis=0)/ N_k[i]
                
                cov_k = torch.zeros((mean_k.shape[0],mean_k.shape[0]), dtype=torch.float64)
                for j in range(z_nk.shape[0]):
                    cov_k +=  z_nk[j,i]* torch.matmul((normalised_hsi[j,:] - mean_k).reshape((-1,1)) ,(normalised_hsi[j,:] - mean_k).reshape((1,-1)))
                mean.append(mean_k)
                cov_k=cov_k/N_k[i] #+ 1e-3 * torch.diag(torch.ones(cov_k.shape[0],dtype=torch.float64))
                cov.append(cov_k)
            mean_tensor = torch.stack(mean, dim=0)
            cov_tensor = torch.stack(cov,dim=0)
            
            # We can also calculate the accuracy using the mean and covariance to see if our GMM model has imroved or not
            gamma_nk = torch.zeros(z_nk.shape)

            log_likelihood=torch.tensor(0.0, dtype=torch.float64)

            for j in range(normalised_hsi.shape[0]):
                 
                likelihood = 0.0  
    
                for k in range(len(pi_k)):
                    likelihood += pi_k[k] * torch.exp(dist.MultivariateNormal(loc=mean_tensor[k], covariance_matrix=cov_tensor[k]).log_prob(normalised_hsi[j]))            
                for k in range(gamma_nk.shape[1]):
                    gamma_nk[j][k] = (pi_k[k] * torch.exp(dist.MultivariateNormal(loc=mean_tensor[k],covariance_matrix= cov_tensor[k]).log_prob(normalised_hsi[j]))) / likelihood
                    
                log_likelihood += torch.log(likelihood)
            
            gmm_label_new = torch.argmax(gamma_nk,dim=1) +1
            gmm_accuracy = torch.sum(gmm_label_new == y_obs_label) / y_obs_label.shape[0]
            store_gmm_accuracy.append(gmm_accuracy)
            
            unnormalise_posterior_value["log_prior_geo_list"].append(log_prior_geo)
            unnormalise_posterior_value["log_likelihood_list"].append(log_likelihood)
            unnormalise_posterior_value["log_posterior_list"].append(log_prior_geo + log_likelihood)
        
        MAP_sample_index=torch.argmax(torch.tensor(unnormalise_posterior_value["log_posterior_list"]))
        plt.figure(figsize=(10,8))
        plt.plot(torch.arange(len(store_accuracy))+1, torch.tensor(store_accuracy))
        plt.savefig("./Results_without_prior_gmm/accuracyp1.png")
        
        plt.figure(figsize=(10,8))
        plt.plot(torch.arange(len(store_accuracy))+1, torch.tensor(store_gmm_accuracy))
        plt.savefig("./Results_without_prior_gmm/accuracy_gmmp1.png")

    if(posterior_condition==2):
        print("Posterior 2 reached")
        unnormalise_posterior_value={}
        unnormalise_posterior_value["log_prior_geo_list"]=[]
        unnormalise_posterior_value["log_prior_hsi_list"]=[]
        unnormalise_posterior_value["log_likelihood_list"]=[]
        unnormalise_posterior_value["log_posterior_list"]=[]
        
        keys_list = list(posterior_samples.keys())
        alpha=1
        
        prior_mean_surface = [item['normal']['mean'].item() for item in test_list]
        prior_std_surface =  [item['normal']['std'].item() for item in test_list[:num_layers]]
        store_accuracy=[]
        
        num_layers=4
        RV_post_mu ={}
        RV_sampledata_post ={} 
        # Create a dictionary which can store all the random variable of our model

        
        for i in range(posterior_samples["mu_1"].shape[0]):
            for j in range(num_layers):  
                RV_post_mu[f"mu_{j+1}"] = posterior_samples[keys_list[6 + j]][i]

            
            for j in range(6):  
                RV_sampledata_post[f"post_sample_data{j+1}"] = posterior_samples[keys_list[j]][i]

            
            cov_tensor = torch.eye(RV_sampledata_post["post_sample_data1"].shape[0],dtype=torch.float64)
            loc_mean = torch.tensor(mean_init,dtype=torch.float64)
            
            # Calculate the log probability of the value
            log_prior_geo = torch.tensor(0.0, dtype=torch.float64)
            for i in range(num_layers):

                log_prior_geo += dist.Normal(prior_mean_surface[i], prior_std_surface[i]).log_prob(RV_post_mu[f"mu_{i+1}"])
            
            interpolation_input = geo_model_test.interpolation_input
            
            # Update the model with the new top layer's location
        
            counter1=1
            for interpolation_input_data in test_list[:num_layers]:
                interpolation_input.surface_points.sp_coords = torch.index_put(interpolation_input.surface_points.sp_coords,(torch.tensor([interpolation_input_data["id"]]), torch.tensor([2])), RV_post_mu["mu_"+ str(counter1)])
                counter1=counter1+1
            # # Compute the geological model
            geo_model_test.solutions = gempy_engine.compute_model(
                interpolation_input=interpolation_input,
                options=geo_model_test.interpolation_options,
                data_descriptor=geo_model_test.input_data_descriptor,
                geophysics_input=geo_model_test.geophysics_input,
            )
            
            # Compute and observe the thickness of the geological layer
            
            custom_grid_values = geo_model_test.solutions.octrees_output[0].last_output_center.custom_grid_values
            accuracy_intermediate = torch.sum(torch.round(custom_grid_values) == y_obs_label) / y_obs_label.shape[0]
            store_accuracy.append(accuracy_intermediate)
            lambda_ = 15.0
            loc_mean = torch.tensor(mean_init,dtype=torch.float64)
            
            cov_matrix = alpha * torch.eye(loc_mean[0].shape[0],dtype=torch.float64)
            z_nk = F.softmax(-lambda_* (torch.linspace(1,cluster,cluster, dtype=torch.float64) - custom_grid_values.reshape(-1,1))**2, dim=1)
            
            
            # Initialize log_prior_hsi
            log_prior_hsi = 0

            
            for idx, (key, value) in enumerate(RV_sampledata_post.items()):
            
                loc = loc_mean[idx]  
                log_prior_hsi += dist.MultivariateNormal(loc=loc, covariance_matrix=cov_matrix).log_prob(value)

        
            log_likelihood=torch.tensor(0.0, dtype=torch.float64)

            N_k = torch.sum(z_nk,axis=0)
            N = len(custom_grid_values)
            pi_k = N_k /N
            cov = []
            for k in range(loc_mean.shape[0]):
                cov_k = torch.zeros((loc_mean.shape[1],loc_mean.shape[1]),dtype=torch.float64)
                for j in range(z_nk.shape[0]):
                    cov_k +=  z_nk[j,k]* torch.matmul((normalised_hsi[j,:] - posterior_samples[keys_list[k]][i]).reshape((-1,1)) ,(normalised_hsi[j,:] - posterior_samples[keys_list[k]][i]).reshape((1,-1)))
                cov_k=cov_k/N_k[k] #+ 1e-3 * torch.diag(torch.ones(cov_k.shape[0],dtype=torch.float64))
                cov.append(cov_k)
            cov_tensor = torch.stack(cov,dim=0)

            for j in range(normalised_hsi.shape[0]):
                likelihood = 0  
                for idx, (key, value) in enumerate(RV_sampledata_post.items()):
                    cov_matrix = cov_tensor[idx]
                    likelihood += pi_k[idx] * torch.exp(dist.MultivariateNormal(loc=value, covariance_matrix=cov_matrix).log_prob(normalised_hsi[j]))
                
                log_likelihood += torch.log(likelihood)


            unnormalise_posterior_value["log_prior_geo_list"].append(log_prior_geo)
            unnormalise_posterior_value["log_prior_hsi_list"].append(log_prior_hsi)
            unnormalise_posterior_value["log_likelihood_list"].append(log_likelihood)
            unnormalise_posterior_value["log_posterior_list"].append(log_prior_geo + log_prior_hsi + log_likelihood)
        
        MAP_sample_index=torch.argmax(torch.tensor(unnormalise_posterior_value["log_posterior_list"]))
        plt.figure(figsize=(10,8))
        plt.plot(torch.arange(len(store_accuracy))+1, torch.tensor(store_accuracy))
        filename_sample_accuracy = directory_path + "/accuracyp2.png"
        plt.savefig(filename_sample_accuracy)
    
    if(posterior_condition==3):
        print("Posterior 3 reached")
        unnormalise_posterior_value={}
        unnormalise_posterior_value["log_prior_geo_list"]=[]
        unnormalise_posterior_value["log_prior_hsi_mean_list"]=[]
        unnormalise_posterior_value["log_prior_hsi_cov_list"]=[]
        unnormalise_posterior_value["log_likelihood_list"]=[]
        unnormalise_posterior_value["log_posterior_list"]=[]
        
        keys_list = list(posterior_samples.keys())
        alpha=100
        beta=1000
        
        prior_mean_surface = [item['normal']['mean'].item() for item in test_list]
        prior_std_surface =  [item['normal']['std'].item() for item in test_list[:num_layers]]
        num_data= torch.tensor(mean_init,dtype=torch.float64).shape[0]
        
        

        store_accuracy=[]
        store_gmm_accuracy = []
        RV_post_mu = {}
        RV_post_mean_data = {}
        RV_post_cov_eigen = {}
        eigen_vector_list , eigen_values_list =[],[]
        for i in range(cov_init.shape[0]):
            eigen_values, eigen_vectors = np.linalg.eig(cov_init[i])
            eigen_values_list.append(eigen_values)
            eigen_vector_list.append(eigen_vectors)
        
        for i in range(posterior_samples[keys_list[0]].shape[0]):
    
            
            for j in range(num_layers):
                key = f"mu_{j+1}"
                RV_post_mu[key] = posterior_samples[keys_list[12 + j]][i]  

            
            for j in range(num_data):
                key = f"data{j+1}"
                RV_post_mean_data[key] = posterior_samples[keys_list[6 + j]][i]  

            
            for j in range(num_data):
                key = f"eval{j+1}"
                RV_post_cov_eigen[key] = posterior_samples[keys_list[j]][i]
        
        
        log_prior_geo = torch.tensor(0.0, dtype=torch.float64)
        for i in range(num_layers):
            log_prior_geo+=dist.Normal(prior_mean_surface[i], prior_std_surface[i]).log_prob(RV_post_mu[f"mu_{i+1}"])
        
        
        
        # Update the model with the new top layer's location
        interpolation_input = geo_model_test.interpolation_input
        counter1=1
        for interpolation_input_data in test_list[:num_layers]:
            interpolation_input.surface_points.sp_coords = torch.index_put(interpolation_input.surface_points.sp_coords,(torch.tensor([interpolation_input_data["id"]]), torch.tensor([2])), RV_post_mu["mu_"+ str(counter1)])
            counter1=counter1+1
        
        
        
        
        # # Compute the geological model
        geo_model_test.solutions = gempy_engine.compute_model(
            interpolation_input=interpolation_input,
            options=geo_model_test.interpolation_options,
            data_descriptor=geo_model_test.input_data_descriptor,
            geophysics_input=geo_model_test.geophysics_input,
        )
        
        # Compute and observe the thickness of the geological layer
        
        custom_grid_values = geo_model_test.solutions.octrees_output[0].last_output_center.custom_grid_values
        accuracy_intermediate = torch.sum(torch.round(custom_grid_values) == y_obs_label) / y_obs_label.shape[0]
        
        store_accuracy.append(accuracy_intermediate)
        lambda_ = 15.0
        
        
        pi_k = torch.mean(F.softmax(-lambda_* (torch.linspace(1,cluster,cluster, dtype=torch.float64) - custom_grid_values.reshape(-1,1))**2, dim=1),dim=0)
        
        loc_mean = torch.tensor(mean_init,dtype=torch.float64)
        
        cov_matrix_mean = alpha * torch.eye(loc_mean[0].shape[0], dtype=torch.float64)
        cov_matrix_cov = beta * torch.eye(loc_mean[0].shape[0], dtype=torch.float64)
        
        D = loc_mean.shape[1]
        log_prior_hsi_mean =torch.tensor(0.0, dtype=torch.float64)
        for j in range(loc_mean.shape[0]):
            log_prior_hsi_mean = log_prior_hsi_mean + dist.MultivariateNormal(loc=loc_mean[j],covariance_matrix=cov_matrix_mean).log_prob(posterior_samples[keys_list[6+j]][i])
            
        
        
        # for covariance
        log_prior_hsi_cov =torch.tensor(0.0, dtype=torch.float64)
        cov = []
        for j in range(loc_mean.shape[0]):
            eigen_values_init = torch.tensor(eigen_values_list[j],dtype=torch.float64)
            eigen_vectors_data = torch.tensor(eigen_vector_list[j], dtype=torch.float64)
            log_prior_hsi_cov = log_prior_hsi_cov + dist.MultivariateNormal(loc=torch.sqrt(eigen_values_init),covariance_matrix=cov_matrix_cov).log_prob(posterior_samples[keys_list[j]][i])
            cov_data = eigen_vectors_data @ torch.diag(posterior_samples[keys_list[j]][i])**2 @ eigen_vectors_data.T
            cov.append(cov_data)
        cov_tensor = torch.stack(cov, dim=0)
            
        log_likelihood=torch.tensor(0.0, dtype=torch.float64)

        for j in range(normalised_hsi.shape[0]):
            likelihood = 0.0  
            
            for k in range(len(pi_k)):
                key = f"data{k+1}"
                likelihood += pi_k[k] * torch.exp(
                    dist.MultivariateNormal(loc=RV_post_mean_data[key], covariance_matrix=cov_tensor[k]).log_prob(normalised_hsi[j])
                )

            
            log_likelihood += torch.log(likelihood)
        
        unnormalise_posterior_value["log_prior_geo_list"].append(log_prior_geo)
        unnormalise_posterior_value["log_prior_hsi_mean_list"].append(log_prior_hsi_mean)
        unnormalise_posterior_value["log_prior_hsi_cov_list"].append(log_prior_hsi_cov)
        unnormalise_posterior_value["log_likelihood_list"].append(log_likelihood)
        unnormalise_posterior_value["log_posterior_list"].append(log_prior_geo +log_prior_hsi_mean + log_prior_hsi_cov +log_likelihood)
    
        MAP_sample_index=torch.argmax(torch.tensor(unnormalise_posterior_value["log_posterior_list"]))
        plt.figure(figsize=(10,8))
        plt.plot(torch.arange(len(store_accuracy))+1, torch.tensor(store_accuracy))
        filename_sample_accuracy = directory_path + "/accuracyp3.png"
        plt.savefig(filename_sample_accuracy)
        

    return MAP_sample_index    