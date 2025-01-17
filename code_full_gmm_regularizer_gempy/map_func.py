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
import json

def calculate_average_entropy(responsibilities):
    """
    Calculate the average Shannon entropy of responsibilities for a GMM.
    
    Args:
        responsibilities (numpy array): An (N, K) array where each element gamma_nk is the responsibility of the
                                         k-th component for the n-th data point.
    
    Returns:
        float: The average Shannon entropy across all data points.
    """
    # Clip responsibilities to avoid log(0)
    responsibilities = np.clip(responsibilities, 1e-10, 1.0)
    
    # Calculate the entropy for each data point
    entropy_per_point = -np.sum(responsibilities * np.log(responsibilities), axis=1)
    
    # Return the average entropy
    return np.mean(entropy_per_point)

def calculate_entropy(mixing_coefficient):
    """
    Calculate the average Shannon entropy of responsibilities for a GMM.
    
    Args:
        responsibilities (numpy array): An (N, K) array where each element gamma_nk is the responsibility of the
                                         k-th component for the n-th data point.
    
    Returns:
        float: The average Shannon entropy across all data points.
    """
    # Clip responsibilities to avoid log(0)
    mixing_coefficient = np.clip(mixing_coefficient, 1e-10, 1.0)
    
    # Calculate the entropy for each data point
    entropy_per_point = -np.sum(mixing_coefficient * np.log(mixing_coefficient))
    
    # Return the average entropy
    return entropy_per_point


def compute_map(posterior_samples,geo_model_test,normalised_hsi,test_list,y_obs_label, mean_init,cov_init, factor, directory_path,num_layers,posterior_condition,scale, cluster, alpha, beta, dtype, device):
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
        scale (float):  scaling factor to generate probability for each voxel
        cluster (int): number of cluster in our dataset
        alpha (float): Parameter to control the covariance matrix of drawing a sample for mean
        beta (float): Parameter to control the covariance matrix of drawing a sample for covariance
    """
    

    directory_path_MAP = directory_path +"/MAP"
    
    # Check if the directory exists
    if not os.path.exists(directory_path_MAP):
        # Create the directory if it does not exist
        os.makedirs(directory_path_MAP)
        print(f"Directory '{directory_path_MAP}' was created.")
    else:
        print(f"Directory '{directory_path_MAP}' already exists.")
    
    
    unnormalise_posterior_value={}
    store_accuracy=[]
    store_gmm_accuracy = []
    store_z_nk_entropy =[]
    store_gmm_entropy=[]
    store_mixing_entropy=[]
    
    # Convert the tensors to lists
    posterior_samples_serializable = {k: v.tolist() for k, v in posterior_samples.items()}
    
    loc_mean = torch.tensor(mean_init,dtype=dtype, device=device)
    loc_cov =  torch.tensor(cov_init, dtype=dtype, device=device)
    
    if(posterior_condition==1):
        print("Posterior 1 reached")
        
        unnormalise_posterior_value["log_prior_geo_list"]=[]
        unnormalise_posterior_value["log_likelihood_list"]=[]
        unnormalise_posterior_value["log_posterior_list"]=[]
        
        keys_list = list(posterior_samples.keys())
    
        ########## TODO###############################################################
        # Extend this to other distribution too
        ###############################################################################
        prior_mean_surface = [item['normal']['mean'].item() for item in test_list[:num_layers]]
        prior_std_surface =  [item['normal']['std'].item() for item in test_list[:num_layers]]
        ###############################################################################

        RV_post_mu ={}
        RV_post_pi ={}
        
        # Get index of the samples in posterior
        for i in range(posterior_samples["mu_1"].shape[0]):
            # Get the georemtrical random variable for a given sample 
            
            for j in range(num_layers):  
                RV_post_mu[f"mu_{j+1}"] = posterior_samples[keys_list[j]][i]
                
            RV_post_pi["pi"] = posterior_samples[keys_list[num_layers]][i]
            
            # Calculate the log probability of the value
            log_prior_geo = torch.tensor(0.0, dtype=dtype, device =device)
            for l in range(num_layers):
                log_prior_geo += dist.Normal(prior_mean_surface[l], prior_std_surface[l]).log_prob(RV_post_mu[f"mu_{l+1}"])
            ##########################################################################
            # Update the model with the new top layer's location
            ##########################################################################
            interpolation_input = geo_model_test.interpolation_input
            
            counter1=1
            for interpolation_input_data in test_list[:num_layers]:
                interpolation_input.surface_points.sp_coords = torch.index_put(interpolation_input.surface_points.sp_coords,(interpolation_input_data["id"], torch.tensor([2])), RV_post_mu["mu_"+ str(counter1)])
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
            
            
            z_nk = F.softmax(-scale* (torch.linspace(1,cluster,cluster, dtype=dtype, device =device) - custom_grid_values.reshape(-1,1))**2, dim=1)
            
            
            N_k = torch.sum(z_nk,axis=0)
            N = len(custom_grid_values)
            pi_k = N_k /N
            mean = []
            cov = []
            ##########################################################################
            # Calculate the mean and covariance 
            ##########################################################################
           
            
            for k in range(loc_mean.shape[0]):
                mean.append(loc_mean[k])
                cov.append(loc_cov[k])
                
            mean_tensor = torch.stack(mean, dim=0)
            cov_tensor = torch.stack(cov,dim=0)
            
            # We can also calculate the accuracy using the mean and covariance to see if our GMM model has imroved or not
            #gamma_nk = torch.zeros(z_nk.shape)

            # log_likelihood=torch.tensor(0.0, dtype=dtype, device =device)

            # gmm_log_probs = dist.MultivariateNormal(mean_tensor, cov_tensor).log_prob(obs_data.unsqueeze(1))  # (N, K)
                
            #     # Combine log responsibilities with priors
            # log_likelihood = torch.sum(z_nk * (log_pi + gmm_log_probs),axis=1)  # (N,)
                
            #     # Factor for log likelihood
            # pyro.factor("log_likelihood", log_likelihood.sum())  # Scalar log joint
            
            # for j in range(normalised_hsi.shape[0]):
                 
            #     likelihood = 0.0  
    
            #     for k in range(len(pi_k)):
            #         likelihood += z_nk[j,k] * torch.exp(dist.MultivariateNormal(loc=mean_tensor[k], covariance_matrix=factor *cov_tensor[k]).log_prob(normalised_hsi[j]))            
            #     for k in range(len(pi_k)):
            #         gamma_nk[j][k] = (z_nk[j,k] * torch.exp(dist.MultivariateNormal(loc=mean_tensor[k],covariance_matrix= factor * cov_tensor[k]).log_prob(normalised_hsi[j]))) / likelihood
                    
            #     log_likelihood += torch.log(likelihood)
            
            gamma_nk = z_nk
            log_likelihood=torch.tensor(0.0, dtype=dtype, device =device)

            log_prior_hsi_pi = dist.Dirichlet(torch.ones(int(len(pi_k)), dtype=dtype, device=device)).log_prob(RV_post_pi["pi"])
            
            log_pi = torch.log(RV_post_pi["pi"]).unsqueeze(0)
            gmm_log_probs = dist.MultivariateNormal(loc=mean_tensor, covariance_matrix=cov_tensor).log_prob(normalised_hsi.unsqueeze(1))
            log_gmm_broadcast = log_pi + gmm_log_probs
            log_likelihood = torch.sum(z_nk * log_gmm_broadcast, axis=1).sum()
            
            
            gmm_label_new = torch.argmax(gamma_nk,dim=1) +1
            gmm_accuracy = torch.sum(gmm_label_new == y_obs_label) / y_obs_label.shape[0]
            store_gmm_accuracy.append(gmm_accuracy)
            
            entropy_gmm = calculate_entropy(gamma_nk.detach().numpy())
            entropy_z_nk = calculate_average_entropy(z_nk.detach().numpy())
            entropy_pi_k = calculate_entropy(pi_k.detach().numpy())
            store_z_nk_entropy.append(entropy_z_nk)
            store_gmm_entropy.append(entropy_gmm)
            store_mixing_entropy.append(entropy_pi_k)
            
            unnormalise_posterior_value["log_prior_geo_list"].append(log_prior_geo)
            unnormalise_posterior_value["log_likelihood_list"].append(log_likelihood)
            unnormalise_posterior_value["log_posterior_list"].append(log_prior_geo + log_likelihood)
        
        plt.figure(figsize=(8,10))
        plt.plot(np.array([ele.detach().detach() for ele in unnormalise_posterior_value["log_prior_geo_list"]]), label='prior_geo', marker=".")
        plt.plot(np.array([ele.detach().detach() for ele in unnormalise_posterior_value["log_likelihood_list"]]), label='prior_likelihood', marker="d")
        plt.plot(np.array([ele.detach().detach() for ele in unnormalise_posterior_value["log_posterior_list"]]), label='posterior', marker="+")
        plt.xlabel('Iteration')
        plt.ylabel('Unnormalised log value')
        plt.title('Log value of each term in posterior')
        plt.legend()
        filename_log = directory_path_MAP + "/log_posterior.png"
        plt.savefig(filename_log)
        plt.close()
    
        
    elif posterior_condition==2:
        
        unnormalise_posterior_value={}
        unnormalise_posterior_value["log_prior_geo_list"]=[]
        unnormalise_posterior_value["log_prior_hsi_mean_list"]=[]
        unnormalise_posterior_value["log_likelihood_list"]=[]
        unnormalise_posterior_value["log_posterior_list"]=[]
        
        keys_list = list(posterior_samples.keys())
       
        
        prior_mean_surface = [item['normal']['mean'].item() for item in test_list]
        prior_std_surface =  [item['normal']['std'].item() for item in test_list[:num_layers]]
        
        
        
        RV_post_mu ={}
        RV_post_mean ={} 
        RV_post_pi ={}
        # Create a dictionary which can store all the random variable of our model

        # Get index of the samples in posterior
        for i in range(posterior_samples["mu_1"].shape[0]):
            # Get the georemtrical random variable for a given sample 
            for j in range(num_layers):  
                RV_post_mu[f"mu_{j+1}"] = posterior_samples[keys_list[cluster + j]][i]
                
            for j in range(cluster):  
                RV_post_mean[f"mean_data{j+1}"] = posterior_samples[keys_list[j]][i]

            RV_post_pi["pi"] = posterior_samples[keys_list[cluster + num_layers]][i]
            
            cov_tensor = torch.eye(RV_post_mean["mean_data1"].shape[0],dtype=dtype, device =device)
            #loc_mean = torch.tensor(mean_init,dtype=dtype, device =device)
            
            # Calculate the log probability of the value
            log_prior_geo = torch.tensor(0.0, dtype=dtype, device =device)
            for j in range(num_layers):

                log_prior_geo += dist.Normal(prior_mean_surface[j], prior_std_surface[j]).log_prob(RV_post_mu[f"mu_{j+1}"])
            
            interpolation_input = geo_model_test.interpolation_input
            
            # Update the model with the new top layer's location
        
            counter1=1
            for interpolation_input_data in test_list[:num_layers]:
                interpolation_input.surface_points.sp_coords = torch.index_put(interpolation_input.surface_points.sp_coords,(interpolation_input_data["id"], torch.tensor([2])), RV_post_mu["mu_"+ str(counter1)])
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
            
            loc_mean = torch.tensor(mean_init,dtype=dtype, device =device)
            
            cov_matrix = alpha * torch.eye(loc_mean[0].shape[0],dtype=dtype, device =device)
            z_nk = F.softmax(-scale* (torch.linspace(1,cluster,cluster, dtype=dtype, device =device) - custom_grid_values.reshape(-1,1))**2, dim=1)
            
            
            # Initialize log_prior_hsi
            log_prior_hsi = 0
            
            for idx, (key, value) in enumerate(RV_post_mean.items()):
            
                loc = loc_mean[idx]  
                log_prior_hsi += dist.MultivariateNormal(loc=loc, covariance_matrix=cov_matrix).log_prob(value)

            # We can also calculate the accuracy using the mean and covariance to see if our GMM model has imroved or not
            #gamma_nk = torch.zeros(z_nk.shape)
            
            l#og_likelihood=torch.tensor(0.0, dtype=dtype, device =device)

            N_k = torch.sum(z_nk,axis=0)
            N = len(custom_grid_values)
            pi_k = N_k /N
            cov = []
            for k in range(loc_mean.shape[0]):
                cov.append(loc_cov[k])
            cov_tensor = torch.stack(cov,dim=0)
            
            gamma_nk = z_nk
            log_likelihood=torch.tensor(0.0, dtype=dtype, device =device)

            log_prior_hsi_pi = dist.Dirichlet(torch.ones(int(len(pi_k)), dtype=dtype, device=device)).log_prob(RV_post_pi["pi"])
            
            log_pi = torch.log(RV_post_pi["pi"]).unsqueeze(0)
            gmm_log_probs = dist.MultivariateNormal(loc=mean_tensor, covariance_matrix=cov_tensor).log_prob(normalised_hsi.unsqueeze(1))
            log_gmm_broadcast = log_pi + gmm_log_probs
            log_likelihood = torch.sum(z_nk * log_gmm_broadcast, axis=1).sum()

            # for j in range(normalised_hsi.shape[0]):
            #     likelihood = 0.0  
            #     for idx, (key, value) in enumerate(RV_post_mean.items()):
            #         cov_matrix = factor * cov_tensor[idx]
            #         likelihood += z_nk[j,idx] * torch.exp(dist.MultivariateNormal(loc=value, covariance_matrix=cov_matrix).log_prob(normalised_hsi[j]))
            #     for idx, (key, value) in enumerate(RV_post_mean.items()):
            #         cov_matrix = factor * cov_tensor[idx]
            #         gamma_nk[j][idx] = (z_nk[j,idx] * torch.exp(dist.MultivariateNormal(loc=value, covariance_matrix=cov_matrix).log_prob(normalised_hsi[j]))) / likelihood
                
            #     log_likelihood += torch.log(likelihood)

            gmm_label_new = torch.argmax(gamma_nk,dim=1) +1
            gmm_accuracy = torch.sum(gmm_label_new == y_obs_label) / y_obs_label.shape[0]
            store_gmm_accuracy.append(gmm_accuracy)
            
            entropy_gmm = calculate_entropy(gamma_nk.detach().numpy())
            entropy_z_nk = calculate_average_entropy(z_nk.detach().numpy())
            entropy_pi_k = calculate_entropy(pi_k.detach().numpy())
            store_z_nk_entropy.append(entropy_z_nk)
            store_gmm_entropy.append(entropy_gmm)
            store_mixing_entropy.append(entropy_pi_k)
            
            unnormalise_posterior_value["log_prior_geo_list"].append(log_prior_geo)
            unnormalise_posterior_value["log_prior_hsi_mean_list"].append(log_prior_hsi)
            unnormalise_posterior_value["log_likelihood_list"].append(log_likelihood)
            unnormalise_posterior_value["log_posterior_list"].append(log_prior_geo + log_prior_hsi + log_likelihood)
        
        plt.figure(figsize=(8,10))
        plt.plot(np.array([ele.detach().detach() for ele in unnormalise_posterior_value["log_prior_geo_list"]]), label='prior_geo', marker=".")
        plt.plot(np.array([ele.detach().detach() for ele in unnormalise_posterior_value["log_prior_hsi_mean_list"]]), label='prior_hsi_mean', marker="*")
        plt.plot(np.array([ele.detach().detach() for ele in unnormalise_posterior_value["log_likelihood_list"]]), label='prior_likelihood', marker="d")
        plt.plot(np.array([ele.detach().detach() for ele in unnormalise_posterior_value["log_posterior_list"]]), label='posterior', marker="+")
        plt.xlabel('Iteration')
        plt.ylabel('Unnormalised log value')
        plt.title('Log value of each term in posterior')
        plt.legend()
        filename_log = directory_path_MAP + "/log_posterior.png"
        plt.savefig(filename_log)
        plt.close()
    
    elif posterior_condition==3:
        
        print("Posterior 3 reached")
        
        unnormalise_posterior_value["log_prior_geo_list"]=[]
        unnormalise_posterior_value["log_prior_hsi_pi_list"]=[]
        unnormalise_posterior_value["log_prior_hsi_mean_list"]=[]
        unnormalise_posterior_value["log_prior_hsi_cov_list"]=[]
        unnormalise_posterior_value["log_likelihood_list"]=[]
        unnormalise_posterior_value["log_posterior_list"]=[]
        
        keys_list = list(posterior_samples.keys())

        
        prior_mean_surface = [item['normal']['mean'].item() for item in test_list]
        prior_std_surface =  [item['normal']['std'].item() for item in test_list[:num_layers]]
        
        K = torch.tensor(mean_init,dtype=dtype, device =device).shape[0]
        
        RV_post_mu = {}
        RV_post_mean_data = {}
        RV_post_pi ={}
        RV_post_cov_eigen = {}
        
        eigen_vector_list , eigen_values_list =[],[]
        for i in range(cov_init.shape[0]):
            eigen_values, eigen_vectors = np.linalg.eig(cov_init[i])
            eigen_values_list.append(eigen_values)
            eigen_vector_list.append(eigen_vectors)
        
        posterior_samples_serializable["eigen_vectors"]= [ele.tolist() for ele in eigen_vector_list]
        
        
        for i in range(posterior_samples[keys_list[0]].shape[0]):
    
            
            for j in range(num_layers):
                key = f"mu_{j+1}"
                RV_post_mu[key] = posterior_samples[keys_list[2 * cluster + j]][i]  

            
            for j in range(cluster):
                key = f"data{j+1}"
                RV_post_mean_data[key] = posterior_samples[keys_list[cluster + j]][i]  
                
            for j in range(cluster):
                key = f"eval{j+1}"
                RV_post_cov_eigen[key] = posterior_samples[keys_list[j]][i]
        
            RV_post_pi["pi"] = posterior_samples[keys_list[2 * cluster + num_layers]][i]
            
            #print(RV_post_pi) 
            
            log_prior_geo = torch.tensor(0.0, dtype=dtype, device =device)
            for l in range(num_layers):
                log_prior_geo+=dist.Normal(prior_mean_surface[l], prior_std_surface[l]).log_prob(RV_post_mu[f"mu_{l+1}"])
            
            
            # Update the model with the new top layer's location
            interpolation_input = geo_model_test.interpolation_input
            counter1=1
            for interpolation_input_data in test_list[:num_layers]:
                interpolation_input.surface_points.sp_coords = torch.index_put(interpolation_input.surface_points.sp_coords,(interpolation_input_data["id"], torch.tensor([2])), RV_post_mu["mu_"+ str(counter1)])
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
            
            
            z_nk = F.softmax(-scale* (torch.linspace(1,cluster,cluster, dtype=dtype, device =device) - custom_grid_values.reshape(-1,1))**2, dim=1)
            N_k = torch.sum(z_nk,axis=0)
            N = len(custom_grid_values)
            pi_k = N_k /N
            
            loc_mean = torch.tensor(mean_init,dtype=dtype, device =device)
            
            cov_matrix_mean = alpha * torch.eye(loc_mean[0].shape[0], dtype=dtype, device =device)
            cov_matrix_cov = beta * torch.eye(loc_mean[0].shape[0], dtype=dtype, device =device)
            
            D = loc_mean.shape[1]
            mean = []
            log_prior_hsi_mean =torch.tensor(0.0, dtype=dtype, device =device)
            
            for j in range(loc_mean.shape[0]):
                mean.append(posterior_samples[keys_list[cluster+j]][i])
                log_prior_hsi_mean = log_prior_hsi_mean + dist.MultivariateNormal(loc=loc_mean[j],covariance_matrix=cov_matrix_mean).log_prob(posterior_samples[keys_list[cluster+j]][i])
                
            mean_tensor = torch.stack(mean, dim=0)
            
            # for covariance
            log_prior_hsi_cov =torch.tensor(0.0, dtype=dtype, device =device)
            cov = []
            for j in range(loc_mean.shape[0]):
                eigen_values_init = torch.tensor(eigen_values_list[j],dtype=dtype, device =device)
                eigen_vectors_data = torch.tensor(eigen_vector_list[j], dtype=dtype, device =device)
                log_prior_hsi_cov = log_prior_hsi_cov + dist.MultivariateNormal(loc=torch.sqrt(eigen_values_init),covariance_matrix=cov_matrix_cov).log_prob(posterior_samples[keys_list[j]][i])
                #log_prior_hsi_cov = log_prior_hsi_cov + dist.MultivariateNormal(loc=torch.zeros(eigen_values_init.shape[0],dtype=dtype, device =device),covariance_matrix=cov_matrix_cov).log_prob(posterior_samples[keys_list[j]][i])
                cov_data = eigen_vectors_data @ (torch.diag(posterior_samples[keys_list[j]][i])**2 + 1e-8 * torch.eye(eigen_values_init.shape[0], dtype=dtype, device =device)) @ eigen_vectors_data.T
                cov.append(cov_data)
            cov_tensor = torch.stack(cov, dim=0)
            
            # We can also calculate the accuracy using the mean and covariance to see if our GMM model has imroved or not
            #gamma_nk = torch.zeros(z_nk.shape)  
            gamma_nk = z_nk
            log_likelihood=torch.tensor(0.0, dtype=dtype, device =device)

            log_prior_hsi_pi = dist.Dirichlet(torch.ones(int(len(pi_k)), dtype=dtype, device=device)).log_prob(RV_post_pi["pi"])
            
            log_pi = torch.log(RV_post_pi["pi"]).unsqueeze(0)
            gmm_log_probs = dist.MultivariateNormal(loc=mean_tensor, covariance_matrix=cov_tensor).log_prob(normalised_hsi.unsqueeze(1))
            log_gmm_broadcast = log_pi + gmm_log_probs
            log_likelihood = torch.sum(z_nk * log_gmm_broadcast, axis=1).sum()
            
            #print(log_likelihood)
            
            
            # for j in range(normalised_hsi.shape[0]):
            #     likelihood = 0.0  
                
            #     for k in range(len(pi_k)):
            #         key = f"data{k+1}"
            #         likelihood += z_nk[j,k] * (torch.exp(
            #             dist.MultivariateNormal(loc=RV_post_mean_data[key], covariance_matrix=factor* cov_tensor[k]).log_prob(normalised_hsi[j])
            #         ))
                    
            #     for k in range(len(pi_k)):
            #         key = f"data{k+1}"
            #         gamma_nk[j][k] = (z_nk[j,k] * torch.exp(
            #             dist.MultivariateNormal(loc=RV_post_mean_data[key], covariance_matrix=factor* cov_tensor[k]).log_prob(normalised_hsi[j]))/ likelihood
            #         )
                
            #     log_likelihood += torch.log(likelihood)
                
                
            
            gmm_label_new = torch.argmax(gamma_nk,dim=1) +1
            gmm_accuracy = torch.sum(gmm_label_new == y_obs_label) / y_obs_label.shape[0]
            store_gmm_accuracy.append(gmm_accuracy)
            
            entropy_gmm = calculate_entropy(gamma_nk.detach().numpy())
            entropy_z_nk = calculate_average_entropy(z_nk.detach().numpy())
            entropy_pi_k = calculate_entropy(pi_k.detach().numpy())
            store_z_nk_entropy.append(entropy_z_nk)
            store_gmm_entropy.append(entropy_gmm)
            store_mixing_entropy.append(entropy_pi_k)
            unnormalise_posterior_value["log_prior_geo_list"].append(log_prior_geo)
            unnormalise_posterior_value["log_prior_hsi_pi_list"].append(log_prior_hsi_pi)
            unnormalise_posterior_value["log_prior_hsi_mean_list"].append(log_prior_hsi_mean)
            unnormalise_posterior_value["log_prior_hsi_cov_list"].append(log_prior_hsi_cov)
            unnormalise_posterior_value["log_likelihood_list"].append(log_likelihood)
            unnormalise_posterior_value["log_posterior_list"].append(log_prior_geo + log_prior_hsi_pi + log_prior_hsi_mean + log_prior_hsi_cov + log_likelihood)
            
        plt.figure(figsize=(8,10))
        plt.plot(np.array([ele.detach().numpy() for ele in unnormalise_posterior_value["log_prior_geo_list"]]), label='prior_geo', marker=".")
        plt.plot(np.array([ele.detach().numpy() for ele in unnormalise_posterior_value["log_prior_hsi_mean_list"]]), label='prior_hsi_mean', marker="*")
        plt.plot(np.array([ele.detach().numpy() for ele in unnormalise_posterior_value["log_prior_hsi_cov_list"]]), label='prior_hsi_cov', marker="_")
        plt.plot(np.array([ele.detach().numpy() for ele in unnormalise_posterior_value["log_likelihood_list"]]), label='prior_likelihood', marker="d")
        plt.plot(np.array([ele.detach().numpy() for ele in unnormalise_posterior_value["log_posterior_list"]]), label='posterior', marker="+")
        plt.xlabel('Iteration')
        plt.ylabel('Unnormalised log value')
        plt.title('Log value of each term in posterior')
        plt.legend()
        filename_log = directory_path_MAP + "/log_posterior.png"
        plt.savefig(filename_log)
        plt.close()
    
    elif posterior_condition==4:
        
        print("Posterior 4 reached")
        
        unnormalise_posterior_value["log_prior_geo_list"]=[]
        unnormalise_posterior_value["log_prior_hsi_mean_list"]=[]
        unnormalise_posterior_value["log_prior_hsi_cov_list"]=[]
        unnormalise_posterior_value["log_likelihood_list"]=[]
        unnormalise_posterior_value["log_posterior_list"]=[]
        
        keys_list = list(posterior_samples.keys())

        
        prior_mean_surface = [item['normal']['mean'].item() for item in test_list]
        prior_std_surface =  [item['normal']['std'].item() for item in test_list[:num_layers]]
        num_data= torch.tensor(mean_init,dtype=dtype, device =device).shape[0]
        
        
        RV_post_mu = {}
        RV_post_mean_data = {}
        RV_post_pi={}
        RV_post_cov_upper = {}
        
        
        
        
        for i in range(posterior_samples[keys_list[0]].shape[0]):
    
            
            for j in range(num_layers):
                key = f"mu_{j+1}"
                RV_post_mu[key] = posterior_samples[keys_list[ cluster + j]][i]  

            
            for j in range(cluster):
                key = f"mean_data_{j+1}"
                RV_post_mean_data[key] = posterior_samples[keys_list[ j]][i]  

            RV_post_pi["pi"] = posterior_samples[keys_list[2 * cluster + num_layers]][i]
            
            for j in range(cluster):
                key = f"upper_tri_cov_{j+1}"
                RV_post_cov_upper[key] = posterior_samples[keys_list[cluster + num_layers+ j]][i]
        
        
            log_prior_geo = torch.tensor(0.0, dtype=dtype, device =device)
            for i in range(num_layers):
                log_prior_geo+=dist.Normal(prior_mean_surface[i], prior_std_surface[i]).log_prob(RV_post_mu[f"mu_{i+1}"])
            
            
            
            # Update the model with the new top layer's location
            interpolation_input = geo_model_test.interpolation_input
            counter1=1
            for interpolation_input_data in test_list[:num_layers]:
                interpolation_input.surface_points.sp_coords = torch.index_put(interpolation_input.surface_points.sp_coords,(interpolation_input_data["id"], torch.tensor([2])), RV_post_mu["mu_"+ str(counter1)])
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
            
            
            z_nk = F.softmax(-scale* (torch.linspace(1,cluster,cluster, dtype=dtype, device =device) - custom_grid_values.reshape(-1,1))**2, dim=1)
            N_k = torch.sum(z_nk,axis=0)
            N = len(custom_grid_values)
            pi_k = N_k /N
            
            loc_mean = torch.tensor(mean_init,dtype=dtype, device =device)
            
            cov_matrix_mean = alpha * torch.eye(loc_mean[0].shape[0], dtype=dtype, device =device)
            
            
            D = loc_mean.shape[1]
            
            # Mean
            log_prior_hsi_mean =torch.tensor(0.0, dtype=dtype, device =device)
            for j in range(loc_mean.shape[0]):
                log_prior_hsi_mean = log_prior_hsi_mean + dist.MultivariateNormal(loc=loc_mean[j],covariance_matrix=cov_matrix_mean).log_prob(posterior_samples[keys_list[j]][i])
                
            
            
            # for covariance
            log_prior_hsi_cov =torch.tensor(0.0, dtype=dtype, device =device)
            cov = []
            n = loc_mean.shape[1]
            num_upper_tri_elements = n * (n + 1) // 2
            
            cov_matrix_cov = beta * torch.eye(num_upper_tri_elements, dtype=dtype, device =device)
            
            for j in range(loc_mean.shape[0]):
                A = torch.zeros((n,n), dtype=dtype, device =device)
                upper_tri_cov = posterior_samples[keys_list[cluster + num_layers+ j]][i]   
                
                
                log_prior_hsi_cov = log_prior_hsi_cov + dist.MultivariateNormal(loc=torch.zeros(num_upper_tri_elements, dtype=dtype, device =device), covariance_matrix=cov_matrix_cov).log_prob(upper_tri_cov)
                
                # Get the upper triangular indices
                upper_tri_indices = torch.triu_indices(n, n)
                
                # Assign the sampled elements to the upper triangular positions
                A = A.index_put((upper_tri_indices[0], upper_tri_indices[1]),upper_tri_cov)
                    # Symmetrize the matrix A
                A = A + A.T - torch.diag(A.diagonal())
                
                cov_data = torch.matrix_exp(A)
                
                cov.append(cov_data)
            cov_tensor = torch.stack(cov, dim=0)
            
            # We can also calculate the accuracy using the mean and covariance to see if our GMM model has imroved or not
            # gamma_nk = torch.zeros(z_nk.shape)  
            # log_likelihood=torch.tensor(0.0, dtype=dtype, device =device)

            # for j in range(normalised_hsi.shape[0]):
            #     likelihood = 0.0  
                
            #     for k in range(len(pi_k)):
            #         key = f"mean_data_{k+1}"
            #         likelihood += z_nk[j,k] * torch.exp(
            #             dist.MultivariateNormal(loc=RV_post_mean_data[key], covariance_matrix= factor* cov_tensor[k]).log_prob(normalised_hsi[j])
            #         )
                    
            #     for k in range(len(pi_k)):
            #         key = f"mean_data_{k+1}"
            #         gamma_nk[j][k] = (z_nk[j,k] * torch.exp(
            #             dist.MultivariateNormal(loc=RV_post_mean_data[key], covariance_matrix=factor *cov_tensor[k]).log_prob(normalised_hsi[j]))/ likelihood
            #         )
                
            #     log_likelihood += torch.log(likelihood)
                
            gamma_nk = z_nk
            log_likelihood=torch.tensor(0.0, dtype=dtype, device =device)

            log_prior_hsi_pi = dist.Dirichlet(torch.ones(int(len(pi_k)), dtype=dtype, device=device)).log_prob(RV_post_pi["pi"])
            
            log_pi = torch.log(RV_post_pi["pi"]).unsqueeze(0)
            gmm_log_probs = dist.MultivariateNormal(loc=mean_tensor, covariance_matrix=cov_tensor).log_prob(normalised_hsi.unsqueeze(1))
            log_gmm_broadcast = log_pi + gmm_log_probs
            log_likelihood = torch.sum(z_nk * log_gmm_broadcast, axis=1).sum()
            
            gmm_label_new = torch.argmax(gamma_nk,dim=1) +1
            gmm_accuracy = torch.sum(gmm_label_new == y_obs_label) / y_obs_label.shape[0]
            store_gmm_accuracy.append(gmm_accuracy)
            
            entropy_gmm = calculate_entropy(gamma_nk.detach().numpy())
            entropy_z_nk = calculate_average_entropy(z_nk.detach().numpy())
            entropy_pi_k = calculate_entropy(pi_k.detach().numpy())
            store_z_nk_entropy.append(entropy_z_nk)
            store_gmm_entropy.append(entropy_gmm)
            store_mixing_entropy.append(entropy_pi_k)
                
            unnormalise_posterior_value["log_prior_geo_list"].append(log_prior_geo)
            unnormalise_posterior_value["log_prior_hsi_mean_list"].append(log_prior_hsi_mean)
            unnormalise_posterior_value["log_prior_hsi_cov_list"].append(log_prior_hsi_cov)
            unnormalise_posterior_value["log_likelihood_list"].append(log_likelihood)
            unnormalise_posterior_value["log_posterior_list"].append(log_prior_geo +log_prior_hsi_mean + log_prior_hsi_cov +log_likelihood)
        
        plt.figure(figsize=(8,10))
        plt.plot(np.array([ele.detach().detach() for ele in unnormalise_posterior_value["log_prior_geo_list"]]), label='prior_geo', marker=".")
        plt.plot(np.array([ele.detach().detach() for ele in unnormalise_posterior_value["log_prior_hsi_mean_list"]]), label='prior_hsi_mean', marker="*")
        plt.plot(np.array([ele.detach().detach() for ele in unnormalise_posterior_value["log_prior_hsi_cov_list"]]), label='prior_hsi_cov', marker="_")
        plt.plot(np.array([ele.detach().detach() for ele in unnormalise_posterior_value["log_likelihood_list"]]), label='prior_likelihood', marker="d")
        plt.plot(np.array([ele.detach().detach() for ele in unnormalise_posterior_value["log_posterior_list"]]), label='posterior', marker="+")
        plt.xlabel('Iteration')
        plt.ylabel('Unnormalised log value')
        plt.title('Log value of each term in posterior')
        plt.legend()
        filename_log = directory_path_MAP + "/log_posterior.png"
        plt.savefig(filename_log)
        plt.close()
    
        
     
    MAP_sample_index=torch.argmax(torch.tensor(unnormalise_posterior_value["log_posterior_list"]))
    plt.figure(figsize=(10,8))
    plt.plot(torch.arange(len(store_accuracy))+1, torch.tensor(store_accuracy))
    plt.savefig(directory_path_MAP + "/accuracy.png")
    plt.close()
    
    plt.figure(figsize=(10,8))
    plt.plot(torch.arange(len(store_accuracy))+1, torch.tensor(store_gmm_accuracy))
    plt.savefig(directory_path_MAP +"/accuracy_gmm.png")
    plt.close()
    
    
    
    plt.figure(figsize=(8,10))
    plt.plot(np.array(store_z_nk_entropy), label="Responsibility Entropy")
    plt.plot(np.array(store_gmm_entropy), label = 'GMM Entropy')
    plt.plot(np.array(store_mixing_entropy), label="Mixing Coefficient Entropy")
    plt.xlabel('Iteration')
    plt.ylabel('average entropy')
    plt.title('Average entropy of the sample')
    plt.legend()
    filename_entropy = directory_path_MAP + "/average_entropy.png"
    plt.savefig(filename_entropy)
    plt.close()   
    
    filename_posterior_samples =directory_path + "/posterior_samples.json"
    # Save to a JSON file
    with open(filename_posterior_samples, 'w') as f:
        json.dump(posterior_samples_serializable, f)
    return MAP_sample_index, unnormalise_posterior_value["log_posterior_list"][MAP_sample_index] , [ele.detach() for ele in unnormalise_posterior_value["log_posterior_list"]  ] 