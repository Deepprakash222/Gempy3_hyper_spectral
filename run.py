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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import argparse

from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import KMeans
from model1 import model_test
def cluster_acc(Y_pred, Y, ignore_label=None):
    """ Rearranging the class labels of prediction so that it maximise the 
        match class labels.

    Args:
        Y_pred (int): An array for predicted labels
        Y (float): An array for true labels
        ignore_label (int, optional): Laels to be ignored

    Returns:
        _type_: _description_
    """
    if ignore_label is not None:
        index = Y!= ignore_label
        Y=Y[index]
        Y_pred=Y_pred[index]
    from scipy.optimize import linear_sum_assignment as linear_assignment
    assert Y_pred.shape == Y.shape
    D = int((max(Y_pred.max(), Y.max())).item())
    w = torch.zeros((D, D))
    for i in range(Y_pred.shape[0]):
        w[int(Y_pred[i].item())-1, int(Y[i].item())-1] += 1
    ind = linear_assignment(w.max() - w)
    return ind[0], ind[1], (w[ind[0], ind[1]]).sum() / Y_pred.shape[0], w

def TSNE_transformation(data, label, filename):
    from sklearn.manifold import TSNE
    model = TSNE(n_components=2, random_state=0)
    transformed_data = model.fit_transform(data) 
    colors = ['black', 'red', 'green', 'blue', 'yellow', 'purple', 'orange']
    plt.figure(figsize=(10,8))
    plt.scatter(transformed_data[:,0],transformed_data[:,1],c= label, cmap=ListedColormap(colors[1:]))
    # Add axis labels
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    # Create a legend
    legend_labels = np.unique(label)
    
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=ListedColormap(colors[1:])(i/len(legend_labels)), markersize=10) for i in legend_labels]
    plt.legend(handles, legend_labels, title="Labels")
    plt.savefig(filename)
    
# def find_MAP(geo_model, posterior_sample_dict,sample_size):
#     for i in range(sample_size):
#         log_posterior_unnormalized = 0
#         for key, values in posterior_sample_dict:
#                 log_posterior_unnormalized += 
    
def main(startval, endval, dimred, numlayers):
    '''
    This function defines a model which uses hyperspectral data, applies clustering methods to find cluster information and then uses Bayesian Modelling to model a posterior.
    The arguments passed to main are:
    startval: It defines the starting X column value of Hyperspectral data.
    endval: It defines the end X column value of Hyperspectral data.
    dimred: Defines the type of dimensionality reduction user wants to follow .Currently either pca or tsne.
    
    '''
    print("This is dev3")
    # Load .mat file
    SalinasA= np.array(scipy.io.loadmat('./HSI_Salinas/SalinasA.mat')['salinasA'])
    SalinasA_corrected= np.array(scipy.io.loadmat('./HSI_Salinas/SalinasA_corrected.mat')['salinasA_corrected'])
    SalinasA_gt= np.array(scipy.io.loadmat('./HSI_Salinas/SalinasA_gt.mat')['salinasA_gt'])
    # Arrange the label in groundtruth
    i=0
    label_data = [0,6,1,5,4,3,2]
    
    for ele in np.unique(SalinasA_gt):
        mask = SalinasA_gt==ele
        SalinasA_gt[mask] = label_data[i]
        i=i+1
    
    ######################################################################
    ## Arrange Data as concatenation of spacial co-ordinate and pixel values
    ###########################################################################
    H, W = SalinasA_gt.shape # get the shape of ground truth
    n_features = SalinasA_corrected.shape[2]+4 # get the number of features including co-ordinates and label
    
    # Create a dataset which has "X","Y","Z", "Label", and spectral channel information
    data_hsi = torch.zeros((H*W, n_features ))
    for i in range(H):
        for j in range(W):
            data_hsi[i*W+j,0] = j
            data_hsi[i*W +j,2] = - i
            data_hsi[i*W +j,3] = SalinasA_gt[i,j]
            data_hsi[i*W +j,4:] = torch.tensor(SalinasA_corrected[i,j,:])
            
    # Create a list of column name
    column_name=["X","Y","Z", "Label"]
    for i in range(SalinasA_corrected.shape[2]):
        column_name.append("feature_"+str(i+1))
        
    # Create a pandas dataframe to store the database
    df_hsi = pd.DataFrame(data_hsi,columns=column_name)
    # Create a database by removing the non labelled pixel information 
    df_with_non_labelled_pixel = df_hsi.loc[(df_hsi['Label']!=0)]
    
    # Normalise along the spectral lines 
    df_with_spectral_normalised = df_with_non_labelled_pixel.copy()
    df_with_spectral_normalised.iloc[:, 4:] = df_with_spectral_normalised.iloc[:, 4:].apply(zscore,axis=1)
    
    
    # column = 20
    # y_obs = torch.tensor(SalinasA_gt[:,column], dtype=torch.float64)
    # mask = y_obs!=0
    # y_obs_label = y_obs[mask]
    # y_obs_hsi = torch.tensor(SalinasA_corrected[:,column,:], dtype=torch.float64)[mask]
    
    # normalised_hsi = zscore(y_obs_hsi, axis=1)
    
    
    ###########################################################################
    ## Obtain the preprocessed data
    ###########################################################################
    normalised_data = df_with_spectral_normalised.loc[(df_with_spectral_normalised["X"]>=18)&(df_with_spectral_normalised["X"]<=22)]
    normalised_hsi =torch.tensor(normalised_data.iloc[:,4:].to_numpy(), dtype=torch.float64)
    
    ## It is difficult to work with data in such a high dimensions, because the covariance matrix 
    ## determinant quickly goes to zero even if eigen-values are in the range of 1e-3. Therefore it is advisable 
    ## to fist apply dimensionality reduction to a lower dimensions
    from sklearn.decomposition import PCA
    pca = PCA(n_components=10)
    transformed_hsi = pca.fit_transform(normalised_hsi)
    normalised_hsi = torch.tensor(transformed_hsi, dtype=torch.float64)
    y_obs_label = torch.tensor(normalised_data.iloc[:,3].to_numpy(), dtype=torch.float64)
    
    ###########################################################################
    ## Apply Classical clustering methods to find different cluster information our data
    ###########################################################################
    gm =gm = BayesianGaussianMixture(n_components=6, random_state=42).fit(normalised_hsi)
    
    # make the labels to start with 1 instead of 0
    gmm_label = gm.predict(normalised_hsi) +1 
    gmm_label_order, y_obs_label_order, accuracy_init, _ = cluster_acc( gmm_label, y_obs_label)
    
    
    # reaarange the label information so it is would be consistent with ground truth label
    gmm_label_rearranged = torch.tensor([y_obs_label_order[x-1] +1  for x in gmm_label], dtype=torch.float64)
    
    #print(gmm_label_rearranged - y_obs_label)
    
    # gmm_label2 = torch.zeros_like(y_obs_label)
    # gmm_label2[gmm_label==2]=6
    # gmm_label2[gmm_label==4]=5
    # gmm_label2[gmm_label==1]=4
    # gmm_label2[gmm_label==3]=3
    # gmm_label2[gmm_label==6]=2
    # gmm_label2[gmm_label==5]=1
    
   
    # rearrange the mean and covariance accordingly too
    #rearrange_list = [4,5,2,0,3,1]
    #rearrange_list = [3,4,2,0,5,1]
    rearrange_list = y_obs_label_order
    mean_init, cov_init = gm.means_[rearrange_list], gm.covariances_[rearrange_list]
    ####################################TODO#################################################
    #   Try to find the initial accuracy of classification
    #########################################################################################
    print("Intial accuracy\n", accuracy_init)
    
    #################################TODO##################################################
    ## Apply different dimentionality reduction techniques and save the plot in Result file
    #######################################################################################
    # TSNE_transformation(data=normalised_data, label=gmm_label_rearranged, filename="./Results/tsne_gmm_label.png")
    if(dimred=="tsne"):
        TSNE_transformation(data=normalised_data, label=gmm_label_rearranged, filename="./Results/tsne_gmm_label.png")
    elif(dimred=="pca"):
        PCA_transformation(data=normalised_data, label=gmm_label_rearranged, filename="./Results/pca_gmm_label.png")
    ######################################################################################
    ## Apply Classical clustering methods to find different cluster information our data
    ######################################################################################
  
    geo_model_test = gp.create_geomodel(
    project_name='Gempy_abc_Test',
    extent=[0, 86, -10, 10, -83, 0],
    resolution=[86,20,83],
    refinement=3,
    structural_frame= gp.data.StructuralFrame.initialize_default_structure()
    )

    gp.add_surface_points(
        geo_model=geo_model_test,
        x=[70.0, 80.0],
        y=[0.0, 0.0],
        z=[-77.0, -71.0],
        elements_names=['surface1', 'surface1']
    )

    gp.add_orientations(
        geo_model=geo_model_test,
        x=[75],
        y=[0.0],
        z=[-74],
        elements_names=['surface1'],
        pole_vector=[[-5/3, 0, 1]]
    )
    geo_model_test.update_transform(gp.data.GlobalAnisotropy.NONE)

    element2 = gp.data.StructuralElement(
        name='surface2',
        color=next(geo_model_test.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([20.0, 60.0]),
            y=np.array([0.0, 0.0]),
            z=np.array([-74, -52]),
            names='surface2'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    geo_model_test.structural_frame.structural_groups[0].append_element(element2)

    element3 = gp.data.StructuralElement(
        name='surface3',
        color=next(geo_model_test.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([0.0, 30.0, 60]),
            y=np.array([0.0, 0.0,0.0]),
            z=np.array([-72, -55.5, -39]),
            names='surface3'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    geo_model_test.structural_frame.structural_groups[0].append_element(element3)

    element4 = gp.data.StructuralElement(
        name='surface4',
        color=next(geo_model_test.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([0.0, 20.0, 60]),
            y=np.array([0.0, 0.0,0.0]),
            z=np.array([-61, -49, -27]),
            names='surface4'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    geo_model_test.structural_frame.structural_groups[0].append_element(element4)

    element5 = gp.data.StructuralElement(
        name='surface5',
        color=next(geo_model_test.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([0.0, 20.0, 40]),
            y=np.array([0.0, 0.0, 0.0]),
            z=np.array([-39, -28, -16]),
            names='surface5'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    geo_model_test.structural_frame.structural_groups[0].append_element(element5)

    element6 = gp.data.StructuralElement(
        name='surface6',
        color=next(geo_model_test.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([0.0, 20.0,30]),
            y=np.array([0.0, 0.0, 0.0]),
            z=np.array([-21, -10, -1]),
            names='surface6'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    geo_model_test.structural_frame.structural_groups[0].append_element(element6)

    geo_model_test.structural_frame.structural_groups[0].elements[0], geo_model_test.structural_frame.structural_groups[0].elements[1],\
    geo_model_test.structural_frame.structural_groups[0].elements[2], geo_model_test.structural_frame.structural_groups[0].elements[3],\
    geo_model_test.structural_frame.structural_groups[0].elements[4], geo_model_test.structural_frame.structural_groups[0].elements[5] = \
    geo_model_test.structural_frame.structural_groups[0].elements[1], geo_model_test.structural_frame.structural_groups[0].elements[0],\
    geo_model_test.structural_frame.structural_groups[0].elements[3], geo_model_test.structural_frame.structural_groups[0].elements[2],\
    geo_model_test.structural_frame.structural_groups[0].elements[5], geo_model_test.structural_frame.structural_groups[0].elements[4]  


    gp.compute_model(geo_model_test)
    picture_test = gpv.plot_2d(geo_model_test, cell_number=5, legend='force')
    plt.savefig("./Results/Prior_model.png")
    
    # Label information need to be in same order as it is created in gempy model
    
    y_obs_label = 7 - y_obs_label
    
    ################################################################################
    # Custom grid
    ################################################################################
    # x_loc = 20
    # y_loc = 0
    # z_loc = np.linspace(0,-82, 83)
    # xyz_coord = np.array([[x_loc, y_loc, z] for z in z_loc])[mask] 
    xyz_coord = normalised_data.iloc[:,:3].to_numpy()
    gp.set_custom_grid(geo_model_test.grid, xyz_coord=xyz_coord)
    ################################################################################
    
    geo_model_test.interpolation_options.mesh_extraction = False
    sol = gp.compute_model(geo_model_test)
    
    sp_coords_copy_test = geo_model_test.interpolation_input.surface_points.sp_coords.copy()
    geo_model_test.transform.apply_inverse(sp_coords_copy_test)
    
    gp.compute_model(geo_model_test)
    
    sp_coords_copy_test = geo_model_test.interpolation_input.surface_points.sp_coords.copy()
    
    ################################################################################
    # Store the Initial Interface data and orientation data
    ################################################################################
    df_sp_init = geo_model_test.surface_points.df
    df_or_init = geo_model_test.orientations.df
    
    df_sp_init.to_csv("./Results/Initial_sp.csv")
    df_or_init.to_csv("./Results/Initial_or.csv")
    ################################################################################
    
    geo_model_test.transform.apply_inverse(sp_coords_copy_test)
    
    # Change the backend to PyTorch for probabilistic modeling
    BackendTensor.change_backend_gempy(engine_backend=gp.data.AvailableBackends.PYTORCH)
    
    # geo_model_test.interpolation_options.uni_degree = 0
    # geo_model_test.interpolation_options.mesh_extraction = False
    geo_model_test.interpolation_options.sigmoid_slope = 40
    
    factor=0.01
    test_list=[]
    test_list.append({"update":"interface_data","id":11, "direction":"Z", "prior_distribution":"normal","normal":{"mean":torch.tensor(3.2,dtype=torch.float64), "std":torch.tensor(0.02,dtype=torch.float64)}})
    test_list.append({"update":"interface_data","id":14, "direction":"Z", "prior_distribution":"normal","normal":{"mean":torch.tensor(2.3,dtype=torch.float64), "std":torch.tensor(0.02,dtype=torch.float64)}})
    test_list.append({"update":"interface_data","id":5, "direction":"Z", "prior_distribution":"normal","normal":{"mean":torch.tensor(1.25,dtype=torch.float64), "std":torch.tensor(0.02,dtype=torch.float64)}})
    test_list.append({"update":"interface_data","id":0, "direction":"Z", "prior_distribution":"normal","normal":{"mean":torch.tensor(0.0,dtype=torch.float64), "std":torch.tensor(0.02,dtype=torch.float64)}})

    #@config_enumerate
                
    
    ################################################################################
    # Prior
    ################################################################################
    pyro.set_rng_seed(42)
    prior = Predictive(model_test, num_samples=10)(normalised_hsi,test_list,geo_model_test,mean_init,cov_init,factor)
    # Key to avoid
    #avoid_key = ['mu_1 < 0','mu_1 > mu_2','mu_2 > -38.5', 'mu_3 < -38.5','mu_3 > -61.4','mu_4 < -61.5', 'mu_4 > -83']
    avoid_key = ['mu_1 < 0','mu_1 > mu_2','mu_2 > mu_3', 'mu_3 > mu_4' , 'mu_4 > -83']
    # Create sub-dictionary without the avoid_key
    prior = dict((key, value) for key, value in prior.items() if key not in avoid_key)
    data = az.from_pyro(prior=prior)
    
    ################################################################################
    # Posterior 
    ################################################################################

    
    pyro.primitives.enable_validation(is_validate=True)
    nuts_kernel = NUTS(model_test, step_size=0.0085, adapt_step_size=True, target_accept_prob=0.9, max_tree_depth=10, init_strategy=init_to_mean)
    mcmc = MCMC(nuts_kernel, num_samples=150, warmup_steps=50, disable_validation=False)
    mcmc.run(normalised_hsi,test_list,geo_model_test,mean_init,cov_init,factor)
    
    posterior_samples = mcmc.get_samples()
    posterior_predictive = Predictive(model_test, posterior_samples)(normalised_hsi,test_list,geo_model_test,mean_init,cov_init,factor)
    data = az.from_pyro(posterior=mcmc, prior=prior, posterior_predictive=posterior_predictive)
    
    # Find the MAP value
    
    unnormalise_posterior_value={}
    unnormalise_posterior_value["log_prior_geo_list"]=[]
    unnormalise_posterior_value["log_prior_hsi_list"]=[]
    unnormalise_posterior_value["log_likelihood_list"]=[]
    unnormalise_posterior_value["log_posterior_list"]=[]
    # log_prior_geo_list=[]
    # log_prior_hsi_list=[]
    # log_likelihood_list=[]
    # log_posterior_list=[]
    keys_list = list(posterior_samples.keys())
   
    prior_mean_surface_1 = sp_coords_copy_test[11, 2]
    prior_mean_surface_2 = sp_coords_copy_test[14, 2]
    prior_mean_surface_3 = sp_coords_copy_test[5, 2]
    prior_mean_surface_4 = sp_coords_copy_test[0, 2]
    
    store_accuracy=[]
    #number of layers returned from model
    num_layers=4
    RV_mu_post ={}
    RV_sampledata_post ={} 
    # Create a dictionary which can store all the random variable of our model

    for i in range(posterior_samples["mu_1"].shape[0]):
        for j in range(0,num_layers):
            RV_mu_post["mu_"+ str(j+1)] = posterior_samples[keys_list[j]][i]
        for j, k in zip(range(1, 7), range(4, 10)):
            RV_sampledata_post["post_sample_data"+ str(j)] = posterior_samples[keys_list[k]][i]

    
    #for i in range(posterior_samples["mu_1"].shape[0]):
        # post_mu_1 = posterior_samples[keys_list[0]][i]
        # post_mu_2 = posterior_samples[keys_list[1]][i]
        # post_mu_3 = posterior_samples[keys_list[2]][i]
        # post_mu_4 = posterior_samples[keys_list[3]][i]
        # post_sample_data1 = posterior_samples[keys_list[4]][i]
        # post_sample_data2 = posterior_samples[keys_list[5]][i]
        # post_sample_data3 = posterior_samples[keys_list[6]][i]
        # post_sample_data4 = posterior_samples[keys_list[7]][i]
        # post_sample_data5 = posterior_samples[keys_list[8]][i]
        # post_sample_data6 = posterior_samples[keys_list[9]][i]
        
        
        # Calculate the log probability of the value
        log_prior_geo = dist.Normal(prior_mean_surface_1, torch.tensor(0.2, dtype=torch.float64)).log_prob(RV_mu_post["mu_1"])+\
                dist.Normal(prior_mean_surface_2, torch.tensor(0.2, dtype=torch.float64)).log_prob(RV_mu_post["mu_2"])+\
                dist.Normal(prior_mean_surface_3, torch.tensor(0.2, dtype=torch.float64)).log_prob(RV_mu_post["mu_3"])+\
                dist.Normal(prior_mean_surface_4, torch.tensor(0.2, dtype=torch.float64)).log_prob(RV_mu_post["mu_4"])
    
        # log_prior_geo = dist.Normal(prior_mean_surface_1, torch.tensor(0.2, dtype=torch.float64)).log_prob(post_mu_1)+\
        #             dist.Normal(prior_mean_surface_2, torch.tensor(0.2, dtype=torch.float64)).log_prob(post_mu_2)+\
        #             dist.Normal(prior_mean_surface_3, torch.tensor(0.2, dtype=torch.float64)).log_prob(post_mu_3)+\
        #             dist.Normal(prior_mean_surface_4, torch.tensor(0.2, dtype=torch.float64)).log_prob(post_mu_4)
        
        # Update the model with the new top layer's location
        interpolation_input = geo_model_test.interpolation_input
        counter1=1
        for interpolation_input_data in test_list[:num_layers]:
            interpolation_input.surface_points.sp_coords = torch.index_put(interpolation_input.surface_points.sp_coords,(torch.tensor([interpolation_input_data["id"]]), torch.tensor([2])), RV_mu_post["post_mu_"+ str(counter1)])
            counter1=counter1+1
        
        # interpolation_input.surface_points.sp_coords = torch.index_put(
        #     interpolation_input.surface_points.sp_coords,
        #     (torch.tensor([11]), torch.tensor([2])),
        #     post_mu_1
        # )
        # interpolation_input.surface_points.sp_coords = torch.index_put(
        #     interpolation_input.surface_points.sp_coords,
        #     (torch.tensor([14]), torch.tensor([2])),
        #     post_mu_2
        # )
        
        # interpolation_input.surface_points.sp_coords = torch.index_put(
        #     interpolation_input.surface_points.sp_coords,
        #     (torch.tensor([5]), torch.tensor([2])),
        #     post_mu_3
        # )
        # interpolation_input.surface_points.sp_coords = torch.index_put(
        #     interpolation_input.surface_points.sp_coords,
        #     (torch.tensor([0]), torch.tensor([2])),
        #     post_mu_4
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
        accuracy_intermediate = torch.sum(torch.round(custom_grid_values) == y_obs_label) / y_obs_label.shape[0]
        store_accuracy.append(accuracy_intermediate)
        lambda_ = 15.0
        loc_mean = torch.tensor(mean_init,dtype=torch.float64)
        loc_cov =  torch.tensor(cov_init, dtype=torch.float64)
        #class_label = F.softmax(-lambda_* (torch.tensor([1,2,3,4,5,6], dtype=torch.float64) - custom_grid_values.reshape(-1,1))**2, dim=1)
        pi_k = torch.mean(F.softmax(-lambda_* (torch.tensor([1,2,3,4,5,6], dtype=torch.float64) - custom_grid_values.reshape(-1,1))**2, dim=1),dim=0)
        
        num_components = len(loc_mean)
        log_prior_hsi = torch.tensor(0.0, dtype=torch.float64)
        for l in range(num_components):
            log_prior_hsi += dist.MultivariateNormal(loc=loc_mean[l],covariance_matrix=loc_cov[l]).log_prob(RV_sampledata_post["post_sample_data"+ str(l+1)]) 

    
        # log_prior_hsi = dist.MultivariateNormal(loc=loc_mean[0],covariance_matrix=loc_cov[0]).log_prob(post_sample_data1)+\
        #                 dist.MultivariateNormal(loc=loc_mean[1],covariance_matrix=loc_cov[1]).log_prob(post_sample_data2)+\
        #                 dist.MultivariateNormal(loc=loc_mean[2],covariance_matrix=loc_cov[2]).log_prob(post_sample_data3)+\
        #                 dist.MultivariateNormal(loc=loc_mean[3],covariance_matrix=loc_cov[3]).log_prob(post_sample_data4)+\
        #                 dist.MultivariateNormal(loc=loc_mean[4],covariance_matrix=loc_cov[4]).log_prob(post_sample_data5)+\
        #                 dist.MultivariateNormal(loc=loc_mean[5],covariance_matrix=loc_cov[5]).log_prob(post_sample_data6)
        log_likelihood=torch.tensor(0.0, dtype=torch.float64)
        for j1, k1 in zip(range(normalised_hsi.shape[0]), range(6)):
            likelihood = pi_k[k1] *torch.exp(dist.MultivariateNormal(loc=RV_sampledata_post["post_sample_data"+ str(k1+1)],covariance_matrix= factor * loc_cov[k1]).log_prob(normalised_hsi[j1])) 
            log_likelihood += torch.log(likelihood)
        # for j in range(normalised_hsi.shape[0]):
        #     likelihood = pi_k[0] *torch.exp(dist.MultivariateNormal(loc=post_sample_data1,covariance_matrix= factor * loc_cov[0]).log_prob(normalised_hsi[j])) +\
        #                  pi_k[1] *torch.exp(dist.MultivariateNormal(loc=post_sample_data2,covariance_matrix= factor * loc_cov[1]).log_prob(normalised_hsi[j]))+\
        #                  pi_k[2] *torch.exp(dist.MultivariateNormal(loc=post_sample_data3,covariance_matrix= factor * loc_cov[2]).log_prob(normalised_hsi[j])) +\
        #                  pi_k[2] *torch.exp(dist.MultivariateNormal(loc=post_sample_data4,covariance_matrix= factor * loc_cov[3]).log_prob(normalised_hsi[j])) +\
        #                  pi_k[4] *torch.exp(dist.MultivariateNormal(loc=post_sample_data5,covariance_matrix= factor * loc_cov[4]).log_prob(normalised_hsi[j])) +\
        #                  pi_k[5] *torch.exp(dist.MultivariateNormal(loc=post_sample_data6,covariance_matrix= factor * loc_cov[5]).log_prob(normalised_hsi[j])) 
        #     log_likelihood += torch.log(likelihood)
        
        # log_prior_geo_list.append(log_prior_geo)
        # log_prior_hsi_list.append(log_prior_hsi)
        # log_likelihood_list.append(log_likelihood)
        # log_posterior_list.append(log_prior_geo + log_prior_hsi + log_likelihood)
        unnormalise_posterior_value["log_prior_geo_list"].append(log_prior_geo)
        unnormalise_posterior_value["log_prior_hsi_list"].append(log_prior_hsi)
        unnormalise_posterior_value["log_likelihood_list"].append(log_likelihood)
        unnormalise_posterior_value["log_posterior_list"].append(log_prior_geo + log_prior_hsi + log_likelihood)
    
    MAP_sample_index=torch.argmax(torch.tensor(unnormalise_posterior_value["log_posterior_list"]))
    plt.figure(figsize=(10,8))
    plt.plot(torch.arange(len(store_accuracy))+1, torch.tensor(store_accuracy))
    plt.savefig("./Results/accuracy.png")
    
    # Extract acceptance probabilities
    
    # # Extract the diagnostics
    # diagnostics = mcmc.diagnostics()
    # accept_probs = diagnostics["accept_prob"]

    # # Plot the acceptance probabilities
    # plt.plot(accept_probs)
    # plt.xlabel('Iteration')
    # plt.ylabel('Acceptance Probability')
    # plt.title('Acceptance Probabilities of NUTS Sampler')
    # plt.savefig("./Results/Acceptance_Probabilities_of_NUTS_Sampler")
    
    ################################################################################
    #  Try Plot the data and save it as file in output folder
    ################################################################################
    # mu_1_post = posterior_samples["mu_1"].mean()
    # mu_2_post = posterior_samples["mu_2"].mean()
    # mu_3_post = posterior_samples["mu_3"].mean()
    # mu_4_post = posterior_samples["mu_4"].mean()
    RV_mu_post2 = {}
    num_layers=4
    for i in range(1,num_layers):
        RV_mu_post2["mu_"+str(i)+"_post"] = posterior_samples["mu_"+str(i)][MAP_sample_index]

    # mu_1_post = posterior_samples["mu_1"][MAP_sample_index]
    # mu_2_post = posterior_samples["mu_2"][MAP_sample_index]
    # mu_3_post = posterior_samples["mu_3"][MAP_sample_index]
    # mu_4_post = posterior_samples["mu_4"][MAP_sample_index]
    
    # # Update the model with the new top layer's location
    counter2=1
    for interpolation_input_data in test_list[:num_layers]:
        interpolation_input.surface_points.sp_coords = torch.index_put(interpolation_input.surface_points.sp_coords,(torch.tensor([interpolation_input_data["id"]]), torch.tensor([2])), RV_mu_post2["mu_"+str(counter2)+"_post"])
        counter2=counter2+1


    # interpolation_input = geo_model_test.interpolation_input
    # interpolation_input.surface_points.sp_coords = torch.index_put(
    #     interpolation_input.surface_points.sp_coords,
    #     (torch.tensor([11]), torch.tensor([2])),
    #     mu_1_post
    # )
    # interpolation_input.surface_points.sp_coords = torch.index_put(
    #     interpolation_input.surface_points.sp_coords,
    #     (torch.tensor([14]), torch.tensor([2])),
    #     mu_2_post
    # )

    # interpolation_input.surface_points.sp_coords = torch.index_put(
    #         interpolation_input.surface_points.sp_coords,
    #         (torch.tensor([5]), torch.tensor([2])),
    #         mu_3_post
    #     )
    # interpolation_input.surface_points.sp_coords = torch.index_put(
    #         interpolation_input.surface_points.sp_coords,
    #         (torch.tensor([0]), torch.tensor([2])),
    #         mu_4_post
    #     )
        
    #print("interpolation_input",interpolation_input.surface_points.sp_coords)

    # # Compute the geological model
    geo_model_test.solutions = gempy_engine.compute_model(
        interpolation_input=interpolation_input,
        options=geo_model_test.interpolation_options,
        data_descriptor=geo_model_test.input_data_descriptor,
        geophysics_input=geo_model_test.geophysics_input,
    )
    
    sp_coords_copy_test2 =interpolation_input.surface_points.sp_coords
    sp_cord= geo_model_test.transform.apply_inverse(sp_coords_copy_test2.detach().numpy())
    
    ################################################################################
    # Store the Initial Interface data and orientation data
    ################################################################################
    df_sp_final = pd.DataFrame(sp_cord, columns=["X","Y","Z"])
    df_sp_final.to_csv("./Results/Final_sp.csv")
    ################################################################################
    
    geo_model_test_post = gp.create_geomodel(
    project_name='Gempy_abc_Test_post',
    extent=[0, 86, -10, 10, -83, 0],
    resolution=[86,20,83],
    refinement=7,
    structural_frame= gp.data.StructuralFrame.initialize_default_structure()
    )

    gp.add_surface_points(
        geo_model=geo_model_test_post,
        x=[70.0, 80.0],
        y=[0.0, 0.0],
        z=[-77.0, -71.0],
        elements_names=['surface1', 'surface1']
    )

    gp.add_orientations(
        geo_model=geo_model_test_post,
        x=[75],
        y=[0.0],
        z=[-74],
        elements_names=['surface1'],
        pole_vector=[[-5/3, 0, 1]]
    )
    geo_model_test_post.update_transform(gp.data.GlobalAnisotropy.NONE)

    element2 = gp.data.StructuralElement(
        name='surface2',
        color=next(geo_model_test_post.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([20.0, 60.0]),
            y=np.array([0.0, 0.0]),
            z=np.array([sp_cord[0,2], -52]),
            names='surface2'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    geo_model_test_post.structural_frame.structural_groups[0].append_element(element2)

    element3 = gp.data.StructuralElement(
        name='surface3',
        color=next(geo_model_test_post.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([0.0, 30.0, 60]),
            y=np.array([0.0, 0.0,0.0]),
            z=np.array([-72, -55.5, -39]),
            names='surface3'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    geo_model_test_post.structural_frame.structural_groups[0].append_element(element3)

    element4 = gp.data.StructuralElement(
        name='surface4',
        color=next(geo_model_test_post.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([0.0, 20.0, 60]),
            y=np.array([0.0, 0.0,0.0]),
            z=np.array([-61, sp_cord[5,2], -27]),
            names='surface4'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    geo_model_test_post.structural_frame.structural_groups[0].append_element(element4)

    element5 = gp.data.StructuralElement(
        name='surface5',
        color=next(geo_model_test_post.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([0.0, 20, 40]),
            y=np.array([0.0, 0.0, 0.0]),
            z=np.array([-39, sp_cord[14,2], -16]),
            names='surface5'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    geo_model_test_post.structural_frame.structural_groups[0].append_element(element5)

    element6 = gp.data.StructuralElement(
        name='surface6',
        color=next(geo_model_test_post.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([0.0, 20.0,30]),
            y=np.array([0.0, 0.0, 0.0]),
            z=np.array([-21, sp_cord[11,2], -1]),
            names='surface6'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    geo_model_test_post.structural_frame.structural_groups[0].append_element(element6)

    geo_model_test_post.structural_frame.structural_groups[0].elements[0], geo_model_test_post.structural_frame.structural_groups[0].elements[1],\
    geo_model_test_post.structural_frame.structural_groups[0].elements[2], geo_model_test_post.structural_frame.structural_groups[0].elements[3],\
    geo_model_test_post.structural_frame.structural_groups[0].elements[4], geo_model_test_post.structural_frame.structural_groups[0].elements[5] = \
    geo_model_test_post.structural_frame.structural_groups[0].elements[1], geo_model_test_post.structural_frame.structural_groups[0].elements[0],\
    geo_model_test_post.structural_frame.structural_groups[0].elements[3], geo_model_test_post.structural_frame.structural_groups[0].elements[2],\
    geo_model_test_post.structural_frame.structural_groups[0].elements[5], geo_model_test_post.structural_frame.structural_groups[0].elements[4]  


    gp.set_custom_grid(geo_model_test_post.grid, xyz_coord=xyz_coord)
    gp.compute_model(geo_model_test_post)
    
    custom_grid_values_post = geo_model_test_post.solutions.octrees_output[0].last_output_center.custom_grid_values
    ####################################TODO#################################################
    #   Try to find the final accuracy to check if it has improved the classification
    #########################################################################################
    accuracy_final = torch.sum(torch.round(torch.tensor(custom_grid_values_post)) == y_obs_label) / y_obs_label.shape[0]
    print("accuracy_init: ", accuracy_init , "accuracy_final: ", accuracy_final)
    
    picture_test_post = gpv.plot_2d(geo_model_test_post, cell_number=5, legend='force')
    plt.savefig("./Results/Posterior_model.png")
    
    #TSNE_transformation(data=normalised_data, label=custom_grid_values_post, filename="./Results/tsne_gempy_final_label.png")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pass values using command line')
    parser.add_argument('--startval', metavar='startcol', required=True, type=int, help='start x column value')
    parser.add_argument('--endval', metavar='endcol', required=True, type=int, help='end x column value')
    parser.add_argument('--dimred', metavar='dimred', type=str, help='type of dimensionality reduction')
    parser.add_argument('--numlayers', metavar='numlayers', type=int, help='numbers of layers')
    args = parser.parse_args()
    main(startval=args.startval,endval=args.endval,dimred=args.dimred,numlayers=args.numlayers)
