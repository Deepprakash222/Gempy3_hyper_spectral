import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import arviz as az
import pandas as pd
from datetime import datetime
import json
import argparse

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


parser = argparse.ArgumentParser(description='pass values using command line')
parser.add_argument('--startval', metavar='startcol', type=int, default=19,  help='start x column value')
parser.add_argument('--endval', metavar='endcol', type=int, default=21, help='end x column value')
parser.add_argument('--cluster', metavar='cluster', type=int, default=4, help='total number of cluster')
parser.add_argument('--dimred', metavar='dimred', type=str , default="pca", help='type of dimensionality reduction')
parser.add_argument('--plot_dimred', metavar='plot_dimred', type=str , default="tsne", help='type of dimensionality reduction for plotting after data is alread reduced in a smaller dimension')
parser.add_argument('--prior_number_samples', metavar='prior_number_samples', type=int , default=100, help='number of samples for prior')
parser.add_argument('--posterior_number_samples', metavar='posterior_number_samples', type=int , default=150, help='number of samples for posterior')
parser.add_argument('--posterior_warmup_steps', metavar='posterior_warmup_steps', type=int , default=50, help='number of  warmup steps for posterior')
parser.add_argument('--directory_path', metavar='directory_path', type=str , default="./Results_without_prior_gmm_KSL", help='name of the directory in which result should be stored')

def cluster_acc(Y_pred, Y, ignore_label=None):
    """ Rearranging the class labels of prediction so that it maximise the 
        match class labels.

    Args:
        Y_pred (int): An array for predicted labels
        Y (float): An array for true labels
        ignore_label (int, optional): Laels to be ignored

    Returns:
       row (int): A list of index of row 
       column (int) : A list of index of column
       accuracy (float): accuracy after we found correct label
       cost_matrix (int) : cost matrix 
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
    """ This function applies TSNE algorithms to reduce the high dimensional data into 2D
        for better visualization

    Args:
        data (float): High dimensional Input data 
        label (int): Label information of each data entry
        filename (str): Location to store the image after dimensionality reduction
    """
    from sklearn.manifold import TSNE
    model = TSNE(n_components=2, random_state=42)
    transformed_data = model.fit_transform(data) 
    label_to_color = { 1: 'red', 2: 'blue', 3: 'green', 4: 'yellow', 5: 'orange', 6: 'purple'}
    plt.figure(figsize=(10,8))
    for label_ in np.unique(label):
        idx =label ==label_
        plt.scatter(transformed_data[idx][:,0],transformed_data[idx][:,1], c=label_to_color[label_],label=f' {label_}',s=50, marker='o',alpha=1.0, edgecolors='w')
    # Create a legend
    plt.legend()
    # Add axis labels
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title("Data after dimensionality reduction")
    
    plt.savefig(filename)
    
def create_initial_gempy_model(refinement,filename, save=True):
    """ Create an initial gempy model objet

    Args:
        refinement (int): Refinement of grid
        save (bool, optional): Whether you want to save the image

    """
    geo_model_test = gp.create_geomodel(
    project_name='Gempy_abc_Test',
    extent=[0, 1000, -10, 10, -900, -700],
    resolution=[100,10,100],
    refinement=refinement,
    structural_frame= gp.data.StructuralFrame.initialize_default_structure()
    )
    brk1 = -855
    brk2 = -845 
    brk3 = -825 
    # brk1 = -847
    # brk2 = -824
    # brk3 = -793
    
    
    gp.add_surface_points(
        geo_model=geo_model_test,
        x=[100.0,300, 900.0],
        y=[0.0, 0.0, 0.0],
        z=[brk1, brk1, brk1],
        elements_names=['surface1','surface1', 'surface1']
    )

    gp.add_orientations(
        geo_model=geo_model_test,
        x=[500],
        y=[0.0],
        z=[brk1],
        elements_names=['surface1'],
        pole_vector=[[0, 0, 1]]
    )
    geo_model_test.update_transform(gp.data.GlobalAnisotropy.NONE)

    element2 = gp.data.StructuralElement(
        name='surface2',
        color=next(geo_model_test.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([100.0, 300.0,900.0]),
            y=np.array([0.0, 0.0, 0.0]),
            z=np.array([brk2, brk2, brk2]),
            names='surface2'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    geo_model_test.structural_frame.structural_groups[0].append_element(element2)
    
    element3 = gp.data.StructuralElement(
        name='surface3',
        color=next(geo_model_test.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([100.0, 300.0,900.0]),
            y=np.array([0.0, 0.0, 0.0]),
            z=np.array([brk3, brk3, brk3]),
            names='surface3'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    geo_model_test.structural_frame.structural_groups[0].append_element(element3)

    geo_model_test.structural_frame.structural_groups[0].elements[0], geo_model_test.structural_frame.structural_groups[0].elements[1] = \
    geo_model_test.structural_frame.structural_groups[0].elements[1], geo_model_test.structural_frame.structural_groups[0].elements[0]

    geo_model_test.structural_frame.structural_groups[0].elements[1], geo_model_test.structural_frame.structural_groups[0].elements[2] = \
    geo_model_test.structural_frame.structural_groups[0].elements[2], geo_model_test.structural_frame.structural_groups[0].elements[1]
    
    geo_model_test.structural_frame.structural_groups[0].elements[0], geo_model_test.structural_frame.structural_groups[0].elements[1] = \
    geo_model_test.structural_frame.structural_groups[0].elements[1], geo_model_test.structural_frame.structural_groups[0].elements[0]
    
    # gp.add_surface_points(
    # geo_model=geo_model_test,
    # x=[300, 300],
    # y=[0.0, 0.0],
    # z=[brk1, brk2],
    # elements_names=['surface1', 'surface2']
    # )
    

    gp.compute_model(geo_model_test)
    picture_test = gpv.plot_2d(geo_model_test, cell_number=5, legend='force')
    if save:
        plt.savefig(filename)
    
    return geo_model_test

def main():
    """
    This function defines a model which uses hyperspectral data, applies clustering methods to find cluster information and then uses Bayesian
    """
    args = parser.parse_args()
    startval=args.startval
    endval=args.endval
    cluster = args.cluster
    dimred=args.dimred
    plot_dimred=args.plot_dimred
    prior_number_samples = args.prior_number_samples
    posterior_number_samples = args.posterior_number_samples
    posterior_warmup_steps = args.posterior_warmup_steps
    directory_path = args.directory_path
    # Check if the directory exists
    if not os.path.exists(directory_path):
        # Create the directory if it does not exist
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' was created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
        
    # Load KSL_file file
    import joblib
    filename_a = './Fw__Hyperspectral_datasets_from_the_KSL_cores/CuSp131.pkl'
    with open(filename_a, 'rb') as myfile:
        a =joblib.load(myfile)
    column_name =[]
    
    for keys, _ in a.items():
        if keys=='XYZ':
            column_name.append("Borehole_id")
            column_name.append("X")
            column_name.append("Y")
            column_name.append("Z")
        else:
            column_name.append(keys+"_R")
            column_name.append(keys+"_G")
            column_name.append(keys+"_B")
    data_a =[]
    for keys, values in a.items():
        if keys=='XYZ':
            label_a = np.ones((235,1))
            data_a.append(label_a)
            data_a.append(values)
        else:
            data_a.append(values)

    # Concatenate the arrays horizontally to create an array of size 5x30
    concatenated_array_a = np.hstack(data_a)
    # sort the data based on the depth
    sorted_indices = np.argsort(-concatenated_array_a[:, 3])
    concatenated_array_a = concatenated_array_a[sorted_indices]
    concatenated_array_a.shape
    
    import pandas as pd
    dataframe_KSL = pd.DataFrame(concatenated_array_a,columns=column_name)
    ######################################################################
    ## Arrange Data as concatationation of spacial co-ordinate and pixel values
    ###########################################################################
    #print(dataframe_KSL.head())
    dataframe_KSL = dataframe_KSL[(dataframe_KSL["Z"]<=-700)]
    # column_list =['Z', 'BR_Anhydrite_R',
    #                 'BR_Anhydrite_G',
    #                 'BR_Anhydrite_B',
    #                 'BR_Qtz_Fsp_Cal_R',
    #                 'BR_Qtz_Fsp_Cal_G',
    #                 'BR_Qtz_Fsp_Cal_B'
    #                 ]
    df_spectral_normalised = dataframe_KSL.copy()
    df_spectral_normalised.iloc[:,4:] =df_spectral_normalised.iloc[:,4:].apply(zscore,axis=1)
    
    # data_hsi = dataframe_KSL.iloc[:,3:]
    # data_hsi= data_hsi[column_list]
    # data_hsi["Z"] =data_hsi["Z"]/1000.0
    data_hsi = df_spectral_normalised.iloc[:,4:]
    
    # Normalise along the spectral lines 
    df_with_spectral_normalised = data_hsi.copy()
    
    
    ###########################################################################
    ## Obtain the preprocessed data
    ###########################################################################
    normalised_hsi =torch.tensor(df_with_spectral_normalised.to_numpy(), dtype=torch.float64)

    ## It is difficult to work with data in such a high dimensions, because the covariance matrix 
    ## determinant quickly goes to zero even if eigen-values are in the range of 1e-3. Therefore it is advisable 
    ## to fist apply dimensionality reduction to a lower dimensions
    # if dimred=="pca":
    #     from sklearn.decomposition import PCA
    #     pca = PCA(n_components=10)
    #     transformed_hsi = pca.fit_transform(normalised_hsi)
    #     normalised_hsi = torch.tensor(transformed_hsi, dtype=torch.float64)
    # if dimred =="tsne":
    #     #######################TODO#####################
    #     ################################################
    #     print("TSNE hasn't implemented for dimensionality reduction yet")
    #     exit()
        
    ###########################################################################
    ## Apply Classical clustering methods to find different cluster information our data
    ###########################################################################
    
    # Create initial model with higher refinement for better resolution and save it
    prior_filename= directory_path + "/prior_model.png"
    geo_model_test = create_initial_gempy_model(refinement=7,filename=prior_filename, save=True)
    # We can initialize again but with lower refinement because gempy solution are inddependent
    geo_model_test = create_initial_gempy_model(refinement=3,filename=prior_filename, save=False)
    
    ################################################################################
    # Custom grid
    ################################################################################
    x_loc = 300
    y_loc = 0
    z_loc = dataframe_KSL.iloc[:,3].to_numpy()
    xyz_coord = np.array([[x_loc, y_loc, z] for z in z_loc])
    # xyz_coord = dataframe_KSL.iloc[:,:3].to_numpy()
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
    
    filename_initial_sp = directory_path + "/Initial_sp.csv"
    filename_initial_op = directory_path + "/Initial_op.csv"
    df_sp_init.to_csv(filename_initial_sp)
    df_or_init.to_csv(filename_initial_op)
    ################################################################################
    y_obs_label = geo_model_test.solutions.octrees_output[0].last_output_center.custom_grid_values
    
    gm = BayesianGaussianMixture(n_components=cluster, covariance_type='full', random_state=42 ).fit(normalised_hsi)
    
    # make the labels to start with 1 instead of 0
    gmm_label = gm.predict(normalised_hsi) +1 
    #print(gmm_label)
    gmm_label_order, y_obs_label_order, accuracy_init, _ = cluster_acc( gmm_label, np.round(y_obs_label))
    
    
    # # reaarange the label information so it is would be consistent with ground truth label
    gmm_label_rearranged = torch.tensor([y_obs_label_order[x-1] +1  for x in gmm_label], dtype=torch.float64)
    

    #print(gmm_label_rearranged)
   
    # rearrange the mean and covariance accordingly too
    #rearrange_list = [1,2,0]
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
    if plot_dimred =="tsne":
        filename_tsne = directory_path + "/tsne_gmm_label.png"
        TSNE_transformation(data=normalised_hsi, label=gmm_label_rearranged, filename=filename_tsne)
    
    
    #geo_model_test.transform.apply_inverse(sp_coords_copy_test)
    
    # Change the backend to PyTorch for probabilistic modeling
    BackendTensor.change_backend_gempy(engine_backend=gp.data.AvailableBackends.PYTORCH)
    
    # geo_model_test.interpolation_options.uni_degree = 0
    # geo_model_test.interpolation_options.mesh_extraction = False
    geo_model_test.interpolation_options.sigmoid_slope = 40
    store_accuracy=[]
    
    @config_enumerate
    def model_test(obs_data):
        """
        This Pyro model represents the probabilistic aspects of the geological model.
        It defines a prior distribution for the top layer's location and
        computes the thickness of the geological layer as an observed variable.
        """
    
        # Define prior for the top layer's location
        prior_mean_surface_1 = sp_coords_copy_test[1, 2]
        prior_mean_surface_2 = sp_coords_copy_test[4, 2]
        prior_mean_surface_3 = sp_coords_copy_test[7, 2]
        
        mu_surface_1 = pyro.sample('mu_1', dist.Normal(prior_mean_surface_1, torch.tensor(0.2, dtype=torch.float64)))
        mu_surface_2 = pyro.sample('mu_2', dist.Normal(prior_mean_surface_2, torch.tensor(0.2, dtype=torch.float64)))
        mu_surface_3 = pyro.sample('mu_3', dist.Normal(prior_mean_surface_3, torch.tensor(0.2, dtype=torch.float64)))
        
        pyro.sample('mu_1 > mu_2', dist.Delta(torch.tensor(1.0, dtype=torch.float64)), obs=(mu_surface_1 > mu_surface_2))
        pyro.sample('mu_2 > mu_3', dist.Delta(torch.tensor(1.0, dtype=torch.float64)), obs=(mu_surface_2 > mu_surface_3))
        #pyro.sample('mu_3 > mu_3', dist.Delta(torch.tensor(1.0, dtype=torch.float64)), obs=(mu_surface_2 > mu_surface_3))
        # Update the model with the new top layer's location
        interpolation_input = geo_model_test.interpolation_input
        
        
        interpolation_input.surface_points.sp_coords = torch.index_put(
            interpolation_input.surface_points.sp_coords,
            (torch.tensor([1]), torch.tensor([2])),
            mu_surface_1
        )
        interpolation_input.surface_points.sp_coords = torch.index_put(
            interpolation_input.surface_points.sp_coords,
            (torch.tensor([4]), torch.tensor([2])),
            mu_surface_2
        )
        interpolation_input.surface_points.sp_coords = torch.index_put(
            interpolation_input.surface_points.sp_coords,
            (torch.tensor([7]), torch.tensor([2])),
            mu_surface_3
        )
        
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
        # accuracy_intermediate = torch.sum(torch.round(custom_grid_values) == y_obs_label) / y_obs_label.shape[0]
        # store_accuracy.append(accuracy_intermediate)
        lambda_ = 20.0
        # loc_mean = torch.tensor(mean_init,dtype=torch.float64)
        # loc_cov =  torch.tensor(cov_init, dtype=torch.float64)
        z_nk = F.softmax(-lambda_* (torch.linspace(1,cluster,cluster, dtype=torch.float64) - custom_grid_values.reshape(-1,1))**2, dim=1)
        #class_label = torch.mean(F.softmax(-lambda_* (torch.tensor([1,2,3,4,5,6], dtype=torch.float64) - custom_grid_values.reshape(-1,1))**2, dim=1),dim=0)
        
        N_k = torch.sum(z_nk,axis=0)
        N = len(custom_grid_values)
        pi_k = N_k /N
        mean = []
        cov = []
        for i in range(z_nk.shape[1]):
            mean_k = torch.sum( z_nk[:,i][:,None] * obs_data, axis=0)/ N_k[i]
            #cov_k = torch.sum( (normalised_hsi - mean_k.reshape((-1,1))) (normalised_hsi - mean_k).T )
            cov_k = torch.zeros((mean_k.shape[0],mean_k.shape[0]),dtype=torch.float64)
            for j in range(z_nk.shape[0]):
                 cov_k +=  z_nk[j,i]* torch.matmul((obs_data[j,:] - mean_k).reshape((-1,1)) ,(obs_data[j,:] - mean_k).reshape((1,-1)))
            mean.append(mean_k)
            cov_k=cov_k/N_k[i] + 1e-8 * torch.diag(torch.ones(cov_k.shape[0],dtype=torch.float64))
            cov.append(cov_k)
        mean_tensor = torch.stack(mean, dim=0)
        cov_tensor = torch.stack(cov,dim=0)
        
        #cov_likelihood = 5.0 * torch.eye(loc_cov[0].shape[0], dtype=torch.float64)
        
        with pyro.plate('N='+str(obs_data.shape[0]), obs_data.shape[0]):
            assignment = pyro.sample("assignment", dist.Categorical(pi_k))
            #print(obs_data.shape, mean_tensor[assignment].shape,cov_tensor[assignment].shape)
            obs = pyro.sample("obs", dist.MultivariateNormal(loc=mean_tensor[assignment],covariance_matrix = cov_tensor[assignment]), obs=obs_data)
    filename_bayes_graph = directory_path + "/Bayesian_graph.png"
    dot = pyro.render_model(model_test, model_args=(normalised_hsi,),render_distributions=True,filename=filename_bayes_graph)
    
    ################################################################################
    # Prior
    ################################################################################
    pyro.set_rng_seed(42)
    prior = Predictive(model_test, num_samples=prior_number_samples)(normalised_hsi)
    # Key to avoid
    #avoid_key = ['mu_1 < 0','mu_1 > mu_2','mu_2 > -38.5', 'mu_3 < -38.5','mu_3 > -61.4','mu_4 < -61.5', 'mu_4 > -83']
    avoid_key = ['mu_1 < 0','mu_1 > mu_2','mu_2 > mu_3', 'mu_3 > mu_4' , 'mu_4 > -83']
    # Create sub-dictionary without the avoid_key
    prior = dict((key, value) for key, value in prior.items() if key not in avoid_key)
    plt.figure(figsize=(8,10))
    data = az.from_pyro(prior=prior)
    az.plot_trace(data.prior)
    filename_prior_plot = directory_path + "/prior.png"
    plt.savefig(filename_prior_plot)
    
    ################################################################################
    # Posterior 
    ################################################################################
    pyro.primitives.enable_validation(is_validate=True)
    nuts_kernel = NUTS(model_test, step_size=0.0085, adapt_step_size=True, target_accept_prob=0.9, max_tree_depth=10, init_strategy=init_to_mean)
    mcmc = MCMC(nuts_kernel, num_samples=posterior_number_samples, warmup_steps=posterior_warmup_steps, disable_validation=False)
    mcmc.run(normalised_hsi)
    
    posterior_samples = mcmc.get_samples()
    posterior_predictive = Predictive(model_test, posterior_samples)(normalised_hsi)
    plt.figure(figsize=(8,10))
    data = az.from_pyro(posterior=mcmc, prior=prior, posterior_predictive=posterior_predictive)
    az.plot_trace(data)
    filename_posteriro_plot = directory_path + "/posterior.png"
    plt.savefig(filename_posteriro_plot)
    
    ###############################################TODO################################
    # Plot and save the file for each parameter
    ###################################################################################
    plt.figure(figsize=(8,10))
    az.plot_density(
        data=[data.posterior, data.prior],
        shade=.9,
        var_names=['mu_1'],
        data_labels=["Posterior Predictive", "Prior Predictive"],
        colors=[default_red, default_blue],
    )
    filename_mu_1 = directory_path + "/mu_1.png"
    plt.savefig(filename_mu_1)
    
    plt.figure(figsize=(8,10))
    az.plot_density(
        data=[data.posterior, data.prior],
        shade=.9,
        var_names=['mu_2'],
        data_labels=["Posterior Predictive", "Prior Predictive"],
        colors=[default_red, default_blue],
    )
    filename_mu_2 = directory_path + "/mu_2.png"
    plt.savefig(filename_mu_2)
    
    plt.figure(figsize=(8,10))
    az.plot_density(
        data=[data.posterior, data.prior],
        shade=.9,
        var_names=['mu_3'],
        data_labels=["Posterior Predictive", "Prior Predictive"],
        colors=[default_red, default_blue],
    )
    filename_mu_3 = directory_path + "/mu_3.png"
    plt.savefig(filename_mu_3)
    ###############################################TODO################################
    # Find the MAP value
    ###################################################################################
    
    unnormalise_posterior_value={}
    unnormalise_posterior_value["log_prior_geo_list"]=[]
    unnormalise_posterior_value["log_likelihood_list"]=[]
    unnormalise_posterior_value["log_posterior_list"]=[]
    # log_prior_geo_list=[]
    # log_prior_hsi_list=[]
    # log_likelihood_list=[]
    # log_posterior_list=[]
    keys_list = list(posterior_samples.keys())
   
    prior_mean_surface_1 = sp_coords_copy_test[1, 2]
    prior_mean_surface_2 = sp_coords_copy_test[4, 2]
    prior_mean_surface_3 = sp_coords_copy_test[7, 2]
   
    
    for i in range(posterior_samples["mu_1"].shape[0]):
        post_mu_1 = posterior_samples[keys_list[0]][i]
        post_mu_2 = posterior_samples[keys_list[1]][i]
        post_mu_3 = posterior_samples[keys_list[2]][i]
        
        # Calculate the log probability of the value
        
        log_prior_geo = dist.Normal(prior_mean_surface_1, torch.tensor(0.2, dtype=torch.float64)).log_prob(post_mu_1)+\
                    dist.Normal(prior_mean_surface_2, torch.tensor(0.2, dtype=torch.float64)).log_prob(post_mu_2)+\
                    dist.Normal(prior_mean_surface_3, torch.tensor(0.2, dtype=torch.float64)).log_prob(post_mu_3)
        # Update the model with the new top layer's location
        interpolation_input = geo_model_test.interpolation_input
        
        
        interpolation_input.surface_points.sp_coords = torch.index_put(
            interpolation_input.surface_points.sp_coords,
            (torch.tensor([1]), torch.tensor([2])),
            post_mu_1
        )
        interpolation_input.surface_points.sp_coords = torch.index_put(
            interpolation_input.surface_points.sp_coords,
            (torch.tensor([4]), torch.tensor([2])),
            post_mu_2
        )
        
        interpolation_input.surface_points.sp_coords = torch.index_put(
            interpolation_input.surface_points.sp_coords,
            (torch.tensor([7]), torch.tensor([2])),
            post_mu_3
        )
        
        
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
        # accuracy_intermediate = torch.sum(torch.round(custom_grid_values) == y_obs_label) / y_obs_label.shape[0]
        # #print("accuracy_intermediate", accuracy_intermediate)
        # store_accuracy.append(accuracy_intermediate)
        lambda_ = 20.0
        # loc_mean = torch.tensor(mean_init,dtype=torch.float64)
        # loc_cov =  torch.tensor(cov_init, dtype=torch.float64)
        z_nk = F.softmax(-lambda_* (torch.linspace(1,cluster,cluster, dtype=torch.float64) - custom_grid_values.reshape(-1,1))**2, dim=1)
        #class_label = torch.mean(F.softmax(-lambda_* (torch.tensor([1,2,3,4,5,6], dtype=torch.float64) - custom_grid_values.reshape(-1,1))**2, dim=1),dim=0)
        
        N_k = torch.sum(z_nk,axis=0)
        N = len(custom_grid_values)
        pi_k = N_k /N
        mean = []
        cov = []
        for i in range(z_nk.shape[1]):
            mean_k = torch.sum( z_nk[:,i][:,None] * normalised_hsi, axis=0)/ N_k[i]
            #cov_k = torch.sum( (normalised_hsi - mean_k.reshape((-1,1))) (normalised_hsi - mean_k).T )
            cov_k = torch.zeros((mean_k.shape[0],mean_k.shape[0]), dtype=torch.float64)
            for j in range(z_nk.shape[0]):
                 cov_k +=  z_nk[j,i]* torch.matmul((normalised_hsi[j,:] - mean_k).reshape((-1,1)) ,(normalised_hsi[j,:] - mean_k).reshape((1,-1)))
            mean.append(mean_k)
            cov_k=cov_k/N_k[i] + 1e-8 * torch.diag(torch.ones(cov_k.shape[0],dtype=torch.float64))
            cov.append(cov_k)
        mean_tensor = torch.stack(mean, dim=0)
        cov_tensor = torch.stack(cov,dim=0)
        
        # We can also calculate the accuracy using the mean and covariance to see if our GMM model has imroved or not
        gamma_nk = torch.zeros(z_nk.shape)

        log_likelihood=torch.tensor(0.0, dtype=torch.float64)

        for j in range(normalised_hsi.shape[0]):
            likelihood = pi_k[0] *torch.exp(dist.MultivariateNormal(loc=mean_tensor[0],covariance_matrix= cov_tensor[0]).log_prob(normalised_hsi[j])) +\
                         pi_k[1] *torch.exp(dist.MultivariateNormal(loc=mean_tensor[1],covariance_matrix= cov_tensor[1]).log_prob(normalised_hsi[j]))+\
                         pi_k[2] *torch.exp(dist.MultivariateNormal(loc=mean_tensor[2],covariance_matrix= cov_tensor[2]).log_prob(normalised_hsi[j])) 
                        
            for k in range(gamma_nk.shape[1]):
                gamma_nk[j][k] = (pi_k[k] * torch.exp(dist.MultivariateNormal(loc=mean_tensor[k],covariance_matrix= cov_tensor[k]).log_prob(normalised_hsi[j]))) / likelihood
                
            log_likelihood += torch.log(likelihood)
        
        
        #print("gmm_label_accuracy", gmm_accuracy)
        # log_prior_geo_list.append(log_prior_geo)
        # log_prior_hsi_list.append(log_prior_hsi)
        # log_likelihood_list.append(log_likelihood)
        # log_posterior_list.append(log_prior_geo + log_prior_hsi + log_likelihood)
        unnormalise_posterior_value["log_prior_geo_list"].append(log_prior_geo)
        unnormalise_posterior_value["log_likelihood_list"].append(log_likelihood)
        unnormalise_posterior_value["log_posterior_list"].append(log_prior_geo + log_likelihood)
    
    MAP_sample_index=torch.argmax(torch.tensor(unnormalise_posterior_value["log_posterior_list"]))
    
    
    #print(unnormalise_posterior_value)
    
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
    # mu_1_mean = posterior_samples["mu_1"].mean()
    # mu_2_mean = posterior_samples["mu_2"].mean()
    # mu_3_mean = posterior_samples["mu_3"].mean()
    # mu_4_mean = posterior_samples["mu_4"].mean()
    
    mu_1_post = posterior_samples["mu_1"][MAP_sample_index]
    mu_2_post = posterior_samples["mu_2"][MAP_sample_index]
    mu_3_post = posterior_samples["mu_3"][MAP_sample_index]
    # # Update the model with the new top layer's location
    interpolation_input = geo_model_test.interpolation_input
    interpolation_input.surface_points.sp_coords = torch.index_put(
        interpolation_input.surface_points.sp_coords,
        (torch.tensor([1]), torch.tensor([2])),
        mu_1_post
    )
    interpolation_input.surface_points.sp_coords = torch.index_put(
        interpolation_input.surface_points.sp_coords,
        (torch.tensor([4]), torch.tensor([2])),
        mu_2_post
    )
    
    interpolation_input.surface_points.sp_coords = torch.index_put(
        interpolation_input.surface_points.sp_coords,
        (torch.tensor([7]), torch.tensor([2])),
        mu_3_post
    )
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
    df_sp_final = df_sp_init.copy()
    df_sp_final["Z"] = sp_cord[:,2] 
    filename_final_sp = directory_path + "/Final_sp.csv"
    df_sp_final.to_csv(filename_final_sp)
    ################################################################################
        
    geo_model_test_post = gp.create_geomodel(
    project_name='Gempy_abc_Test_post',
    extent=[0, 1000, -10, 10, -900, -700],
    resolution=[100,10,100],
    refinement=7,
    structural_frame= gp.data.StructuralFrame.initialize_default_structure()
    )
    
    brk1 = -855
    brk2 = -845 
    brk3 = -825 
    # brk1 = -847
    # brk2 = -824
    # brk3 = -793

    gp.add_surface_points(
        geo_model=geo_model_test_post,
        x=[100.0,300, 900.0],
        y=[0.0,0.0, 0.0],
        z=[brk1,sp_cord[7,2], brk1],
        elements_names=['surface1','surface1', 'surface1']
    )

    gp.add_orientations(
        geo_model=geo_model_test_post,
        x=[500],
        y=[0.0],
        z=[brk1],
        elements_names=['surface1'],
        pole_vector=[[0, 0, 1]]
    )
    geo_model_test_post.update_transform(gp.data.GlobalAnisotropy.NONE)

    element2 = gp.data.StructuralElement(
        name='surface2',
        color=next(geo_model_test_post.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([100.0,300, 900.0]),
            y=np.array([0.0,0.0, 0.0]),
            z=np.array([brk2, sp_cord[4,2], brk2]),
            names='surface2'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    geo_model_test_post.structural_frame.structural_groups[0].append_element(element2)

    element3 = gp.data.StructuralElement(
        name='surface3',
        color=next(geo_model_test_post.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([100.0,300, 900.0]),
            y=np.array([0.0,0.0, 0.0]),
            z=np.array([brk3, sp_cord[1,2], brk3]),
            names='surface3'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    geo_model_test_post.structural_frame.structural_groups[0].append_element(element3)
    
    geo_model_test_post.structural_frame.structural_groups[0].elements[0], geo_model_test_post.structural_frame.structural_groups[0].elements[1] = \
    geo_model_test_post.structural_frame.structural_groups[0].elements[1], geo_model_test_post.structural_frame.structural_groups[0].elements[0]

    geo_model_test_post.structural_frame.structural_groups[0].elements[1], geo_model_test_post.structural_frame.structural_groups[0].elements[2] = \
    geo_model_test_post.structural_frame.structural_groups[0].elements[2], geo_model_test_post.structural_frame.structural_groups[0].elements[1]
    
    geo_model_test_post.structural_frame.structural_groups[0].elements[0], geo_model_test_post.structural_frame.structural_groups[0].elements[1] = \
    geo_model_test_post.structural_frame.structural_groups[0].elements[1], geo_model_test_post.structural_frame.structural_groups[0].elements[0]
    # gp.add_surface_points(
    # geo_model=geo_model_test_post,
    # x=[300, 300],
    # y=[0.0, 0.0],
    # z=[sp_cord[2,2], sp_cord[5,2]],
    # elements_names=['surface1', 'surface2']
    # )
    

    gp.set_custom_grid(geo_model_test_post.grid, xyz_coord=xyz_coord)
    gp.compute_model(geo_model_test_post)
    
    custom_grid_values_post = geo_model_test_post.solutions.octrees_output[0].last_output_center.custom_grid_values
    
    ####################################TODO#################################################
    #   Try to find the final accuracy to check if it has improved the classification
    #########################################################################################

    
    ###################################TODO###################################################
    # store the weights, mean, and covariance of Gaussian mixture model
    ##########################################################################################
    lambda_ = 20.0
    # loc_mean = torch.tensor(mean_init,dtype=torch.float64)
    # loc_cov =  torch.tensor(cov_init, dtype=torch.float64)
    z_nk = F.softmax(-lambda_* (torch.linspace(1,cluster,cluster, dtype=torch.float64) - torch.tensor(custom_grid_values_post).reshape(-1,1))**2, dim=1)
    #class_label = torch.mean(F.softmax(-lambda_* (torch.tensor([1,2,3,4,5,6], dtype=torch.float64) - custom_grid_values.reshape(-1,1))**2, dim=1),dim=0)
    
    N_k = torch.sum(z_nk,axis=0)
    N = len(custom_grid_values)
    pi_k = N_k /N
    mean = []
    cov = []
    for i in range(z_nk.shape[1]):
        mean_k = torch.sum( z_nk[:,i][:,None] * normalised_hsi, axis=0)/ N_k[i]
        #cov_k = torch.sum( (normalised_hsi - mean_k.reshape((-1,1))) (normalised_hsi - mean_k).T )
        cov_k = torch.zeros((mean_k.shape[0],mean_k.shape[0]), dtype=torch.float64)
        for j in range(z_nk.shape[0]):
                cov_k +=  z_nk[j,i]* torch.matmul((normalised_hsi[j,:] - mean_k).reshape((-1,1)) ,(normalised_hsi[j,:] - mean_k).reshape((1,-1)))
        mean.append(mean_k.detach().numpy())
        cov_k=cov_k/N_k[i] #+ 1e-6 * torch.diag(torch.ones(cov_k.shape[0],dtype=torch.float64))
        cov.append(cov_k.detach().numpy())
    mean_tensor = np.stack(mean, axis=0)
    cov_tensor = np.stack(cov,axis=0)
    
    gmm_data ={}
    gmm_data["weights"]=pi_k.detach().numpy().tolist()
    gmm_data["means"] = mean_tensor.tolist()
    gmm_data["covariances"] = cov_tensor.tolist()
    # Save to file
    filename_gmm_data = directory_path + "/gmm_data.json"
    with open(filename_gmm_data, "w") as json_file:
        json.dump(gmm_data, json_file)
    ##########################################################################################
    
    picture_test_post = gpv.plot_2d(geo_model_test_post, cell_number=5, legend='force')
    filename_posterior_model = directory_path + "/Posterior_model.png"
    plt.savefig(filename_posterior_model)
    # Reduced data with final label from gempy
    if plot_dimred=="tsne":
        #TSNE_transformation(data=normalised_data, label= 7- np.round(custom_grid_values_post), filename="./Results_without_prior_gmm/tsne_gempy_final_label.png")
        filename_tsne_final_label = directory_path + "/tsne_gempy_final_label.png"
        TSNE_transformation(data=normalised_hsi, label= np.round(custom_grid_values_post), filename=filename_tsne_final_label)
if __name__ == "__main__":
    
    # Your main script code starts here
    print("Script started...")
    
    # Record the start time
    start_time = datetime.now()
    # 

    
    # Run the main function
    #main()
    main()
    # Record the end time
    end_time = datetime.now()

    # Your main script code ends here
    print("Script ended...")
    
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    
    print(f"Elapsed time: {elapsed_time}")