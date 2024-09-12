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
parser.add_argument('--cluster', metavar='cluster', type=int, default=3, help='total number of cluster')
parser.add_argument('--dimred', metavar='dimred', type=str , default="pca", help='type of dimensionality reduction')
parser.add_argument('--plot_dimred', metavar='plot_dimred', type=str , default="tsne", help='type of dimensionality reduction for plotting after data is alread reduced in a smaller dimension')
parser.add_argument('--prior_number_samples', metavar='prior_number_samples', type=int , default=100, help='number of samples for prior')
parser.add_argument('--posterior_number_samples', metavar='posterior_number_samples', type=int , default=150, help='number of samples for posterior')
parser.add_argument('--posterior_warmup_steps', metavar='posterior_warmup_steps', type=int , default=50, help='number of  warmup steps for posterior')
parser.add_argument('--directory_path', metavar='directory_path', type=str , default="./Results_with_prior_mean_cov_3_layer_KSL", help='name of the directory in which result should be stored')
parser.add_argument('--posterior_num_chain', metavar='posterior_num_chain', type=int , default=1, help='number of chain')
parser.add_argument('--slope_gempy', metavar='slope_gempy', type=float , default=40.0, help='slope for gempy')
parser.add_argument('--scale', metavar='scale', type=float , default=20.0, help='scaling factor to generate probability for each voxel')
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
    model = TSNE(n_components=1, random_state=42)
    transformed_data = model.fit_transform(data[:,1:]) 
    label_to_color = { 1: 'red', 2: 'blue', 3: 'green', 4: 'yellow', 5: 'orange', 6: 'purple'}
    transformed_data = np.concatenate([data[:,0].reshape((-1,1)), transformed_data.reshape((-1,1))], axis=1)
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
    plt.close()
    
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
    responsibilities = np.clip(responsibilities, 1e-20, 1.0)
    
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
    mixing_coefficient = np.clip(mixing_coefficient, 1e-20, 1.0)
    
    # Calculate the entropy for each data point
    entropy_per_point = -np.sum(mixing_coefficient * np.log(mixing_coefficient))
    
    # Return the average entropy
    return entropy_per_point
    
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
   
    brk1 = -845 
    brk2 = -825 
    
    
    gp.add_surface_points(
        geo_model=geo_model_test,
        x=[100.0,300, 900.0],
        y=[0.0, 0.0, 0.0],
        z=[brk1, brk1-10, brk1],
        elements_names=['surface1','surface1', 'surface1']
    )

    gp.add_orientations(
        geo_model=geo_model_test,
        x=[800],
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
            z=np.array([brk2, brk2+10, brk2]),
            names='surface2'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    geo_model_test.structural_frame.structural_groups[0].append_element(element2)

    geo_model_test.structural_frame.structural_groups[0].elements[0], geo_model_test.structural_frame.structural_groups[0].elements[1] = \
    geo_model_test.structural_frame.structural_groups[0].elements[1], geo_model_test.structural_frame.structural_groups[0].elements[0]

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
    posterior_num_chain = args.posterior_num_chain
    directory_path = args.directory_path
    slope_gempy = args.slope_gempy
    scale = args.scale
    
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
    #df_spectral_normalised.iloc[:,3] = 30 * df_spectral_normalised.iloc[:,3]
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
    
    if dimred=="pca":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=0.99)
        transformed_hsi = pca.fit_transform(normalised_hsi)
        normalised_hsi = torch.tensor(transformed_hsi, dtype=torch.float64)
        
    if dimred =="tsne":
        #######################TODO#####################
        ################################################
        raise Exception("TSNE hasn't implemented for dimensionality reduction yet")
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
    geo_model_test.interpolation_options.sigmoid_slope = slope_gempy
    gp.compute_model(geo_model_test)
    
    sp_coords_copy_test = geo_model_test.interpolation_input.surface_points.sp_coords.copy()
    
    # Compute and observe the thickness of the geological layer
    
    custom_grid_values_prior = torch.tensor(geo_model_test.solutions.octrees_output[0].last_output_center.custom_grid_values, dtype=torch.float64)
    
    
    z_nk = F.softmax(-scale* (torch.linspace(1,cluster,cluster, dtype=torch.float64) - custom_grid_values_prior.reshape(-1,1))**2, dim=1)
    entropy_z_nk_prior = calculate_average_entropy(z_nk.detach().numpy())
    entropy_mixing_prior = calculate_entropy(torch.mean(z_nk, dim=1).detach().numpy())
    entropy_z_nk_per_pixel_prior =[calculate_entropy(ele) for ele in z_nk.detach().numpy()]
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
    
    # Define the range for the number of components
    n_components_range = range(1, 10)

    # Initialize lists to store BIC scores
    bic_scores = []
    aic_scores = []
    # Fit GMM for each number of components and calculate BIC
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(normalised_hsi)
        bic_scores.append(gmm.bic(normalised_hsi))
        aic_scores.append(gmm.aic(normalised_hsi))
    # Find the number of components with the lowest BIC
    optimal_n_components_bic = n_components_range[np.argmin(bic_scores)]
    optimal_n_components_aic = n_components_range[np.argmin(aic_scores)]
    print(f"Optimal number of components: {optimal_n_components_bic}")
    print(f"Optimal number of components: {optimal_n_components_aic}")
    # Plot the BIC scores
    plt.figure(figsize=(8,10))
    plt.plot(n_components_range, bic_scores, marker='o')
    plt.xlabel('Number of components')
    plt.ylabel('BIC')
    plt.title('BIC for different number of components in GMM')
    filename_BIC = directory_path + "/bic.png"
    plt.savefig(filename_BIC)
    plt.close()
    
    # Plot the AIC scores
    plt.figure(figsize=(8,10))
    plt.plot(n_components_range, aic_scores, marker='o')
    plt.xlabel('Number of components')
    plt.ylabel('AIC')
    plt.title('AIC for different number of components in GMM')
    filename_AIC = directory_path + "/aic.png"
    plt.savefig(filename_AIC)
    plt.close()
    
    ################################################################################################
    gm = BayesianGaussianMixture(n_components=cluster, covariance_type='full', random_state=42 ).fit(normalised_hsi)
    
    # make the labels to start with 1 instead of 0
    gmm_label = gm.predict(normalised_hsi) +1 
    gamma_prior = gm.predict_proba(normalised_hsi)
    entropy_gmm_prior = calculate_average_entropy(gamma_prior)
    print("entropy_gmm_prior\n", entropy_gmm_prior)
    entropy_gmm_per_pixel_prior = [calculate_entropy(ele) for ele in gamma_prior]
    
    gmm_label_order, y_obs_label_order, accuracy_init, _ = cluster_acc( gmm_label, np.round(y_obs_label))
    
    
    # # reaarange the label information so it is would be consistent with ground truth label
    gmm_label_rearranged = torch.tensor([y_obs_label_order[x-1] +1  for x in gmm_label], dtype=torch.float64)
    

    #print(gmm_label_rearranged)
   
    # rearrange the mean and covariance accordingly too
    #rearrange_list = [1,2,0]
    #rearrange_list = [3,4,2,0,5,1]
    rearrange_list = y_obs_label_order
    mean_init, cov_init = gm.means_[rearrange_list], gm.covariances_[rearrange_list]
    
    eigen_vector_list , eigen_values_list =[],[]
    for i in range(cov_init.shape[0]):
        eigen_values, eigen_vectors = np.linalg.eig(cov_init[i])
        # Lambda_ = np.diag(eigen_values)
        # D = np.array(eigen_vectors)
        # A = D @ Lambda_ @ D.T
        # # Check if the values is close to Zero
        # threshold= 1e-15
        # Delta = A - cov_init[i]
        # Delta[np.abs(Delta)< threshold] =0
        # print(Delta)
        eigen_values_list.append(eigen_values)
        eigen_vector_list.append(eigen_vectors)
    
    ####################################TODO#################################################
    #   Try to find the initial accuracy of classification
    #########################################################################################
    print("Intial accuracy\n", accuracy_init)
    Z_data = torch.tensor(dataframe_KSL.iloc[:,3].to_numpy(), dtype=torch.float64)
    #################################TODO##################################################
    ## Apply different dimentionality reduction techniques and save the plot in Result file
    #######################################################################################
    if plot_dimred =="tsne":
        data = torch.cat([Z_data.reshape((-1,1)), normalised_hsi], dim=1)
        #print(data.shape)
        filename_tsne = directory_path + "/tsne_gmm_label.png"
        TSNE_transformation(data=data, label=gmm_label_rearranged, filename=filename_tsne)
    
    #geo_model_test.transform.apply_inverse(sp_coords_copy_test)
    
    # Change the backend to PyTorch for probabilistic modeling
    BackendTensor.change_backend_gempy(engine_backend=gp.data.AvailableBackends.PYTORCH)
    
    # geo_model_test.interpolation_options.uni_degree = 0
    # geo_model_test.interpolation_options.mesh_extraction = False
    geo_model_test.interpolation_options.sigmoid_slope = slope_gempy
    
    store_accuracy=[]
    factor=100 #0.01 , 100
    alpha = 10
    beta  = 10
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
        
        
        mu_surface_1 = pyro.sample('mu_1', dist.Normal(prior_mean_surface_1, torch.tensor(0.2, dtype=torch.float64)))
        mu_surface_2 = pyro.sample('mu_2', dist.Normal(prior_mean_surface_2, torch.tensor(0.2, dtype=torch.float64)))
        
        
        pyro.sample('mu_1 > mu_2', dist.Delta(torch.tensor(1.0, dtype=torch.float64)), obs=(mu_surface_1 > mu_surface_2))
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
        # loc_mean = torch.tensor(mean_init,dtype=torch.float64)
        # loc_cov =  torch.tensor(cov_init, dtype=torch.float64)
        z_nk = F.softmax(-scale* (torch.linspace(1,cluster,cluster, dtype=torch.float64) - custom_grid_values.reshape(-1,1))**2, dim=1)
        #class_label = torch.mean(F.softmax(-lambda_* (torch.tensor([1,2,3,4,5,6], dtype=torch.float64) - custom_grid_values.reshape(-1,1))**2, dim=1),dim=0)
        
        N_k = torch.sum(z_nk,axis=0)
        N = len(custom_grid_values)
        pi_k = N_k /N
        
        loc_mean = torch.tensor(mean_init,dtype=torch.float64)
        loc_cov =  torch.tensor(cov_init, dtype=torch.float64)
        cov_matrix_mean = alpha * torch.eye(loc_mean[0].shape[0], dtype=torch.float64)
        cov_matrix_cov = beta * torch.eye(loc_mean[0].shape[0], dtype=torch.float64)
        
        mean =[]
        cov = []
        for i in range(loc_mean.shape[0]):
            mean_data= pyro.sample("mean_data_"+str(i+1), dist.MultivariateNormal(loc=loc_mean[i],covariance_matrix=cov_matrix_mean))
            mean.append(mean_data)
            eigen_values_init = torch.tensor(eigen_values_list[i],dtype=torch.float64)
            eigen_vectors_data = torch.tensor(eigen_vector_list[i], dtype=torch.float64)
            cov_eigen_values = pyro.sample("cov_eigen_values_"+str(i+1), dist.MultivariateNormal(loc=torch.sqrt(eigen_values_init),covariance_matrix=cov_matrix_cov))
            cov_data = eigen_vectors_data @ torch.diag(cov_eigen_values)**2 @ eigen_vectors_data.T #+ 1e-6 * torch.eye(loc_mean[0].shape[0], dtype=torch.float64)
            #print(torch.linalg.det(sample_cov_data))
            cov.append(cov_data)
        
        mean_tensor = torch.stack(mean, dim=0)
        cov_tensor = torch.stack(cov, dim=0)
        
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
    plt.close()
    ################################################################################
    # Posterior 
    ################################################################################
    pyro.primitives.enable_validation(is_validate=True)
    nuts_kernel = NUTS(model_test, step_size=0.0085, adapt_step_size=True, target_accept_prob=0.9, max_tree_depth=10, init_strategy=init_to_mean)
    mcmc = MCMC(nuts_kernel, num_samples=posterior_number_samples, warmup_steps=posterior_warmup_steps, disable_validation=False)
    mcmc.run(normalised_hsi)
    
    posterior_samples = mcmc.get_samples()
    # Convert the tensors to lists
    posterior_samples_serializable = {k: v.tolist() for k, v in posterior_samples.items()}
    
    posterior_samples_serializable["eigen_vectors"]= [ele.tolist() for ele in eigen_vector_list]
    filename_posterior_samples =directory_path + "/posterior_samples.json"
    # Save to a JSON file
    with open(filename_posterior_samples, 'w') as f:
        json.dump(posterior_samples_serializable, f)
        
    posterior_predictive = Predictive(model_test, posterior_samples)(normalised_hsi)
    plt.figure(figsize=(8,10))
    data = az.from_pyro(posterior=mcmc, prior=prior, posterior_predictive=posterior_predictive)
    az.plot_trace(data)
    filename_posteriro_plot = directory_path + "/posterior.png"
    plt.savefig(filename_posteriro_plot)
    plt.close()
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
    plt.close()
    
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
    plt.close()
    ###############################################TODO################################
    # Find the MAP value
    ###################################################################################
    
    unnormalise_posterior_value={}
    unnormalise_posterior_value["log_prior_geo_list"]=[]
    unnormalise_posterior_value["log_prior_hsi_mean_list"]=[]
    unnormalise_posterior_value["log_prior_hsi_cov_list"]=[]
    unnormalise_posterior_value["log_likelihood_list"]=[]
    unnormalise_posterior_value["log_posterior_list"]=[]
    # log_prior_geo_list=[]
    # log_prior_hsi_list=[]
    # log_likelihood_list=[]
    # log_posterior_list=[]
    keys_list = list(posterior_samples.keys())
   
    prior_mean_surface_1 = sp_coords_copy_test[1, 2]
    prior_mean_surface_2 = sp_coords_copy_test[4, 2]
   
    store_z_nk_entropy =[]
    store_gmm_entropy=[]
    store_mixing_entropy=[]
    
    for i in range(posterior_samples["mu_1"].shape[0]):
        
        post_cov_eigen_values_1 = posterior_samples[keys_list[0]][i]
        post_cov_eigen_values_2 = posterior_samples[keys_list[1]][i]
        post_cov_eigen_values_3 = posterior_samples[keys_list[2]][i]
        post_mean_data_1 = posterior_samples[keys_list[3]][i]
        post_mean_data_2 = posterior_samples[keys_list[4]][i]
        post_mean_data_3 = posterior_samples[keys_list[5]][i]
        post_mu_1 = posterior_samples[keys_list[6]][i]
        post_mu_2 = posterior_samples[keys_list[7]][i]
        
        
       
        # Calculate the log probability of the value
        
        log_prior_geo = dist.Normal(prior_mean_surface_1, torch.tensor(0.2, dtype=torch.float64)).log_prob(post_mu_1)+\
                    dist.Normal(prior_mean_surface_2, torch.tensor(0.2, dtype=torch.float64)).log_prob(post_mu_2)
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
        # loc_mean = torch.tensor(mean_init,dtype=torch.float64)
        # loc_cov =  torch.tensor(cov_init, dtype=torch.float64)
        z_nk = F.softmax(-scale* (torch.linspace(1,cluster,cluster, dtype=torch.float64) - custom_grid_values.reshape(-1,1))**2, dim=1)
        #class_label = torch.mean(F.softmax(-scale* (torch.tensor([1,2,3,4,5,6], dtype=torch.float64) - custom_grid_values.reshape(-1,1))**2, dim=1),dim=0)
        
        N_k = torch.sum(z_nk,axis=0)
        N = len(custom_grid_values)
        pi_k = N_k /N
        
        loc_mean = torch.tensor(mean_init,dtype=torch.float64)
        #loc_cov =  torch.tensor(cov_init, dtype=torch.float64)
        cov_matrix_mean = alpha * torch.eye(loc_mean[0].shape[0], dtype=torch.float64)
        cov_matrix_cov = beta * torch.eye(loc_mean[0].shape[0], dtype=torch.float64)
        log_prior_hsi_mean =torch.tensor(0.0, dtype=torch.float64)
        for j in range(loc_mean.shape[0]):
            log_prior_hsi_mean = log_prior_hsi_mean + dist.MultivariateNormal(loc=loc_mean[j],covariance_matrix=cov_matrix_mean).log_prob(posterior_samples[keys_list[3+j]][i])
        # log_prior_hsi = dist.MultivariateNormal(loc=loc_mean[0],covariance_matrix=cov_matrix).log_prob(post_sample_data1)+\
        #                 dist.MultivariateNormal(loc=loc_mean[1],covariance_matrix=cov_matrix).log_prob(post_sample_data2)+\
        #                 dist.MultivariateNormal(loc=loc_mean[2],covariance_matrix=cov_matrix).log_prob(post_sample_data3)+\
        #                 dist.MultivariateNormal(loc=loc_mean[3],covariance_matrix=cov_matrix).log_prob(post_sample_data4)
        
        # We can also calculate the accuracy using the mean and covariance to see if our GMM model has imroved or not
        log_prior_hsi_cov =torch.tensor(0.0, dtype=torch.float64)
        cov = []
        for j in range(loc_mean.shape[0]):
            eigen_values_init = torch.tensor(eigen_values_list[j],dtype=torch.float64)
            eigen_vectors_data = torch.tensor(eigen_vector_list[j], dtype=torch.float64)
            log_prior_hsi_cov = log_prior_hsi_cov + dist.MultivariateNormal(loc=torch.sqrt(eigen_values_init),covariance_matrix=cov_matrix_cov).log_prob(posterior_samples[keys_list[j]][i])
            cov_data = eigen_vectors_data @ torch.diag(posterior_samples[keys_list[j]][i])**2 @ eigen_vectors_data.T
            cov.append(cov_data)
        cov_tensor = torch.stack(cov, dim=0)
        
        gamma_nk = torch.zeros(z_nk.shape, dtype=torch.float64)

        # if i==7:
        #     print("Let's check the 7th sample ")
        #     print("mu_1\n", post_mu_1)
        #     print("mu_2\n", post_mu_2)
        #     #print("custom_grid_values\n", custom_grid_values)
        #     print("pi_k\n", pi_k)
        log_likelihood=torch.tensor(0.0, dtype=torch.float64)
        for j in range(normalised_hsi.shape[0]):
            likelihood = pi_k[0] *torch.exp(dist.MultivariateNormal(loc=post_mean_data_1,covariance_matrix= cov_tensor[0]).log_prob(normalised_hsi[j])) +\
                         pi_k[1] *torch.exp(dist.MultivariateNormal(loc=post_mean_data_2,covariance_matrix= cov_tensor[1]).log_prob(normalised_hsi[j]))+\
                         pi_k[2] *torch.exp(dist.MultivariateNormal(loc=post_mean_data_3,covariance_matrix= cov_tensor[2]).log_prob(normalised_hsi[j])) 
                         
            for k in range(loc_mean.shape[0]):    
                gamma_nk[j][k] = (pi_k[k] * torch.exp(dist.MultivariateNormal(loc=posterior_samples[keys_list[3+k]][i],covariance_matrix= cov_tensor[k]).log_prob(normalised_hsi[j]))) / likelihood
                
            log_likelihood += torch.log(likelihood)
        
        entropy_gmm = calculate_entropy(gamma_nk.detach().numpy())
        entropy_z_nk = calculate_average_entropy(z_nk.detach().numpy())
        entropy_pi_k = calculate_entropy(pi_k.detach().numpy())
        store_z_nk_entropy.append(entropy_z_nk)
        store_gmm_entropy.append(entropy_gmm)
        store_mixing_entropy.append(entropy_pi_k)
        #print("gmm_label_accuracy", gmm_accuracy)
        # log_prior_geo_list.append(log_prior_geo)
        # log_prior_hsi_list.append(log_prior_hsi)
        # log_likelihood_list.append(log_likelihood)
        # log_posterior_list.append(log_prior_geo + log_prior_hsi + log_likelihood)
        unnormalise_posterior_value["log_prior_geo_list"].append(log_prior_geo)
        unnormalise_posterior_value["log_prior_hsi_mean_list"].append(log_prior_hsi_mean)
        unnormalise_posterior_value["log_prior_hsi_cov_list"].append(log_prior_hsi_cov)
        unnormalise_posterior_value["log_likelihood_list"].append(log_likelihood)
        unnormalise_posterior_value["log_posterior_list"].append(log_prior_geo +log_prior_hsi_mean + log_prior_hsi_cov +log_likelihood)
    
    MAP_sample_index=torch.argmax(torch.tensor(unnormalise_posterior_value["log_posterior_list"]))
    
    
    #print(unnormalise_posterior_value)
    
    # Extract acceptance probabilities

    plt.figure(figsize=(8,10))
    plt.plot(np.array([ele.detach().detach() for ele in unnormalise_posterior_value["log_prior_geo_list"]]), label='prior_geo', marker=".")
    plt.plot(np.array([ele.detach().detach() for ele in unnormalise_posterior_value["log_prior_hsi_mean_list"]]), label='prior_hsi_mean', marker="o")
    plt.plot(np.array([ele.detach().detach() for ele in unnormalise_posterior_value["log_prior_hsi_cov_list"]]), label='prior_hsi_cov', marker="*")
    plt.plot(np.array([ele.detach().detach() for ele in unnormalise_posterior_value["log_likelihood_list"]]), label='prior_likelihood')
    plt.plot(np.array([ele.detach().detach() for ele in unnormalise_posterior_value["log_posterior_list"]]), label='posterior')
    plt.xlabel('Iteration')
    plt.ylabel('Unnormalised log value')
    plt.title('Log value of each term in posterior')
    plt.legend()
    filename_log = directory_path + "/log_posteriro.png"
    plt.savefig(filename_log)
    plt.close()
    
    plt.figure(figsize=(8,10))
    plt.plot(np.array(store_z_nk_entropy), label="Responsibility Entropy")
    plt.plot(np.array(store_gmm_entropy), label = 'GMM Entropy')
    plt.plot(np.array(store_mixing_entropy), label="Mixing Coefficient Entropy")
    plt.xlabel('Iteration')
    plt.ylabel('average entropy')
    plt.title('Average entropy of the sample')
    plt.legend()
    filename_entropy = directory_path + "/average_entropy.png"
    plt.savefig(filename_entropy)
    plt.close()
    ################################################################################
    #  Try Plot the data and save it as file in output folder
    ################################################################################
    # mu_1_mean = posterior_samples["mu_1"].mean()
    # mu_2_mean = posterior_samples["mu_2"].mean()
    # mu_3_mean = posterior_samples["mu_3"].mean()
    # mu_4_mean = posterior_samples["mu_4"].mean()
    
    
    mu_1_post = posterior_samples["mu_1"][MAP_sample_index]
    mu_2_post = posterior_samples["mu_2"][MAP_sample_index]
    
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
    
    #print("interpolation_input",interpolation_input.surface_points.sp_coords)

    # # Compute the geological model
    geo_model_test.solutions = gempy_engine.compute_model(
        interpolation_input=interpolation_input,
        options=geo_model_test.interpolation_options,
        data_descriptor=geo_model_test.input_data_descriptor,
        geophysics_input=geo_model_test.geophysics_input,
    )
    # Compute and observe the thickness of the geological layer
     
    custom_grid_values_test = geo_model_test.solutions.octrees_output[0].last_output_center.custom_grid_values
    
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
    
    brk1 = -845 
    brk2 = -825 
    

    gp.add_surface_points(
        geo_model=geo_model_test_post,
        x=[100.0,300, 900.0],
        y=[0.0,0.0, 0.0],
        z=[brk1,sp_cord[4,2], brk1],
        elements_names=['surface1','surface1', 'surface1']
    )

    gp.add_orientations(
        geo_model=geo_model_test_post,
        x=[800],
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
            z=np.array([brk2, sp_cord[1,2], brk2]),
            names='surface2'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    geo_model_test_post.structural_frame.structural_groups[0].append_element(element2)

    
    
    geo_model_test_post.structural_frame.structural_groups[0].elements[0], geo_model_test_post.structural_frame.structural_groups[0].elements[1] = \
    geo_model_test_post.structural_frame.structural_groups[0].elements[1], geo_model_test_post.structural_frame.structural_groups[0].elements[0]

    

    gp.set_custom_grid(geo_model_test_post.grid, xyz_coord=xyz_coord)
    gp.compute_model(geo_model_test_post)
    
    custom_grid_values_post = geo_model_test_post.solutions.octrees_output[0].last_output_center.custom_grid_values
    
    ####################################TODO#################################################
    #   Try to find the final accuracy to check if it has improved the classification
    #########################################################################################

    
    ###################################TODO###################################################
    # store the weights, mean, and covariance of Gaussian mixture model
    ##########################################################################################
    
    # loc_mean = torch.tensor(mean_init,dtype=torch.float64)
    # loc_cov =  torch.tensor(cov_init, dtype=torch.float64)
    z_nk = F.softmax(-scale* (torch.linspace(1,cluster,cluster, dtype=torch.float64) - torch.tensor(custom_grid_values_test.detach().numpy()).reshape(-1,1))**2, dim=1)
    #class_label = torch.mean(F.softmax(-scale* (torch.tensor([1,2,3,4,5,6], dtype=torch.float64) - custom_grid_values.reshape(-1,1))**2, dim=1),dim=0)
    
    N_k = torch.sum(z_nk,axis=0)
    N = len(custom_grid_values)
    pi_k = N_k /N
    mean = []
    cov = []
    
    for i in range(z_nk.shape[1]):
        mean_k = posterior_samples[keys_list[3+i]][MAP_sample_index]
        #cov_k = torch.sum( (normalised_hsi - mean_k.reshape((-1,1))) (normalised_hsi - mean_k).T )
        cov_eig_k = posterior_samples[keys_list[i]][MAP_sample_index]
        eigen_vectors_data = torch.tensor(eigen_vector_list[i], dtype=torch.float64)
        cov_k = eigen_vectors_data @ torch.diag(cov_eig_k)**2 @ eigen_vectors_data.T
        mean.append(mean_k)
        cov.append(cov_k)
    mean_tensor = torch.stack(mean, dim=0)
    cov_tensor = torch.stack(cov,dim=0)
    
    gmm_data ={}
    gmm_data["weights"]=pi_k.detach().numpy().tolist()
    gmm_data["means"] = mean_tensor.detach().numpy().tolist()
    gmm_data["cov"] = cov_tensor.detach().numpy().tolist()
    
    gamma_post =torch.zeros(z_nk.shape, dtype=torch.float64)
    for j in range(normalised_hsi.shape[0]):
            likelihood = pi_k[0] *torch.exp(dist.MultivariateNormal(loc=mean_tensor[0],covariance_matrix= cov_tensor[0]).log_prob(normalised_hsi[j])) +\
                         pi_k[1] *torch.exp(dist.MultivariateNormal(loc=mean_tensor[1],covariance_matrix= cov_tensor[1]).log_prob(normalised_hsi[j]))+\
                         pi_k[2] *torch.exp(dist.MultivariateNormal(loc=mean_tensor[2],covariance_matrix= cov_tensor[2]).log_prob(normalised_hsi[j])) 
                         
            for k in range(loc_mean.shape[0]):    
                gamma_post[j][k] = (pi_k[k] * torch.exp(dist.MultivariateNormal(loc=mean_tensor[k],covariance_matrix= cov_tensor[k]).log_prob(normalised_hsi[j]))) / likelihood
    
    entropy_MAP_gmm = calculate_average_entropy(gamma_post.detach().numpy())
    entropy_MAP_z_nk = calculate_average_entropy(z_nk.detach().numpy())
    entropy_MAP_mixing = calculate_entropy(pi_k.detach().numpy())
    entropy_gmm_per_pixel_post = [calculate_entropy(ele) for ele in gamma_post.detach().numpy()]
    entropy_z_nk_per_pixel_post =[calculate_entropy(ele) for ele in z_nk.detach().numpy()]
    # Plot the entropy
    plt.figure(figsize=(8,10))
    plt.plot(np.array(df_spectral_normalised.iloc[:,3]), np.array(entropy_gmm_per_pixel_prior), label=" Responsibility_prior")
    plt.plot(np.array(df_spectral_normalised.iloc[:,3]),np.array(entropy_gmm_per_pixel_post), label = 'Responsibility_post')
    plt.xlabel('pixel')
    plt.ylabel('Entropy per pixel')
    plt.title('Entropy vs pixel')
    plt.legend()
    filename_entropy_pixel_responsibility = directory_path + "/entropy_per_pixel_responsibility.png"
    plt.savefig(filename_entropy_pixel_responsibility)
    
    plt.figure(figsize=(8,10))
    plt.plot(np.array(df_spectral_normalised.iloc[:,3]),np.array(entropy_z_nk_per_pixel_prior), label="z_nk_prior")
    plt.plot(np.array(df_spectral_normalised.iloc[:,3]),np.array(entropy_z_nk_per_pixel_post), label="z_nk_post")
    plt.xlabel('pixel')
    plt.ylabel('Entropy per pixel')
    plt.title('Entropy vs pixel')
    plt.legend()
    filename_entropy_pixel_z_nk = directory_path + "/entropy_per_pixel_z_nk.png"
    plt.savefig(filename_entropy_pixel_z_nk)
    plt.close()
    
    print("MAP index\n", MAP_sample_index)
    print("entropy_prior_gmm\n",entropy_gmm_prior,"\n", "entropy_MAP_gmm\n", entropy_MAP_gmm)
    print("entropy_z_nk_prior\n", entropy_z_nk_prior)
    print("entropy_MAP_z_nk\n", entropy_MAP_z_nk)
    print("entropy_mixing_prior\n", entropy_mixing_prior)
    print("entropy_MAP_mixing\n", entropy_MAP_mixing)
    
    entropy_data ={}
    entropy_data["MAP index"] = MAP_sample_index.detach().numpy().tolist()
    entropy_data["entropy_prior_gmm"] = entropy_gmm_prior.tolist()
    entropy_data["entropy_MAP_gmm"] = entropy_MAP_gmm.tolist()
    entropy_data["entropy_z_nk_prior"] = entropy_z_nk_prior.tolist()
    entropy_data["entropy_MAP_z_nk"] = entropy_MAP_z_nk.tolist()
    entropy_data["entropy_mixing_prior"] = entropy_mixing_prior.tolist()
    entropy_data["entropy_MAP_mixing"] = entropy_MAP_mixing.tolist()
    
    
    filename_entropy_data =directory_path + "/entropy_data.json"
    with open(filename_entropy_data, "w") as json_file:
        json.dump(entropy_data, json_file)
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
        data = torch.cat([Z_data.reshape((-1,1)), normalised_hsi], dim=1)
        #TSNE_transformation(data=normalised_data, label= 7- np.round(custom_grid_values_post), filename="./Results_without_prior_gmm/tsne_gempy_final_label.png")
        filename_tsne_final_label = directory_path + "/tsne_gempy_final_label.png"
        TSNE_transformation(data=data, label= np.round(custom_grid_values_post), filename=filename_tsne_final_label)
    ##########################################################################################
    #  Mean 
    ##########################################################################################
    # mu_1_mean = posterior_samples["mu_1"].mean()
    # mu_2_mean = posterior_samples["mu_2"].mean()
    # mu_3_mean = posterior_samples["mu_3"].mean()
    # mu_4_mean = posterior_samples["mu_4"].mean()

    
    mu_1_mean = posterior_samples["mu_1"].mean()
    mu_2_mean = posterior_samples["mu_2"].mean()

    # # Update the model with the new top layer's location
    BackendTensor.change_backend_gempy(engine_backend=gp.data.AvailableBackends.PYTORCH)
    
    interpolation_input = geo_model_test.interpolation_input
    interpolation_input.surface_points.sp_coords = torch.index_put(
        interpolation_input.surface_points.sp_coords,
        (torch.tensor([1]), torch.tensor([2])),
        mu_1_mean
    )
    interpolation_input.surface_points.sp_coords = torch.index_put(
        interpolation_input.surface_points.sp_coords,
        (torch.tensor([4]), torch.tensor([2])),
        mu_2_mean
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
     
    custom_grid_values_test = geo_model_test.solutions.octrees_output[0].last_output_center.custom_grid_values

    sp_coords_copy_test2 =interpolation_input.surface_points.sp_coords
    sp_cord= geo_model_test.transform.apply_inverse(sp_coords_copy_test2.detach().numpy())

    ################################################################################
    # Store the Initial Interface data and orientation data
    ################################################################################
    df_sp_final = df_sp_init.copy()
    df_sp_final["Z"] = sp_cord[:,2] 
    filename_final_sp = directory_path + "/Final_sp_mean.csv"
    df_sp_final.to_csv(filename_final_sp)
    ################################################################################
        
    geo_model_test_mean = gp.create_geomodel(
    project_name='Gempy_abc_Test_mean',
    extent=[0, 1000, -10, 10, -900, -700],
    resolution=[100,10,100],
    refinement=7,
    structural_frame= gp.data.StructuralFrame.initialize_default_structure()
    )
    
    brk1 = -845 
    brk2 = -825 
    

    gp.add_surface_points(
        geo_model=geo_model_test_mean,
        x=[100.0,300, 900.0],
        y=[0.0,0.0, 0.0],
        z=[brk1,sp_cord[4,2], brk1],
        elements_names=['surface1','surface1', 'surface1']
    )

    gp.add_orientations(
        geo_model=geo_model_test_mean,
        x=[800],
        y=[0.0],
        z=[brk1],
        elements_names=['surface1'],
        pole_vector=[[0, 0, 1]]
    )
    geo_model_test_mean.update_transform(gp.data.GlobalAnisotropy.NONE)

    element2 = gp.data.StructuralElement(
        name='surface2',
        color=next(geo_model_test_mean.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([100.0,300, 900.0]),
            y=np.array([0.0,0.0, 0.0]),
            z=np.array([brk2, sp_cord[1,2], brk2]),
            names='surface2'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    geo_model_test_mean.structural_frame.structural_groups[0].append_element(element2)

    
    
    geo_model_test_mean.structural_frame.structural_groups[0].elements[0], geo_model_test_mean.structural_frame.structural_groups[0].elements[1] = \
    geo_model_test_mean.structural_frame.structural_groups[0].elements[1], geo_model_test_mean.structural_frame.structural_groups[0].elements[0]

    

    gp.set_custom_grid(geo_model_test_mean.grid, xyz_coord=xyz_coord)
    gp.compute_model(geo_model_test_mean)
    
    custom_grid_values_mean = geo_model_test_post.solutions.octrees_output[0].last_output_center.custom_grid_values
    ###################################################################################
    # loc_mean = torch.tensor(mean_init,dtype=torch.float64)
    # loc_cov =  torch.tensor(cov_init, dtype=torch.float64)
    z_nk = F.softmax(-scale* (torch.linspace(1,cluster,cluster, dtype=torch.float64) - torch.tensor(custom_grid_values_test.detach().numpy()).reshape(-1,1))**2, dim=1)
    #class_label = torch.mean(F.softmax(-scale* (torch.tensor([1,2,3,4,5,6], dtype=torch.float64) - custom_grid_values.reshape(-1,1))**2, dim=1),dim=0)
    
    N_k = torch.sum(z_nk,axis=0)
    N = len(custom_grid_values)
    pi_k = N_k /N
    mean = []
    cov = []
    
    for i in range(z_nk.shape[1]):
        mean_k = torch.mean(posterior_samples[keys_list[3+i]], dim=0)
        #cov_k = torch.sum( (normalised_hsi - mean_k.reshape((-1,1))) (normalised_hsi - mean_k).T )
        cov_eig_k = torch.mean(posterior_samples[keys_list[i]],dim=0)
        eigen_vectors_data = torch.tensor(eigen_vector_list[i], dtype=torch.float64)
        cov_k = eigen_vectors_data @ torch.diag(cov_eig_k)**2 @ eigen_vectors_data.T
        mean.append(mean_k)
        cov.append(cov_k)
    mean_tensor = torch.stack(mean, dim=0)
    cov_tensor = torch.stack(cov,dim=0)
    
    gmm_data ={}
    gmm_data["weights"]=pi_k.detach().numpy().tolist()
    gmm_data["means"] = mean_tensor.detach().numpy().tolist()
    gmm_data["cov"] = cov_tensor.detach().numpy().tolist()
    
    gamma_post =torch.zeros(z_nk.shape, dtype=torch.float64)
    for j in range(normalised_hsi.shape[0]):
            likelihood = pi_k[0] *torch.exp(dist.MultivariateNormal(loc=mean_tensor[0],covariance_matrix= cov_tensor[0]).log_prob(normalised_hsi[j])) +\
                         pi_k[1] *torch.exp(dist.MultivariateNormal(loc=mean_tensor[1],covariance_matrix= cov_tensor[1]).log_prob(normalised_hsi[j]))+\
                         pi_k[2] *torch.exp(dist.MultivariateNormal(loc=mean_tensor[2],covariance_matrix= cov_tensor[2]).log_prob(normalised_hsi[j])) 
                         
            for k in range(loc_mean.shape[0]):    
                gamma_post[j][k] = (pi_k[k] * torch.exp(dist.MultivariateNormal(loc=mean_tensor[k],covariance_matrix= cov_tensor[k]).log_prob(normalised_hsi[j]))) / likelihood
    
    entropy_MAP_gmm = calculate_average_entropy(gamma_post.detach().numpy())
    entropy_MAP_z_nk = calculate_average_entropy(z_nk.detach().numpy())
    entropy_MAP_mixing = calculate_entropy(pi_k.detach().numpy())
    entropy_gmm_per_pixel_post = [calculate_entropy(ele) for ele in gamma_post.detach().numpy()]
    entropy_z_nk_per_pixel_post =[calculate_entropy(ele) for ele in z_nk.detach().numpy()]
    # Plot the entropy
    plt.figure(figsize=(8,10))
    plt.plot(np.array(df_spectral_normalised.iloc[:,3]), np.array(entropy_gmm_per_pixel_prior), label=" Responsibility_prior")
    plt.plot(np.array(df_spectral_normalised.iloc[:,3]),np.array(entropy_gmm_per_pixel_post), label = 'Responsibility_post')
    plt.xlabel('pixel')
    plt.ylabel('Entropy per pixel')
    plt.title('Entropy vs pixel')
    plt.legend()
    filename_entropy_pixel_responsibility = directory_path + "/entropy_per_pixel_responsibility_mean.png"
    plt.savefig(filename_entropy_pixel_responsibility)
    
    plt.figure(figsize=(8,10))
    plt.plot(np.array(df_spectral_normalised.iloc[:,3]),np.array(entropy_z_nk_per_pixel_prior), label="z_nk_prior")
    plt.plot(np.array(df_spectral_normalised.iloc[:,3]),np.array(entropy_z_nk_per_pixel_post), label="z_nk_post")
    plt.xlabel('pixel')
    plt.ylabel('Entropy per pixel')
    plt.title('Entropy vs pixel')
    plt.legend()
    filename_entropy_pixel_z_nk = directory_path + "/entropy_per_pixel_z_nk_mean.png"
    plt.savefig(filename_entropy_pixel_z_nk)
    plt.close()
    
    
    print("entropy_prior_gmm\n",entropy_gmm_prior,"\n", "entropy_MAP_gmm\n", entropy_MAP_gmm)
    print("entropy_z_nk_prior\n", entropy_z_nk_prior)
    print("entropy_MAP_z_nk\n", entropy_MAP_z_nk)
    print("entropy_mixing_prior\n", entropy_mixing_prior)
    print("entropy_MAP_mixing\n", entropy_MAP_mixing)
    
    entropy_data ={}
    entropy_data["entropy_prior_gmm"] = entropy_gmm_prior.tolist()
    entropy_data["entropy_MAP_gmm"] = entropy_MAP_gmm.tolist()
    entropy_data["entropy_z_nk_prior"] = entropy_z_nk_prior.tolist()
    entropy_data["entropy_MAP_z_nk"] = entropy_MAP_z_nk.tolist()
    entropy_data["entropy_mixing_prior"] = entropy_mixing_prior.tolist()
    entropy_data["entropy_MAP_mixing"] = entropy_MAP_mixing.tolist()
    
    
    filename_entropy_data =directory_path + "/entropy_data_mean.json"
    with open(filename_entropy_data, "w") as json_file:
        json.dump(entropy_data, json_file)
    # Save to file
    filename_gmm_data = directory_path + "/gmm_data_mean.json"
    with open(filename_gmm_data, "w") as json_file:
        json.dump(gmm_data, json_file)
        
    ##########################################################################################
    
    picture_test_post = gpv.plot_2d(geo_model_test_post, cell_number=5, legend='force')
    filename_posterior_model = directory_path + "/mean_model.png"
    plt.savefig(filename_posterior_model)
    # Reduced data with final label from gempy
    if plot_dimred=="tsne":
        data = torch.cat([Z_data.reshape((-1,1)), normalised_hsi], dim=1)
        #TSNE_transformation(data=normalised_data, label= 7- np.round(custom_grid_values_post), filename="./Results_without_prior_gmm/tsne_gempy_final_label.png")
        filename_tsne_final_label = directory_path + "/tsne_gempy_final_label_mean.png"
        TSNE_transformation(data=data, label= np.round(custom_grid_values_mean), filename=filename_tsne_final_label)
    
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