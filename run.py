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
parser.add_argument('--cluster', metavar='cluster', type=int, default=6, help='total number of cluster')
parser.add_argument('--dimred', metavar='dimred', type=str , default="pca", help='type of dimensionality reduction')
parser.add_argument('--plot_dimred', metavar='plot_dimred', type=str , default="tsne", help='type of dimensionality reduction for plotting after data is alread reduced in a smaller dimension')
parser.add_argument('--prior_number_samples', metavar='prior_number_samples', type=int , default=5, help='number of samples for prior')
parser.add_argument('--posterior_number_samples', metavar='posterior_number_samples', type=int , default=4, help='number of samples for posterior')
parser.add_argument('--posterior_warmup_steps', metavar='posterior_warmup_steps', type=int , default=0, help='number of  warmup steps for posterior')
parser.add_argument('--directory_path', metavar='directory_path', type=str , default="./Results_test", help='name of the directory in which result should be stored')
parser.add_argument('--posterior_num_chain', metavar='posterior_num_chain', type=int , default=2, help='number of chain')

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
    extent=[0, 86, -10, 10, -83, 0],
    resolution=[86,20,83],
    refinement=refinement,
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

    geo_model_test.structural_frame.structural_groups[0].elements[0], geo_model_test.structural_frame.structural_groups[0].elements[1]=\
    geo_model_test.structural_frame.structural_groups[0].elements[1], geo_model_test.structural_frame.structural_groups[0].elements[0]
    geo_model_test.structural_frame.structural_groups[0].elements[2], geo_model_test.structural_frame.structural_groups[0].elements[1]=\
    geo_model_test.structural_frame.structural_groups[0].elements[1], geo_model_test.structural_frame.structural_groups[0].elements[2]
    geo_model_test.structural_frame.structural_groups[0].elements[3], geo_model_test.structural_frame.structural_groups[0].elements[2]=\
    geo_model_test.structural_frame.structural_groups[0].elements[2], geo_model_test.structural_frame.structural_groups[0].elements[3]
    geo_model_test.structural_frame.structural_groups[0].elements[4], geo_model_test.structural_frame.structural_groups[0].elements[3]=\
    geo_model_test.structural_frame.structural_groups[0].elements[3], geo_model_test.structural_frame.structural_groups[0].elements[4]
    geo_model_test.structural_frame.structural_groups[0].elements[5], geo_model_test.structural_frame.structural_groups[0].elements[4]=\
    geo_model_test.structural_frame.structural_groups[0].elements[4], geo_model_test.structural_frame.structural_groups[0].elements[5]
    
    
    geo_model_test.structural_frame.structural_groups[0].elements[0], geo_model_test.structural_frame.structural_groups[0].elements[1]=\
    geo_model_test.structural_frame.structural_groups[0].elements[1], geo_model_test.structural_frame.structural_groups[0].elements[0]
    geo_model_test.structural_frame.structural_groups[0].elements[2], geo_model_test.structural_frame.structural_groups[0].elements[1]=\
    geo_model_test.structural_frame.structural_groups[0].elements[1], geo_model_test.structural_frame.structural_groups[0].elements[2]
    geo_model_test.structural_frame.structural_groups[0].elements[3], geo_model_test.structural_frame.structural_groups[0].elements[2]=\
    geo_model_test.structural_frame.structural_groups[0].elements[2], geo_model_test.structural_frame.structural_groups[0].elements[3]
    geo_model_test.structural_frame.structural_groups[0].elements[4], geo_model_test.structural_frame.structural_groups[0].elements[3]=\
    geo_model_test.structural_frame.structural_groups[0].elements[3], geo_model_test.structural_frame.structural_groups[0].elements[4]
    
    
    geo_model_test.structural_frame.structural_groups[0].elements[0], geo_model_test.structural_frame.structural_groups[0].elements[1]=\
    geo_model_test.structural_frame.structural_groups[0].elements[1], geo_model_test.structural_frame.structural_groups[0].elements[0]
    geo_model_test.structural_frame.structural_groups[0].elements[2], geo_model_test.structural_frame.structural_groups[0].elements[1]=\
    geo_model_test.structural_frame.structural_groups[0].elements[1], geo_model_test.structural_frame.structural_groups[0].elements[2]
    geo_model_test.structural_frame.structural_groups[0].elements[3], geo_model_test.structural_frame.structural_groups[0].elements[2]=\
    geo_model_test.structural_frame.structural_groups[0].elements[2], geo_model_test.structural_frame.structural_groups[0].elements[3]
    
    
    geo_model_test.structural_frame.structural_groups[0].elements[0], geo_model_test.structural_frame.structural_groups[0].elements[1]=\
    geo_model_test.structural_frame.structural_groups[0].elements[1], geo_model_test.structural_frame.structural_groups[0].elements[0]
    geo_model_test.structural_frame.structural_groups[0].elements[2], geo_model_test.structural_frame.structural_groups[0].elements[1]=\
    geo_model_test.structural_frame.structural_groups[0].elements[1], geo_model_test.structural_frame.structural_groups[0].elements[2]
   
    
    geo_model_test.structural_frame.structural_groups[0].elements[0], geo_model_test.structural_frame.structural_groups[0].elements[1]=\
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
    dimred=args.dimred
    cluster = args.cluster
    plot_dimred=args.plot_dimred
    prior_number_samples = args.prior_number_samples
    posterior_number_samples = args.posterior_number_samples
    posterior_warmup_steps = args.posterior_warmup_steps
    posterior_num_chain = args.posterior_num_chain
    directory_path = args.directory_path
    
    # Check if the directory exists
    if not os.path.exists(directory_path):
        # Create the directory if it does not exist
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' was created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
    
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
    SalinasA_gt = 7 - SalinasA_gt
    ######################################################################
    ## Arrange Data as concatationation of spacial co-ordinate and pixel values
    ###########################################################################
    H, W = SalinasA_gt.shape # get the shape of groud truth
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
    df_with_non_labelled_pixel = df_hsi.loc[(df_hsi['Label']!=7)]
    
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
    normalised_data = df_with_spectral_normalised.loc[(df_with_spectral_normalised["X"]>=startval)&(df_with_spectral_normalised["X"]<=endval)]
    normalised_hsi =torch.tensor(normalised_data.iloc[:,4:].to_numpy(), dtype=torch.float64)
    
    ## It is difficult to work with data in such a high dimensions, because the covariance matrix 
    ## determinant quickly goes to zero even if eigen-values are in the range of 1e-3. Therefore it is advisable 
    ## to fist apply dimensionality reduction to a lower dimensions
    if dimred=="pca":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=10)
        transformed_hsi = pca.fit_transform(normalised_hsi)
        normalised_hsi = torch.tensor(transformed_hsi, dtype=torch.float64)
        y_obs_label = torch.tensor(normalised_data.iloc[:,3].to_numpy(), dtype=torch.float64)
    if dimred =="tsne":
        #######################TODO#####################
        ################################################
        print("TSNE hasn't implemented for dimensionality reduction yet")
        exit()
    
    ###########################################################################
    ## Apply Classical clustering methods to find different cluster information our data
    ###########################################################################
    gm =gm = BayesianGaussianMixture(n_components=cluster, random_state=42).fit(normalised_hsi)
    
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
    if plot_dimred =="tsne":
        filename_tsne = directory_path + "/tsne_gmm_label.png"
        TSNE_transformation(data=normalised_data, label=gmm_label_rearranged, filename=filename_tsne)
    
    ######################################################################################
    ## Apply Classical clustering methods to find different cluster information our data
    ######################################################################################
  
    # geo_model_test = gp.create_geomodel(
    # project_name='Gempy_abc_Test',
    # extent=[0, 86, -10, 10, -83, 0],
    # resolution=[86,20,83],
    # refinement=3,
    # structural_frame= gp.data.StructuralFrame.initialize_default_structure()
    # )

    # gp.add_surface_points(
    #     geo_model=geo_model_test,
    #     x=[70.0, 80.0],
    #     y=[0.0, 0.0],
    #     z=[-77.0, -71.0],
    #     elements_names=['surface1', 'surface1']
    # )

    # gp.add_orientations(
    #     geo_model=geo_model_test,
    #     x=[75],
    #     y=[0.0],
    #     z=[-74],
    #     elements_names=['surface1'],
    #     pole_vector=[[-5/3, 0, 1]]
    # )
    # geo_model_test.update_transform(gp.data.GlobalAnisotropy.NONE)

    # element2 = gp.data.StructuralElement(
    #     name='surface2',
    #     color=next(geo_model_test.structural_frame.color_generator),
    #     surface_points=gp.data.SurfacePointsTable.from_arrays(
    #         x=np.array([20.0, 60.0]),
    #         y=np.array([0.0, 0.0]),
    #         z=np.array([-74, -52]),
    #         names='surface2'
    #     ),
    #     orientations=gp.data.OrientationsTable.initialize_empty()
    # )

    # geo_model_test.structural_frame.structural_groups[0].append_element(element2)

    # element3 = gp.data.StructuralElement(
    #     name='surface3',
    #     color=next(geo_model_test.structural_frame.color_generator),
    #     surface_points=gp.data.SurfacePointsTable.from_arrays(
    #         x=np.array([0.0, 30.0, 60]),
    #         y=np.array([0.0, 0.0,0.0]),
    #         z=np.array([-72, -55.5, -39]),
    #         names='surface3'
    #     ),
    #     orientations=gp.data.OrientationsTable.initialize_empty()
    # )

    # geo_model_test.structural_frame.structural_groups[0].append_element(element3)

    # element4 = gp.data.StructuralElement(
    #     name='surface4',
    #     color=next(geo_model_test.structural_frame.color_generator),
    #     surface_points=gp.data.SurfacePointsTable.from_arrays(
    #         x=np.array([0.0, 20.0, 60]),
    #         y=np.array([0.0, 0.0,0.0]),
    #         z=np.array([-61, -49, -27]),
    #         names='surface4'
    #     ),
    #     orientations=gp.data.OrientationsTable.initialize_empty()
    # )

    # geo_model_test.structural_frame.structural_groups[0].append_element(element4)

    # element5 = gp.data.StructuralElement(
    #     name='surface5',
    #     color=next(geo_model_test.structural_frame.color_generator),
    #     surface_points=gp.data.SurfacePointsTable.from_arrays(
    #         x=np.array([0.0, 20.0, 40]),
    #         y=np.array([0.0, 0.0, 0.0]),
    #         z=np.array([-39, -28, -16]),
    #         names='surface5'
    #     ),
    #     orientations=gp.data.OrientationsTable.initialize_empty()
    # )

    # geo_model_test.structural_frame.structural_groups[0].append_element(element5)

    # element6 = gp.data.StructuralElement(
    #     name='surface6',
    #     color=next(geo_model_test.structural_frame.color_generator),
    #     surface_points=gp.data.SurfacePointsTable.from_arrays(
    #         x=np.array([0.0, 20.0,30]),
    #         y=np.array([0.0, 0.0, 0.0]),
    #         z=np.array([-21, -10, -1]),
    #         names='surface6'
    #     ),
    #     orientations=gp.data.OrientationsTable.initialize_empty()
    # )

    # geo_model_test.structural_frame.structural_groups[0].append_element(element6)

    # geo_model_test.structural_frame.structural_groups[0].elements[0], geo_model_test.structural_frame.structural_groups[0].elements[1],\
    # geo_model_test.structural_frame.structural_groups[0].elements[2], geo_model_test.structural_frame.structural_groups[0].elements[3],\
    # geo_model_test.structural_frame.structural_groups[0].elements[4], geo_model_test.structural_frame.structural_groups[0].elements[5] = \
    # geo_model_test.structural_frame.structural_groups[0].elements[1], geo_model_test.structural_frame.structural_groups[0].elements[0],\
    # geo_model_test.structural_frame.structural_groups[0].elements[3], geo_model_test.structural_frame.structural_groups[0].elements[2],\
    # geo_model_test.structural_frame.structural_groups[0].elements[5], geo_model_test.structural_frame.structural_groups[0].elements[4]  


    # gp.compute_model(geo_model_test)
    # picture_test = gpv.plot_2d(geo_model_test, cell_number=5, legend='force')
    # plt.savefig("./Results/Prior_model.png")
    
    # Create initial model with higher refinement for better resolution and save it
    prior_filename= directory_path + "/prior_model.png"
    geo_model_test = create_initial_gempy_model(refinement=7,filename=prior_filename, save=True)
    # We can initialize again but with lower refinement because gempy solution are inddependent
    geo_model_test = create_initial_gempy_model(refinement=3,filename=prior_filename, save=False)
    
    # Label information need to be in same order as it is created in gempy model
    
    #y_obs_label = 7 - y_obs_label
    
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
    
    filename_initial_sp = directory_path + "/Initial_sp.csv"
    filename_initial_op = directory_path + "/Initial_op.csv"
    df_sp_init.to_csv(filename_initial_sp)
    df_or_init.to_csv(filename_initial_op)
    
    ################################################################################
    
    geo_model_test.transform.apply_inverse(sp_coords_copy_test)
    
    # Change the backend to PyTorch for probabilistic modeling
    BackendTensor.change_backend_gempy(engine_backend=gp.data.AvailableBackends.PYTORCH)
    
    # geo_model_test.interpolation_options.uni_degree = 0
    # geo_model_test.interpolation_options.mesh_extraction = False
    geo_model_test.interpolation_options.sigmoid_slope = 40
    
    factor=1 #0.01 , 100
    alpha = 100000
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
        prior_mean_surface_4 = sp_coords_copy_test[12, 2]

        mu_surface_1 = pyro.sample('mu_1', dist.Normal(prior_mean_surface_1, torch.tensor(0.2, dtype=torch.float64)))
        mu_surface_2 = pyro.sample('mu_2', dist.Normal(prior_mean_surface_2, torch.tensor(0.2, dtype=torch.float64)))
        mu_surface_3 = pyro.sample('mu_3', dist.Normal(prior_mean_surface_3, torch.tensor(0.1, dtype=torch.float64)))
        mu_surface_4 = pyro.sample('mu_4', dist.Normal(prior_mean_surface_4, torch.tensor(0.2, dtype=torch.float64)))
        #print(mu_surface_1, mu_surface_2)
        # Ensure that mu_surface_1 is greater than mu_surface_2
        # pyro.sample('mu_1 < 0', dist.Delta(torch.tensor(1.0, dtype=torch.float64)), obs=(mu_surface_1 < 3.7))
        # pyro.sample('mu_1 > mu_2', dist.Delta(torch.tensor(1.0, dtype=torch.float64)), obs=(mu_surface_1 > mu_surface_2))
        # pyro.sample('mu_2 > -38.5', dist.Delta(torch.tensor(1.0, dtype=torch.float64)), obs=(mu_surface_2 > 1.775))
        # pyro.sample('mu_3 < -38.5', dist.Delta(torch.tensor(1.0, dtype=torch.float64)), obs=(mu_surface_3 < 1.4))
        # pyro.sample('mu_3 > -61.5', dist.Delta(torch.tensor(1.0, dtype=torch.float64)), obs=(mu_surface_3 > 0.625))
        # pyro.sample('mu_4 < -61.5', dist.Delta(torch.tensor(1.0, dtype=torch.float64)), obs=(mu_surface_4 < 0.625))
        # pyro.sample('mu_4 > -83', dist.Delta(torch.tensor(1.0, dtype=torch.float64)), obs=(mu_surface_4 > - 0.2 ))
        
        
        pyro.sample('mu_1 < 0', dist.Delta(torch.tensor(1.0, dtype=torch.float64)), obs=(mu_surface_1 < 3.7))
        pyro.sample('mu_1 > mu_2', dist.Delta(torch.tensor(1.0, dtype=torch.float64)), obs=(mu_surface_1 > mu_surface_2))
        pyro.sample('mu_2 > mu_3', dist.Delta(torch.tensor(1.0, dtype=torch.float64)), obs=(mu_surface_2 > mu_surface_3))
        pyro.sample('mu_3 > mu_4', dist.Delta(torch.tensor(1.0, dtype=torch.float64)), obs=(mu_surface_3 > mu_surface_4))
        #pyro.sample('mu_3 > -61.5', dist.Delta(torch.tensor(1.0, dtype=torch.float64)), obs=(mu_surface_3 > 0.625))
        #pyro.sample('mu_4 < -61.5', dist.Delta(torch.tensor(1.0, dtype=torch.float64)), obs=(mu_surface_4 < 0.625))
        pyro.sample('mu_4 > -83', dist.Delta(torch.tensor(1.0, dtype=torch.float64)), obs=(mu_surface_4 > - 0.2 ))
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
        interpolation_input.surface_points.sp_coords = torch.index_put(
            interpolation_input.surface_points.sp_coords,
            (torch.tensor([12]), torch.tensor([2])),
            mu_surface_4
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
        #accuracy_intermediate = torch.sum(torch.round(custom_grid_values) == y_obs_label) / y_obs_label.shape[0]
        #store_accuracy.append(accuracy_intermediate)
        
        lambda_ = 15.0
        loc_mean = torch.tensor(mean_init,dtype=torch.float64)
        loc_cov =  torch.tensor(cov_init, dtype=torch.float64)
        cov_matrix = alpha * torch.eye(loc_mean[0].shape[0],dtype=torch.float64)
        z_nk = F.softmax(-lambda_* (torch.linspace(1,cluster,cluster, dtype=torch.float64) - custom_grid_values.reshape(-1,1))**2, dim=1)
        #class_label = torch.mean(F.softmax(-lambda_* (torch.linspace(1,cluster,cluster, dtype=torch.float64) - custom_grid_values.reshape(-1,1))**2, dim=1),dim=0)
        N_k = torch.sum(z_nk,axis=0)
        N = len(custom_grid_values)
        pi_k = N_k /N
        mean =[]
        cov =[]
        for i in range(loc_mean.shape[0]):
            #sample_data = pyro.sample("sample_data"+str(i+1), dist.MultivariateNormal(loc=loc_mean[i],covariance_matrix=loc_cov[i]))
            mean_data = pyro.sample("mean_data"+str(i+1), dist.MultivariateNormal(loc=loc_mean[i],covariance_matrix=cov_matrix))
            mean.append(mean_data)
            
        for i in range(loc_mean.shape[0]):
            cov_k = torch.zeros((loc_mean.shape[1],loc_mean.shape[1]),dtype=torch.float64)
            for j in range(z_nk.shape[0]):
                 cov_k +=  z_nk[j,i]* torch.matmul((obs_data[j,:] - mean[i]).reshape((-1,1)) ,(obs_data[j,:] - mean[i]).reshape((1,-1)))
            cov_k=cov_k/N_k[i] #+ 1e-3 * torch.diag(torch.ones(cov_k.shape[0],dtype=torch.float64))
            cov.append(cov_k)
        mean_tesnor = torch.stack(mean, dim=0)
        cov_tensor = torch.stack(cov,dim=0)
        #cov_likelihood = 5.0 * torch.eye(loc_cov[0].shape[0], dtype=torch.float64)
        
        with pyro.plate('N='+str(obs_data.shape[0]), obs_data.shape[0]):
            assignment = pyro.sample("assignment", dist.Categorical(pi_k))
            obs = pyro.sample("obs", dist.MultivariateNormal(loc=mean_tesnor[assignment],covariance_matrix= cov_tensor[assignment]), obs=obs_data)
            #obs = pyro.sample("obs", dist.MultivariateNormal(loc=sample_tesnor[assignment],covariance_matrix=factor * cov_matrix), obs=obs_data)
            
    
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
    mcmc = MCMC(nuts_kernel, num_samples=posterior_number_samples, warmup_steps=posterior_warmup_steps,num_chains=posterior_num_chain, disable_validation=False)
    mcmc.run(normalised_hsi)
    
    posterior_samples = mcmc.get_samples(group_by_chain=True)
    # sumarry = mcmc.summary()
    # filename_summarry = directory_path + "/sumarry.csv"
    # mcmc.write_csv()
    print(posterior_samples)
    exit()
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
    
    plt.figure(figsize=(8,10))
    az.plot_density(
        data=[data.posterior, data.prior],
        shade=.9,
        var_names=['mu_4'],
        data_labels=["Posterior Predictive", "Prior Predictive"],
        colors=[default_red, default_blue],
    )
    filename_mu_4 = directory_path + "/mu_4.png"
    plt.savefig(filename_mu_4)
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
    
    for i in range(posterior_samples["mu_1"].shape[0]):
        
        post_mu_1 = posterior_samples[keys_list[6]][i]
        post_mu_2 = posterior_samples[keys_list[7]][i]
        post_mu_3 = posterior_samples[keys_list[8]][i]
        post_mu_4 = posterior_samples[keys_list[9]][i]
        
        post_sample_data1 = posterior_samples[keys_list[0]][i]
        post_sample_data2 = posterior_samples[keys_list[1]][i]
        post_sample_data3 = posterior_samples[keys_list[2]][i]
        post_sample_data4 = posterior_samples[keys_list[3]][i]
        post_sample_data5 = posterior_samples[keys_list[4]][i]
        post_sample_data6 = posterior_samples[keys_list[5]][i]
        
        cov_tensor = torch.eye(post_sample_data1.shape[0],dtype=torch.float64)
        loc_mean = torch.tensor(mean_init,dtype=torch.float64)
        #loc_cov =  torch.tensor(cov_init, dtype=torch.float64)
        # Calculate the log probability of the value
        
        log_prior_geo = dist.Normal(prior_mean_surface_1, torch.tensor(0.2, dtype=torch.float64)).log_prob(post_mu_1)+\
                    dist.Normal(prior_mean_surface_2, torch.tensor(0.2, dtype=torch.float64)).log_prob(post_mu_2)+\
                    dist.Normal(prior_mean_surface_3, torch.tensor(0.1, dtype=torch.float64)).log_prob(post_mu_3)+\
                    dist.Normal(prior_mean_surface_4, torch.tensor(0.2, dtype=torch.float64)).log_prob(post_mu_4)
        
        # log_prior_hsi = likelihood = torch.exp(dist.MultivariateNormal(loc=loc_mean[0],covariance_matrix= cov_tensor).log_prob(post_sample_data1)) +\
        #                  torch.exp(dist.MultivariateNormal(loc=loc_mean[1],covariance_matrix= cov_tensor).log_prob(post_sample_data2))+\
        #                  torch.exp(dist.MultivariateNormal(loc=loc_mean[2],covariance_matrix= cov_tensor).log_prob(post_sample_data3)) +\
        #                  torch.exp(dist.MultivariateNormal(loc=loc_mean[3],covariance_matrix= cov_tensor).log_prob(post_sample_data4)) +\
        #                  torch.exp(dist.MultivariateNormal(loc=loc_mean[4],covariance_matrix= cov_tensor).log_prob(post_sample_data5)) +\
        #                  torch.exp(dist.MultivariateNormal(loc=loc_mean[5],covariance_matrix= cov_tensor).log_prob(post_sample_data6)) 
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
        interpolation_input.surface_points.sp_coords = torch.index_put(
            interpolation_input.surface_points.sp_coords,
            (torch.tensor([12]), torch.tensor([2])),
            post_mu_4
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
        accuracy_intermediate = torch.sum(torch.round(custom_grid_values) == y_obs_label) / y_obs_label.shape[0]
        store_accuracy.append(accuracy_intermediate)
        lambda_ = 15.0
        loc_mean = torch.tensor(mean_init,dtype=torch.float64)
        #loc_cov =  torch.tensor(cov_init, dtype=torch.float64)
        cov_matrix = alpha * torch.eye(loc_mean[0].shape[0],dtype=torch.float64)
        z_nk = F.softmax(-lambda_* (torch.tensor([1,2,3,4,5,6], dtype=torch.float64) - custom_grid_values.reshape(-1,1))**2, dim=1)
        #pi_k = torch.mean(F.softmax(-lambda_* (torch.linspace(1,cluster,cluster, dtype=torch.float64) - custom_grid_values.reshape(-1,1))**2, dim=1),dim=0)
        
    
        # log_prior_hsi = dist.MultivariateNormal(loc=loc_mean[0],covariance_matrix=loc_cov[0]).log_prob(post_sample_data1)+\
        #                 dist.MultivariateNormal(loc=loc_mean[1],covariance_matrix=loc_cov[1]).log_prob(post_sample_data2)+\
        #                 dist.MultivariateNormal(loc=loc_mean[2],covariance_matrix=loc_cov[2]).log_prob(post_sample_data3)+\
        #                 dist.MultivariateNormal(loc=loc_mean[3],covariance_matrix=loc_cov[3]).log_prob(post_sample_data4)+\
        #                 dist.MultivariateNormal(loc=loc_mean[4],covariance_matrix=loc_cov[4]).log_prob(post_sample_data5)+\
        #                 dist.MultivariateNormal(loc=loc_mean[5],covariance_matrix=loc_cov[5]).log_prob(post_sample_data6)
        
        log_prior_hsi = dist.MultivariateNormal(loc=loc_mean[0],covariance_matrix=cov_matrix).log_prob(post_sample_data1)+\
                        dist.MultivariateNormal(loc=loc_mean[1],covariance_matrix=cov_matrix).log_prob(post_sample_data2)+\
                        dist.MultivariateNormal(loc=loc_mean[2],covariance_matrix=cov_matrix).log_prob(post_sample_data3)+\
                        dist.MultivariateNormal(loc=loc_mean[3],covariance_matrix=cov_matrix).log_prob(post_sample_data4)+\
                        dist.MultivariateNormal(loc=loc_mean[4],covariance_matrix=cov_matrix).log_prob(post_sample_data5)+\
                        dist.MultivariateNormal(loc=loc_mean[5],covariance_matrix=cov_matrix).log_prob(post_sample_data6)
        
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
            likelihood = pi_k[0] *torch.exp(dist.MultivariateNormal(loc=post_sample_data1,covariance_matrix= cov_tensor[0]).log_prob(normalised_hsi[j])) +\
                         pi_k[1] *torch.exp(dist.MultivariateNormal(loc=post_sample_data2,covariance_matrix= cov_tensor[1]).log_prob(normalised_hsi[j]))+\
                         pi_k[2] *torch.exp(dist.MultivariateNormal(loc=post_sample_data3,covariance_matrix= cov_tensor[2]).log_prob(normalised_hsi[j])) +\
                         pi_k[2] *torch.exp(dist.MultivariateNormal(loc=post_sample_data4,covariance_matrix= cov_tensor[3]).log_prob(normalised_hsi[j])) +\
                         pi_k[4] *torch.exp(dist.MultivariateNormal(loc=post_sample_data5,covariance_matrix= cov_tensor[4]).log_prob(normalised_hsi[j])) +\
                         pi_k[5] *torch.exp(dist.MultivariateNormal(loc=post_sample_data6,covariance_matrix= cov_tensor[5]).log_prob(normalised_hsi[j])) 
                         
            # likelihood = pi_k[0] *torch.exp(dist.MultivariateNormal(loc=post_sample_data1,covariance_matrix= factor * cov_matrix).log_prob(normalised_hsi[j])) +\
            #              pi_k[1] *torch.exp(dist.MultivariateNormal(loc=post_sample_data2,covariance_matrix= factor * cov_matrix).log_prob(normalised_hsi[j]))+\
            #              pi_k[2] *torch.exp(dist.MultivariateNormal(loc=post_sample_data3,covariance_matrix= factor * cov_matrix).log_prob(normalised_hsi[j])) +\
            #              pi_k[2] *torch.exp(dist.MultivariateNormal(loc=post_sample_data4,covariance_matrix= factor * cov_matrix).log_prob(normalised_hsi[j])) +\
            #              pi_k[4] *torch.exp(dist.MultivariateNormal(loc=post_sample_data5,covariance_matrix= factor * cov_matrix).log_prob(normalised_hsi[j])) +\
            #              pi_k[5] *torch.exp(dist.MultivariateNormal(loc=post_sample_data6,covariance_matrix= factor * cov_matrix).log_prob(normalised_hsi[j])) 
            log_likelihood += torch.log(likelihood)
        
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
    filename_sample_accuracy = directory_path + "/accuracy.png"
    plt.savefig(filename_sample_accuracy)
    
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
    
    mu_1_post = posterior_samples["mu_1"][MAP_sample_index]
    mu_2_post = posterior_samples["mu_2"][MAP_sample_index]
    mu_3_post = posterior_samples["mu_3"][MAP_sample_index]
    mu_4_post = posterior_samples["mu_4"][MAP_sample_index]
    
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
    interpolation_input.surface_points.sp_coords = torch.index_put(
            interpolation_input.surface_points.sp_coords,
            (torch.tensor([12]), torch.tensor([2])),
            mu_4_post
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
            z=np.array([sp_cord[12,2], -52]),
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
            z=np.array([-61, sp_cord[7,2], -27]),
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
            z=np.array([-39, sp_cord[4,2], -16]),
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
            z=np.array([-21, sp_cord[1,2], -1]),
            names='surface6'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    geo_model_test_post.structural_frame.structural_groups[0].append_element(element6)

    geo_model_test_post.structural_frame.structural_groups[0].elements[0], geo_model_test_post.structural_frame.structural_groups[0].elements[1]=\
    geo_model_test_post.structural_frame.structural_groups[0].elements[1], geo_model_test_post.structural_frame.structural_groups[0].elements[0]
    geo_model_test_post.structural_frame.structural_groups[0].elements[2], geo_model_test_post.structural_frame.structural_groups[0].elements[1]=\
    geo_model_test_post.structural_frame.structural_groups[0].elements[1], geo_model_test_post.structural_frame.structural_groups[0].elements[2]
    geo_model_test_post.structural_frame.structural_groups[0].elements[3], geo_model_test_post.structural_frame.structural_groups[0].elements[2]=\
    geo_model_test_post.structural_frame.structural_groups[0].elements[2], geo_model_test_post.structural_frame.structural_groups[0].elements[3]
    geo_model_test_post.structural_frame.structural_groups[0].elements[4], geo_model_test_post.structural_frame.structural_groups[0].elements[3]=\
    geo_model_test_post.structural_frame.structural_groups[0].elements[3], geo_model_test_post.structural_frame.structural_groups[0].elements[4]
    geo_model_test_post.structural_frame.structural_groups[0].elements[5], geo_model_test_post.structural_frame.structural_groups[0].elements[4]=\
    geo_model_test_post.structural_frame.structural_groups[0].elements[4], geo_model_test_post.structural_frame.structural_groups[0].elements[5]
    
    geo_model_test_post.structural_frame.structural_groups[0].elements[0], geo_model_test_post.structural_frame.structural_groups[0].elements[1]=\
    geo_model_test_post.structural_frame.structural_groups[0].elements[1], geo_model_test_post.structural_frame.structural_groups[0].elements[0]
    geo_model_test_post.structural_frame.structural_groups[0].elements[2], geo_model_test_post.structural_frame.structural_groups[0].elements[1]=\
    geo_model_test_post.structural_frame.structural_groups[0].elements[1], geo_model_test_post.structural_frame.structural_groups[0].elements[2]
    geo_model_test_post.structural_frame.structural_groups[0].elements[3], geo_model_test_post.structural_frame.structural_groups[0].elements[2]=\
    geo_model_test_post.structural_frame.structural_groups[0].elements[2], geo_model_test_post.structural_frame.structural_groups[0].elements[3]
    geo_model_test_post.structural_frame.structural_groups[0].elements[4], geo_model_test_post.structural_frame.structural_groups[0].elements[3]=\
    geo_model_test_post.structural_frame.structural_groups[0].elements[3], geo_model_test_post.structural_frame.structural_groups[0].elements[4]

    geo_model_test_post.structural_frame.structural_groups[0].elements[0], geo_model_test_post.structural_frame.structural_groups[0].elements[1]=\
    geo_model_test_post.structural_frame.structural_groups[0].elements[1], geo_model_test_post.structural_frame.structural_groups[0].elements[0]
    geo_model_test_post.structural_frame.structural_groups[0].elements[2], geo_model_test_post.structural_frame.structural_groups[0].elements[1]=\
    geo_model_test_post.structural_frame.structural_groups[0].elements[1], geo_model_test_post.structural_frame.structural_groups[0].elements[2]
    geo_model_test_post.structural_frame.structural_groups[0].elements[3], geo_model_test_post.structural_frame.structural_groups[0].elements[2]=\
    geo_model_test_post.structural_frame.structural_groups[0].elements[2], geo_model_test_post.structural_frame.structural_groups[0].elements[3]

    geo_model_test_post.structural_frame.structural_groups[0].elements[0], geo_model_test_post.structural_frame.structural_groups[0].elements[1]=\
    geo_model_test_post.structural_frame.structural_groups[0].elements[1], geo_model_test_post.structural_frame.structural_groups[0].elements[0]
    geo_model_test_post.structural_frame.structural_groups[0].elements[2], geo_model_test_post.structural_frame.structural_groups[0].elements[1]=\
    geo_model_test_post.structural_frame.structural_groups[0].elements[1], geo_model_test_post.structural_frame.structural_groups[0].elements[2]
   
    geo_model_test_post.structural_frame.structural_groups[0].elements[0], geo_model_test_post.structural_frame.structural_groups[0].elements[1]=\
    geo_model_test_post.structural_frame.structural_groups[0].elements[1], geo_model_test_post.structural_frame.structural_groups[0].elements[0]
    

    gp.set_custom_grid(geo_model_test_post.grid, xyz_coord=xyz_coord)
    gp.compute_model(geo_model_test_post)
    
    custom_grid_values_post = geo_model_test_post.solutions.octrees_output[0].last_output_center.custom_grid_values
    ####################################TODO#################################################
    #   Try to find the final accuracy to check if it has improved the classification
    #########################################################################################
    accuracy_final = torch.sum(torch.round(torch.tensor(custom_grid_values_post)) == y_obs_label) / y_obs_label.shape[0]
    print("accuracy_init: ", accuracy_init , "accuracy_final: ", accuracy_final)
    
    picture_test_post = gpv.plot_2d(geo_model_test_post, cell_number=5, legend='force')
    filename_posterior_model = directory_path + "/Posterior_model.png"
    plt.savefig(filename_posterior_model)
    if plot_dimred=="tsne":
        #TSNE_transformation(data=normalised_data, label= 7- np.round(custom_grid_values_post), filename="./Results/tsne_gempy_final_label.png")
        filename_tsne_final_label = directory_path + "/tsne_gempy_final_label.png"
        TSNE_transformation(data=normalised_data, label= np.round(custom_grid_values_post), filename=filename_tsne_final_label)
    
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