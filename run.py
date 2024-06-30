import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import arviz as az
import pandas as pd
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import KMeans



def main(startval,endval,dimred):
    '''
    This function defines a model which uses hyperspectral data, applies clustering methods to find cluster information and then uses Bayesian Modelling to model a posterior.
    The arguments passed to main are:
    startval: It defines the starting X column value of Hyperspectral data.
    endval: It defines the end X column value of Hyperspectral data.
    dimred: Defines the type of dimensionality reduction user wants to follow .Currently either pca or tsne.
    '''
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
    df_with_non_labelled_pixel = df_hsi.loc[(df_hsi['Label']!=0)]
    
    # Normalise along the spectral lines 
    df_with_spectral_normalised = df_with_non_labelled_pixel
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
    y_obs_label = torch.tensor(normalised_data.iloc[:,3].to_numpy(), dtype=torch.float64)
    
    ###########################################################################
    ## Apply Classical clustering methods to find different cluster information our data
    ###########################################################################
    gm =gm = BayesianGaussianMixture(n_components=6, random_state=0).fit(normalised_hsi)
    
    # make the labels to start with 1 instead of 0
    gmm_label = gm.predict(normalised_hsi) +1 
    
    # reaarange the label information so it is would be consistent with ground truth label
    gmm_label2 = torch.zeros_like(y_obs_label)
    gmm_label2[gmm_label==2]=6
    gmm_label2[gmm_label==4]=5
    gmm_label2[gmm_label==1]=4
    gmm_label2[gmm_label==3]=3
    gmm_label2[gmm_label==6]=2
    gmm_label2[gmm_label==5]=1
    
    # rearrange the mean and covariance accordingly too
    mean_init, cov_init = gm.means_[[4,5,2,0,3,1]], gm.covariances_[[4,5,2,0,3,1]]
    #################################TODO##################################################
    ## Apply different dimentionality reduction techniques and save the plot in Result file
    #######################################################################################
    if(dimred=="pca"):
        # Separate spatial coordinates and features
        spatial_coords = normalised_data.iloc[:,0:3]
        features = normalised_data.iloc[:,4:]
        # Normalize the features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)

        # Apply PCA
        pca = PCA(n_components=0.95,svd_solver="full")  # Keep 95% of variance
        features_pca = pca.fit_transform(features_normalized)
        print(f"Number of components kept: {pca.n_components_}")
        print(f"Cumulative explained variance: {cumulative_variance_ratio[-1]:.4f}")
        plt.figure(figsize=(10, 8))
        plt.scatter(features_pca[:, 0], features_pca[:, 1], c=normalised_data.iloc[:,3].to_numpy(), cmap='viridis')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('PCA with 0.96 variance accounted by 2 PCs')
        plt.colorbar(label='Output Label')
        #plt.show()
        plt.savefig('./Results/PCA_redn.png')
    if(dimred=="tsne"):
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(normalised_data.iloc[:,4:])
        #tsne.kl_divergence_
        plt.figure(figsize=(10, 8))
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=normalised_data.iloc[:,3].to_numpy(), cmap='viridis')
        plt.title('TSNE Reduction from 204 to 2 components')
        plt.colorbar(label='Output Label')
        #plt.show()
        plt.savefig('./Results/TSNE_redn.png')
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
    geo_model_test.interpolation_options.sigmoid_slope = 30
    
    
    @config_enumerate
    def model_test(obs_data):
        """
        This Pyro model represents the probabilistic aspects of the geological model.
        It defines a prior distribution for the top layer's location and
        computes the thickness of the geological layer as an observed variable.
        """
        # Define prior for the top layer's location
        prior_mean_surface_1 = sp_coords_copy_test[11, 2]
        prior_mean_surface_2 = sp_coords_copy_test[14, 2]
        prior_mean_surface_3 = sp_coords_copy_test[5, 2]
        prior_mean_surface_4 = sp_coords_copy_test[0, 2]

        mu_surface_1 = pyro.sample('mu_1', dist.Normal(prior_mean_surface_1, torch.tensor(0.2, dtype=torch.float64)))
        mu_surface_2 = pyro.sample('mu_2', dist.Normal(prior_mean_surface_2, torch.tensor(0.2, dtype=torch.float64)))
        mu_surface_3 = pyro.sample('mu_3', dist.Normal(prior_mean_surface_3, torch.tensor(0.2, dtype=torch.float64)))
        mu_surface_4 = pyro.sample('mu_4', dist.Normal(prior_mean_surface_4, torch.tensor(0.2, dtype=torch.float64)))
        #print(mu_surface_1, mu_surface_2)
        # Ensure that mu_surface_1 is greater than mu_surface_2
        pyro.sample('mu_1 < 0', dist.Delta(torch.tensor(1.0, dtype=torch.float64)), obs=(mu_surface_1 < 3.7))
        pyro.sample('mu_1 > mu_2', dist.Delta(torch.tensor(1.0, dtype=torch.float64)), obs=(mu_surface_1 > mu_surface_2))
        pyro.sample('mu_2 > -38.5', dist.Delta(torch.tensor(1.0, dtype=torch.float64)), obs=(mu_surface_2 > 1.775))
        pyro.sample('mu_3 < -38.5', dist.Delta(torch.tensor(1.0, dtype=torch.float64)), obs=(mu_surface_3 < 1.4))
        pyro.sample('mu_3 > -61.5', dist.Delta(torch.tensor(1.0, dtype=torch.float64)), obs=(mu_surface_3 > 0.625))
        pyro.sample('mu_4 < -61.5', dist.Delta(torch.tensor(1.0, dtype=torch.float64)), obs=(mu_surface_4 < 0.625))
        pyro.sample('mu_4 > -83', dist.Delta(torch.tensor(1.0, dtype=torch.float64)), obs=(mu_surface_4 > - 0.2 ))
        # Update the model with the new top layer's location
        interpolation_input = geo_model_test.interpolation_input
        
        
        interpolation_input.surface_points.sp_coords = torch.index_put(
            interpolation_input.surface_points.sp_coords,
            (torch.tensor([11]), torch.tensor([2])),
            mu_surface_1
        )
        interpolation_input.surface_points.sp_coords = torch.index_put(
            interpolation_input.surface_points.sp_coords,
            (torch.tensor([14]), torch.tensor([2])),
            mu_surface_2
        )
        
        interpolation_input.surface_points.sp_coords = torch.index_put(
            interpolation_input.surface_points.sp_coords,
            (torch.tensor([5]), torch.tensor([2])),
            mu_surface_3
        )
        interpolation_input.surface_points.sp_coords = torch.index_put(
            interpolation_input.surface_points.sp_coords,
            (torch.tensor([0]), torch.tensor([2])),
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
        lambda_ = 10.0
        loc_mean = torch.tensor(mean_init,dtype=torch.float64)
        loc_cov =  torch.tensor(cov_init, dtype=torch.float64)
        class_label = F.softmax(-lambda_* (torch.tensor([1,2,3,4,5,6], dtype=torch.float64) - custom_grid_values.reshape(-1,1))**2, dim=1)
        sample =[]
        for i in range(loc_mean.shape[0]):
            sample_data = pyro.sample("sample_data"+str(i+1), dist.MultivariateNormal(loc=loc_mean[i],covariance_matrix=loc_cov[i]))
            sample.append(sample_data)
        sample_tesnor = torch.stack(sample, dim=0)
        
        with pyro.plate('N='+str(obs_data.shape[0]), obs_data.shape[0]):
            assignment = pyro.sample("assignment", dist.Categorical(class_label))
            obs = pyro.sample("obs", dist.MultivariateNormal(loc=sample_tesnor[assignment],covariance_matrix=loc_cov[assignment]), obs=obs_data)
        
    ################################################################################
    # Prior
    ################################################################################
    pyro.set_rng_seed(42)
    prior = Predictive(model_test, num_samples=10)(normalised_hsi)
    # Key to avoid
    avoid_key = ['mu_1 < 0','mu_1 > mu_2','mu_2 > -38.5', 'mu_3 < -38.5','mu_3 > -61.4','mu_4 < -61.5', 'mu_4 > -83']
    # Create sub-dictionary without the avoid_key
    prior = dict((key, value) for key, value in prior.items() if key not in avoid_key)
    data = az.from_pyro(prior=prior)
    
    ################################################################################
    # Posterior 
    ################################################################################
    pyro.primitives.enable_validation(is_validate=True)
    nuts_kernel = NUTS(model_test, step_size=0.0085, adapt_step_size=True, target_accept_prob=0.9, max_tree_depth=10, init_strategy=init_to_mean)
    mcmc = MCMC(nuts_kernel, num_samples=100, warmup_steps=50, disable_validation=False)
    mcmc.run(normalised_hsi)
    
    posterior_samples = mcmc.get_samples()
    posterior_predictive = Predictive(model_test, posterior_samples)(normalised_hsi)
    data = az.from_pyro(posterior=mcmc, prior=prior, posterior_predictive=posterior_predictive)
    
    ################################################################################
    #  Try Plot the data and save it as file in output folder
    ################################################################################
    mu_1_mean = posterior_samples["mu_1"].mean()
    mu_2_mean = posterior_samples["mu_2"].mean()
    mu_3_mean = posterior_samples["mu_3"].mean()
    mu_4_mean = posterior_samples["mu_4"].mean()
    
    # # Update the model with the new top layer's location
    interpolation_input = geo_model_test.interpolation_input
    interpolation_input.surface_points.sp_coords = torch.index_put(
        interpolation_input.surface_points.sp_coords,
        (torch.tensor([11]), torch.tensor([2])),
        posterior_samples["mu_1"].mean()
    )
    interpolation_input.surface_points.sp_coords = torch.index_put(
        interpolation_input.surface_points.sp_coords,
        (torch.tensor([14]), torch.tensor([2])),
        posterior_samples["mu_2"].mean()
    )

    interpolation_input.surface_points.sp_coords = torch.index_put(
            interpolation_input.surface_points.sp_coords,
            (torch.tensor([5]), torch.tensor([2])),
            posterior_samples["mu_3"].mean()
        )
    interpolation_input.surface_points.sp_coords = torch.index_put(
            interpolation_input.surface_points.sp_coords,
            (torch.tensor([0]), torch.tensor([2])),
            posterior_samples["mu_4"].mean()
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
    picture_test_post = gpv.plot_2d(geo_model_test_post, cell_number=5, legend='force')
    plt.savefig("./Results/Posterior_model.png")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pass values using command line')
    parser.add_argument('--startval', metavar='startcol', required=True, type=int, help='start x column value')
    parser.add_argument('--endval', metavar='endcol', required=True, type=int, help='end x column value')
    parser.add_argument('--dimred', metavar='dimred', type=str, help='type of dimensionality reduction')
    args = parser.parse_args()
    main(startval=args.startval,endval=args.endval,dimred=args.dimred)
