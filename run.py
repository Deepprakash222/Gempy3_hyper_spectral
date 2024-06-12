import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import arviz as az

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


    
def main():
    
    # Load .mat file
    SalinasA= np.array(scipy.io.loadmat('./HSI_Salinas/SalinasA.mat')['salinasA'])
    SalinasA_corrected= np.array(scipy.io.loadmat('./HSI_Salinas/SalinasA_corrected.mat')['salinasA_corrected'])
    SalinasA_gt= np.array(scipy.io.loadmat('./HSI_Salinas/SalinasA_gt.mat')['salinasA_gt'])
    # Arrange the label in groundtruth
    i=0
    label_data = [0,6,1,5,4,3,2]
    print(np.unique(SalinasA_gt))
    for ele in np.unique(SalinasA_gt):
        mask = SalinasA_gt==ele
        SalinasA_gt[mask] = label_data[i]
        i=i+1
    
    ###### TODO################################################################
    ## Arrange Data as concatationation of spacial co-ordinate and pixel values
    ###########################################################################
    column = 20
    y_obs = torch.tensor(SalinasA_gt[:,column], dtype=torch.float64)
    mask = y_obs!=0
    y_obs_label = y_obs[mask]
    y_obs_hsi = torch.tensor(SalinasA_corrected[:,column,:], dtype=torch.float64)[mask]
    print(y_obs_hsi.shape)
    
if __name__ == "__main__":
    main()
