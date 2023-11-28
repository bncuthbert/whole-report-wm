#!/usr/bin/env python

"""Contains model fitting functions"""

import numpy as np
import pandas as pd 
import pymc3 as pm
import pymc3.distributions.transforms as tr
import scipy.stats as stats
from scipy import optimize
from sklearn.utils import resample
from astropy.stats import circmean
from tqdm.autonotebook import tqdm
from tenacity import retry
import sys
import logging
import pickle

from utils import min_angle, max_kde, vonmises_pdf

SAVE_DIR = '../models/' # relative to notebook folder for convenience

# model loading utils

def load_hbm_fit(modality, condition, dataset, set_size, kappa_obs):
    save_path = f"{SAVE_DIR}{modality}/{condition}/{dataset}/"
    fit_name = f"hbm_ss_{set_size}_k_{kappa_obs}"
    save_name = f'{save_path}{fit_name}.pickle'

    try:
        with open(save_name, 'rb') as p:
            df_fit = pickle.load(p)
        return df_fit
    except:
        print(f'Failed to load {save_name}')

def load_bfmm_fit(modality, condition, dataset, set_size, kappa_obs, K):
    save_path = f"{SAVE_DIR}{modality}/{condition}/{dataset}/"
    fit_name = f"bfmm{K}_ss_{set_size}_k_{kappa_obs}"
    save_name = f'{save_path}{fit_name}.pickle'

    try:
        with open(save_name, 'rb') as p:
            df_fit = pickle.load(p)
        return df_fit
    except:
        print(f'Failed to load {save_name}')


# VM and sine fitting

def fit_vm_by_sub(df, var='error_rad'):
    """Iterates through all subjects in a dataframe and uses MLE to fit a von Mises
    to the distribution of specified var.

    Args:
        df (pandas.DataFrame): Dataframe containing columns 'subject' and var. 
        var (string): Column name to fit von Mises to. Should reference a column
                      containing angular values in radians.

    Returns:
        df_fit (pandas.DataFrame): Dataframe of results with columns for 'subject', 
        'mu_hat', and 'kappa_hat'
    """    
    sub_ids = list(np.sort(df.subject.unique()))
    pbar = tqdm(sub_ids, file=sys.stdout)

    mus = []
    kappas = []

    for sub in pbar:
        x = df[df.subject == sub][var].dropna()
        kappa, mu, _ = stats.vonmises.fit(x, fscale=1)
        kappas.append(kappa)
        mus.append(mu)

        pbar.set_description(f'sub {sub} complete')

    df_fit = pd.DataFrame({'subject': sub_ids,
                           'mu_hat': mus,
                           'kappa_hat': kappas})

    return df_fit

def phase_locked_sine(x, a):
    return a * np.sin(x)

def fit_phase_locked_sine(x, y):
    a0 = np.random.uniform(low=min(y), high=max(y))
    params, _ = optimize.curve_fit(phase_locked_sine, x, y, p0=a0)
    return params[0]

def bootstrap_sine_fit(data, var, bias_toward, n_iters=1000):
    """
    """
    X, Y = compute_relative_error(data, var=var, bias_toward=bias_toward)
    
    samples = []
    for i in range(n_iters):
            sample_inds = resample(range(len(X)))
            x_i = np.array(X)[sample_inds]
            y_i = np.array(Y)[sample_inds]
            samples.append(fit_phase_locked_sine(x_i, y_i))
    
    return pd.DataFrame({'samples': np.hstack(samples)})

def bootstrap_sine_fit_by_sub(data, var, bias_toward, n_iters=1000):
    sub_ids = list(np.sort(data.subject.unique()))
    pbar = tqdm(sub_ids, file=sys.stdout)

    df_list = []
    for sub in pbar:
        df_temp = data[data['subject'] == sub]
        df_bootstrap = bootstrap_sine_fit(df_temp, var=var, bias_toward=bias_toward,
                                          n_iters=n_iters)
        df_bootstrap['subject'] = sub
        df_list.append(df_bootstrap)

        pbar.set_description(f'sub {sub} complete')

    return pd.concat(df_list).reset_index(drop=True)


# HBM & BFMM utils

@retry
def sample_model(model, observed, n_samples=5000, tune=2000, target_accept=0.95, 
                 return_inferencedata=True, **kwargs):
    with model:
        pm.set_data({"x_obs": observed})
        trace = pm.sample(n_samples, 
                          tune=tune, 
                          target_accept=target_accept,
                          return_inferencedata=return_inferencedata,
                          **kwargs)

    return trace

def get_noisy_observations(x, kappa_obs):
    return np.random.vonmises(x, kappa_obs)

def compute_bias_stats(data, var, bias_toward, bins=36, presented='presented_rad'):
    relative_mean, relative_error = compute_relative_error(data, var=var, 
                                                           bias_toward=bias_toward)

    # sort
    ind = np.argsort(relative_mean)
    relative_mean_sorted = np.array(relative_mean)[ind]
    error_sorted = np.array(relative_error)[ind]

    mean, bin_edges, _ = stats.binned_statistic(relative_mean_sorted, 
                                                error_sorted, statistic='mean', bins=bins)
    
    std, bin_edges, _ = stats.binned_statistic(relative_mean_sorted, 
                                               error_sorted, statistic='std', bins=bins)

    x = bin_edges[:-1]

    return x, mean, std

def compute_relative_error(data, var, bias_toward, presented='presented_rad'):
    relative_mean = min_angle(data[presented], data[bias_toward], radians=True)
    relative_error = min_angle(data[presented], data[var], radians=True)

    return relative_mean, relative_error


# Hierarchical Bayesian Model (HBM; two levels)

def get_hbm(set_size, kappa_obs, kappa_prior=[0, 100]):

    hbm = pm.Model()
    with hbm:
        # priors
        mu = pm.Uniform("mu", lower=-np.pi, upper=np.pi)
        kappa = pm.Uniform("kappa", lower=kappa_prior[0], upper=kappa_prior[1])

        # latent stimuli
        theta = pm.VonMises("theta", mu=mu, kappa=kappa, shape=set_size)

        # noisy observations
        x_shared = pm.Data("x_obs", np.random.vonmises(0, 1, set_size))
        x = pm.VonMises("x", mu=theta, kappa=kappa_obs, observed=x_shared)

    return hbm

def get_hbm_estimates(trace):
    stack_dims = ("chain", "draw")
    theta_samples = trace['posterior']['theta'].stack(z=stack_dims)
    mu_samples = trace['posterior']['mu'].stack(z=stack_dims)
    kappa_samples = trace['posterior']['kappa'].stack(z=stack_dims)

    theta_mean = circmean(np.array(theta_samples), axis=1)
    mu_mean = circmean(np.array(mu_samples))
    kappa_mean = np.array(kappa_samples).mean()

    return {'theta': theta_mean, 'mu': mu_mean, 'kappa':kappa_mean}

def fit_hbm(df, model, kappa_obs, log_level=logging.CRITICAL):
    # set logging level for clean progress bar
    logger = logging.getLogger('pymc3')
    logger.setLevel(log_level)

    # add empty columns to df
    df.reset_index(inplace=True, drop=True)
    presented_cols = [col for col in df.columns if 'presented_rad' in col]
    set_size = len(presented_cols)
    obs_cols =  [f'obs_{i}' for i in (np.arange(set_size) + 1)]
    df.loc[:, obs_cols] = np.nan
    df.loc[:, ['mu_hat', 'kappa_hat']] = np.nan
    theta_cols = [f'theta_hat_{i}' for i in (np.arange(set_size) + 1)]
    df.loc[:, theta_cols] = np.nan
    
    pbar = tqdm(range(len(df)))
    for row in pbar:
        pbar.set_description(f'fitting trial {row + 1}...')

        presented = np.array(df.loc[row, presented_cols], dtype='float64')
        observed = get_noisy_observations(presented, kappa_obs)
        df.loc[row, obs_cols] = observed

        trace = sample_model(model, observed, progressbar=False)
        results = get_hbm_estimates(trace)

        df.loc[row, 'mu_hat'] = results['mu']
        df.loc[row, 'kappa_hat'] = results['kappa']
        df.loc[row, theta_cols] = results['theta']

    logger.setLevel(logging.INFO)
    
    return df


# Bayesian Finite Mixture Model (BFMM)

def get_bfmm(set_size, K, kappa_obs):

    bfmm = pm.Model()
    with bfmm:
        # component weights
        w = pm.Dirichlet("w", np.ones(K))

        # component means
        mu = pm.Uniform("mu", lower=-np.pi, upper=np.pi, shape=K, 
                        transform=tr.Ordered(), testval=np.linspace(0, 1, K))

        # component precision
        tau = pm.Gamma("tau", 1.0, 1.0, shape=K)
        lambda_ = pm.Gamma("lambda_", 10.0, 1.0, shape=K)

        vm_comps = pm.VonMises.dist(mu=mu, kappa=lambda_ * tau)

        # latent stimuli
        theta = pm.Mixture("theta", w, vm_comps, shape=set_size, transform=tr.circular) 

        # noisy observations
        x_shared = pm.Data("x_obs", np.random.vonmises(0, 1, set_size))
        x = pm.VonMises("x", mu=theta, kappa=kappa_obs, observed=x_shared)
    
    return bfmm

def get_bfmm_estimates(trace, mu_mode=True, kappa_mode=True):
    stack_dims = ("chain", "draw")
    
    w_samples = trace['posterior']['w'].stack(z=stack_dims)
    theta_samples = trace['posterior']['theta'].stack(z=stack_dims)
    mu_samples = trace['posterior']['mu'].stack(z=stack_dims)
    tau_samples = trace['posterior']['tau'].stack(z=stack_dims)
    lambda_samples = trace['posterior']['lambda_'].stack(z=stack_dims)
    kappa_samples = tau_samples.values * lambda_samples.values
    
    K = kappa_samples.shape[0] # number of components
    
    # weight estimate
    w_hat = np.array(w_samples).mean(axis=1)
    
    # latent variable estimate
    theta_hat = circmean(np.array(theta_samples), axis=1)
    
    # mean estimate
    if mu_mode:
        mu_hat = np.empty(K)
        for k in range(K):
            mu_hat[k] = max_kde(x=np.linspace(-np.pi, np.pi, 1000), 
                                   y=mu_samples.values[k])
    else:
        mu_hat = circmean(np.array(mu_samples), axis=1)
    
    # precision estimate
    if kappa_mode:
        kappa_hat = np.empty(K)
        for k in range(K):
            kappa_hat[k] = max_kde(x=np.linspace(0, 100, 1000), 
                                   y=kappa_samples[k])
    else:
        kappa_hat = np.median(kappa_samples, axis=1)
    
    # cluster assignment
    unweighted = vonmises_pdf(np.atleast_3d(theta_hat), 
                              mu_hat, 
                              kappa_hat)
    cluster_density = np.squeeze(unweighted) * w_hat
    cluster_assignment = np.argmax(cluster_density, axis=1)
    
    estimates = {'w': w_hat, 
                 'theta': theta_hat, 
                 'mu': mu_hat, 
                 'kappa':kappa_hat,
                 'cluster':cluster_assignment}
    
    return estimates

def fit_bfmm(df, model, kappa_obs, n_attempts=3, log_level=logging.CRITICAL):
    # set logging level for clean progress bar
    logger = logging.getLogger('pymc3')
    logger.setLevel(log_level)

    # add empty columns to df
    df.reset_index(inplace=True, drop=True)
    presented_cols = [col for col in df.columns if 'presented_rad' in col]
    set_size = len(presented_cols)
    obs_cols =  [f'obs_{i}' for i in (np.arange(set_size) + 1)]
    df.loc[:, obs_cols] = np.nan
    df.loc[:, ['mu_hat', 'kappa_hat']] = np.nan
    theta_cols = [f'theta_hat_{i}' for i in (np.arange(set_size) + 1)]
    df.loc[:, theta_cols] = np.nan
    mu_cols = [f'mu_hat_{i}' for i in (np.arange(set_size) + 1)]
    df.loc[:, mu_cols] = np.nan
    kappa_cols = [f'kappa_hat_{i}' for i in (np.arange(set_size) + 1)]
    df.loc[:, kappa_cols] = np.nan
    
    pbar = tqdm(range(len(df)))
    for row in pbar:
        pbar.set_description(f'fitting trial {row + 1}...')

        presented = np.array(df.loc[row, presented_cols], dtype='float64')
        observed = get_noisy_observations(presented, kappa_obs)
        df.loc[row, obs_cols] = observed

        trace = sample_model(model, observed, progressbar=False)
        results = get_bfmm_estimates(trace)

        # match component params to reports using cluster assignment

        df.loc[row, mu_cols] = results['mu'][results['cluster']]
        df.loc[row, kappa_cols] = results['kappa'][results['cluster']]
        df.loc[row, theta_cols] = results['theta']

    logger.setLevel(logging.INFO)
    
    return df