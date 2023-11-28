#!/usr/bin/env python

"""Contains broadly useful functions for analysis"""

import numpy as np
from astropy.stats import circcorrcoef, circmean
from pingouin import circ_r
import scipy.stats as stats
from scipy.special import i0
import pandas as pd
from sklearn.utils import resample
from sklearn.neighbors import KernelDensity
import statsmodels.api as sm
from patsy import dmatrices
import warnings

def min_angle(presented, reported, radians=False):
    """Computes and returns the minimum signed angle between `presented` and `reported` 
    angles. Defaults to degrees, but returned values are radians if `radians`=True. 
    Assumes inputs are in [-180, 180] or [-pi, pi]. If radians, range is checked and
     corrected. All output angles equal to -180 (-pi) are cast to 180 (pi).
    """
    if radians:
        if np.max(presented) > np.pi and np.max(reported) > np.pi:
            presented = presented - np.pi
            reported = reported - np.pi
        elif bool(np.max(presented) > np.pi) ^ bool(np.max(reported) > np.pi):
            raise Exception('Range mismatch. Check inputs.')

        presented = np.degrees(presented)
        reported = np.degrees(reported)

    angles = np.round(((reported - presented + 180) % 360) - 180)
    angles[angles == -180] = 180
    angles_rad = np.radians(angles)

    return angles_rad if radians else angles

def round_to_angle(x, possible_angles, radians=True):
    """Rounds all angles in `x` to the nearest angle in `possible_angles`.

    Args:
        x (array-like): Array of angles to be rounded.
        possible_angles (array-like): Array of angles to round `x` to. 
        radians (bool, optional): Whether angles are radians (otherwise degrees). 
                                  Defaults to True.

    Returns:
        (np.ndarray): Array of rounded angles.
    """    
    rounded = []
    for val in x:
        residue = min_angle(possible_angles, val, radians=radians)
        rounded.append(possible_angles[np.argmin(np.abs(residue))])
    
    return np.array(rounded)

def round_df_angles(df, var, n_angles, radians=True):
    """Rounds all angles in `df[var]` to a discrete set of `n_angles` angles. Used to simulate
    reports by rounding continuous posterior estimates to allowable task values.

    Args:
        df (pd.DataFrame): Dataframe containing a column `var` with angles (floats or ints) 
                           to be rounded. 
        var (string): Name of column to be rounded. 
        n_angles (int): Number of angles to round to. Use 8 for discrete sim; 360 for continuous. 
        radians (bool, optional): Whether to round in rad; uses degress if false. Defaults to True.

    Returns:
        (pd.DataFrame): Input DataFrame with `var` column rounded. 
    """    
    if radians:
        angle_min = -np.pi
        angle_max = np.pi
    else:
        angle_min = -180
        angle_max = 180

    diff = np.round((angle_max - angle_min) / n_angles, decimals=15)
    possible_angles = np.round(np.linspace(angle_min + diff, angle_max, n_angles), decimals=15)

    df[var] = round_to_angle(df[var], possible_angles, radians=radians)

    return df

def vonmises_pdf(x, mu, kappa):
    """Custom von Mises probability density function. Wraps around [-pi, pi], which the scipy
    version doesn't for some reason.

    Args:
        x (array-like): points to evaluate pdf at
        mu (float): mean of the von mises distribution 
        kappa (float): concentration of the von mises distribution 

    Returns:
        (np.ndarray): vector of evaluated pdf values 
    """    
    return np.exp(kappa * np.cos(x - mu)) / (2. * np.pi * i0(kappa))

def vonmises_kde(x, kappa=5, n_bins=360):
    """Kernel density estimation for circular data using a von mises kernel and circular
    convolution (using the product of ffts). Adapted from https://stackoverflow.com/a/46896009.

    Args:
        x (array-like): Vector of circular data in the range [-pi, pi]. 
        kappa (float, optional): Concentration of vm kernel (aka bandwidth). Defaults to 5.
        n_bins (float, optional): Number of bins to use for estimate. Defaults to 360.

    Returns:
        bin_centers (np.ndarray): Vector of n_bins points used for kde. Useful for plotting.
        kde (np.ndarray): Vector of density estimates. 
    """    
    bins = np.linspace(-np.pi, np.pi, n_bins + 1, endpoint=True)
    hist_n, bin_edges = np.histogram(x, bins=bins)
    bin_centers = np.mean([bin_edges[1:], bin_edges[:-1]], axis=0)
    kernel = vonmises_pdf(x=bin_centers, mu=0, kappa=kappa)
    kde = np.fft.fftshift(np.fft.irfft(np.fft.rfft(kernel) * np.fft.rfft(hist_n)))
    kde /= np.trapz(kde, x=bin_centers)

    return bin_centers, kde

def circ_corr(x, y):
    """Computes the circular correlation of x and y (arrays of
    angles in radians) and performs a significance test. 
    Returns r (sample correlation coefficient) and a p (p-value).
    For more detail see Jammalamadaka and SenGupta (2003) section 8.2.
    """
    r = circcorrcoef(x, y)
    n = len(x)

    lam_20 = np.mean(np.sin(x - circmean(x)) ** 2)
    lam_02 = np.mean(np.sin(y - circmean(y)) ** 2)
    lam_22 = np.mean((np.sin(x - circmean(x)) ** 2) * (np.sin(y - circmean(y)) ** 2))
    test_stat = np.sqrt((n * lam_20 * lam_02) / lam_22) * r
    p = 2 * (1 - stats.norm.cdf(np.abs(test_stat)))

    return r, p

def max_kde(x, y):
    kde = KernelDensity()
    kde.fit(y[:, None])
    density = np.exp(kde.score_samples(x[:, np.newaxis]))
    
    return x[np.argmax(density)]

def get_relative_dist_df(data, var='presented', ref_response=1, radians=True):
    """**kind of deprecated since `get_trial_df()` is better, still used by compile_data for now**
    Get dataframe with relative distances suitable for plotting with pandas.FacetGrid().
    Inputs:
        - data (pd.DataFrame): data from only one condition, modality, and set_size
        - var (str): gets dists between either 'presented' or 'reported' angles. Defaults to 'presented'
        - ref_response (int): response to compute distance relative to
        - radians (bool): convert to radians or leave in degrees. Defaults to True.
    
    Returns a dataframe containing original ['subject', 'block', 'trial', 'set_size', 'response'] columns with two additional columns:
    - 'relative_dist': the distance between `ref_response` and current response in specified units
    - 'ref_response': ref_response 
    """

    if data.set_size.nunique() > 1:
        raise Exception('Must pass df with only one set size')
    
    dists = np.stack(np.array(data[f'{var}_dist']))
    relative_dist = dists[:, ref_response - 1]

    if radians:
        col = f'{var}_rad'
        relative_dist = np.radians(relative_dist)
    else:
        col = f'{var}_deg'

    relative_dist_df = data[['subject', 'block', 'trial', 'set_size', 'response', col]].copy()
    relative_dist_df.insert(len(relative_dist_df.columns), 'relative_to', ref_response)
    relative_dist_df.insert(len(relative_dist_df.columns), 'relative_dist', relative_dist)

    return relative_dist_df

def get_extract_cols(var_list):
    def extract_cols(group):
        """To be applied as a pandas groupby operation. 
        Input `group` is a dataframe corresponding to one specific trial 
        (ie data grouped by ['subject','block','trial'])
        """
 
        set_size = len(group)
        values = np.zeros((len(var_list), set_size))
        
        for ind, var in enumerate(var_list):
            values[ind] = group[var].values

        return np.concatenate(values)
    
    return extract_cols

def get_trial_df(data, var_list):
    """Returns a new dataframe where each row is one trial. Input `var_list` specifies which
    variables from the original dataframe to include in the new dataframe.
    Column names are variable names with response number appended.
    """
    if data.set_size.nunique() > 1:
        raise Exception('Must pass df with only one set size')
    
    set_size = data.reset_index().set_size[0]
    col_names = []
    for var in var_list:
        for response in np.arange(set_size) + 1:
            col_names.append(f'{var}_{response}')

    extract_cols = get_extract_cols(var_list)
    series = data.groupby(['subject', 'block', 'trial']).apply(extract_cols)
    values = np.vstack(series.values)
    df = series.index.to_frame(index=False)

    trial_df = pd.concat([df, pd.DataFrame(values)], axis=1)
    trial_df.columns = list(df.columns) + col_names
    trial_df.insert(3, 'set_size', set_size)
    
    return trial_df.dropna()

def get_extract_responses(set_size, var_list, trial_var_list=None):
    def extract_responses(group):
        """Meant to be applied as a pandas groupby operation... but I couldn't get it to work.
        Unpacks one row of a trial_df and returns a response_df of length = set_size.
        """
        response_df = pd.DataFrame()

        for var in ['subject', 'block', 'trial', 'set_size']:
            response_df[var] = set_size * [int(group[var])]

        response_df['response'] = np.arange(set_size) + 1

        for var in var_list:
            response_df[var] = [group[f'{var}_{response}'] for response in np.arange(set_size) + 1]

        if trial_var_list is not None:
            for trial_var in trial_var_list:
                response_df[trial_var] = set_size * [group[trial_var]]

        return response_df
    return extract_responses

def get_response_df(data, var_list, trial_var_list=None):
    """Returns a new dataframe where each row is one response (opposite of get_trial_df()).
    Input `var_list` specifies which variables from the original dataframe to include in the 
    new dataframe. **probably a way to do this with groupby, but I surrender**
    """
    col_names = list(data.columns)
    set_size = data.reset_index().set_size[0]

    extract_responses = get_extract_responses(set_size, var_list, trial_var_list)
    response_df = pd.DataFrame()

    with warnings.catch_warnings(): # suppress futurewarning
        warnings.simplefilter(action='ignore', category=FutureWarning)
        response_df = response_df.append(extract_responses(data.iloc[0]), ignore_index=True)

        for row in range(1, len(data)):
            response_df = response_df.append(extract_responses(data.iloc[row]), ignore_index=True)

    return response_df

def get_kl_df(data, ref_response=1, var='reported', n_iters=1000, kl_bins=18, sample_bins=8,
              uniform=False, empirical=False, condition=None):
    """ Performs bootstrapped KL divergence estimation between relative distance distributions
        and circular uniform samples. Returns all bootstrap samples as one Dataframe.

    Args:
        data (pandas.DataFrame): Dataframe containing data from only one modality and
                                 set size.
        var (str, optional): Variable to compute relative distance for ('reported' or 'presented').
                             Defaults to 'reported'. See utils.get_relative_dist() for info.
        n_iters (int, optional): Number of bootstrap iterations to perform. Defaults to 1000.
        kl_bins (int, optional): Number of bins to use when estimating KL div. Ideally, use 360
                                 bins for continuous dataset and 8 for discrete. Defaults to 8.
        sample_bins (int, optional): Number of bins to sample from when generating uniform samples.
                                     Use 360 for continuous, 8 for discrete. Defaults to 8.
        uniform (bool, optional): If true, uniform samples the same size as the data are used 
                                  instead of the data (baseline). Defaults to False.
        condition (str, optional): Condition label to be inserted into kl dataframe (used for
                                   plotting with seaborn). Defaults to None.

    Returns:
        kl_df: Dataframe containing n_iters rows with columns ['condition', 'samples', 'response']. 
    """    
    relative_dist_df = get_relative_dist_df(data, var=var, ref_response=ref_response).dropna()
    set_size = relative_dist_df.reset_index().set_size[0]
    responses = np.delete(np.arange(set_size) + 1, ref_response - 1)
    samples = []

    for response in responses:
        x = relative_dist_df[relative_dist_df.response == response]['relative_dist']
        results = bootstrap_kl_vs_uniform(x, n_iters=n_iters, kl_bins=kl_bins, 
                                          sample_bins=sample_bins, uniform=uniform, 
                                          empirical=empirical, confidence=0.95)
        samples.append(results['samples'])

    response = [[resp] * n_iters for resp in responses]
    response = list(np.array(response).flat)
    kl_df = pd.DataFrame({'response': response, 'samples': np.hstack(samples)})

    if condition is not None:
        kl_df['condition'] = condition

    return kl_df

def compute_trial_mean(data, mean_var='presented_rad', var_list=['presented_rad', 'reported_rad'], 
                       trial_var_list=None, is_response_df=True):
    """Accepts a tidy dataframe of data or model fits and returns a tidy dataframe with an 
    additional column for the trial mean of `mean_var`. 
    The trial mean column is labeled \'{mean_var}_mean\'. Input can be either a response_df 
    (one response/row) or a trial_df (one trial/row), output is always a response_df.
    For more on trial_df <-> response_df conversion, see `get_trial_df()` and `get_response_df()`.

    Args:
        data (pd.DataFrame): Tidy dataframe containing only one set_size.
        mean_var (str, optional): Column to take the trial mean of. Defaults to 'presented_rad'.
        var_list (list, optional): List of column names to be unpacked and repacked (ie variables 
                                   with more than one entry per trial). 
                                   Defaults to ['presented_rad', 'reported_rad'].
        trial_var_list (list, optional): List of column names to be repacked from trial_df -> response_df.
                                         Defaults to None. ['subject', 'block', 'trial', 'set_size'] are
                                         automatically repacked and do not need to be included.
        is_response_df (bool, optional): If True, `data` is converted from response_df -> trial_df 
                                         before taking mean. Defaults to True.

    Raises:
        Exception: Raised if `data` has more than one set_size.

    Returns:
        (pd.DataFrame): response_df with a column containing the trial mean of `mean_var`.
    """

    if data.set_size.nunique() > 1:
        raise Exception('Must pass df with only one set size')
    
    set_size = data.reset_index().set_size[0]

    if is_response_df: # convert to trial_df
        trial_df = get_trial_df(data, var_list=var_list)
    else:
        trial_df = data

    # get circular mean for each trial
    mean_name = mean_var + '_mean'
    columns = [f'{mean_var}_{response}' for response in range(1, set_size + 1)]
    presented_list = [trial_df[col].to_numpy() for col in columns]
    presented_array = np.stack(presented_list).T
    trial_df[mean_name] = circmean(presented_array, axis=1)

    if trial_var_list is None:
        trial_var_list = [mean_name]
    else:
        trial_var_list.insert(0, mean_name)

    mean_df = get_response_df(trial_df, var_list=var_list, trial_var_list=trial_var_list)

    return mean_df

def compute_ensemble_bias(data):
    """Returns a dataframe with additional columns for presented mean (centered on report_rad) and bias (toward presented mean).
    """
    if data.set_size.nunique() > 1:
        raise Exception('Must pass df with only one set size')
    
    set_size = data.reset_index().set_size[0]

    trial_df = get_trial_df(data, var_list=['presented_rad', 'reported_rad', 'error_rad'])
    columns = [f'presented_rad_{response}' for response in range(1, set_size + 1)]
    presented_list = [trial_df[col].to_numpy() for col in columns]
    presented_array = np.stack(presented_list).T

    for response in np.arange(set_size) + 1:
        # subtract ref_response val from all presented
        presented_corrected = min_angle(np.tile(presented_array[:, response - 1],(set_size, 1)).T,
                                        presented_array, radians=True)
        # compute mean and bias for every response
        trial_df[f'presented_mean_{response}'] = circmean(presented_corrected, axis=1)
        trial_df[f'bias_{response}'] = trial_df[f'error_rad_{response}'] / trial_df[f'presented_mean_{response}'] * 100

    # remove infs
    trial_df.drop(trial_df.index[np.isinf(trial_df).any(1)], inplace=True)

    # convert back to response_df
    var_list = ['presented_rad', 'reported_rad', 'error_rad', 'presented_mean', 'bias']
    response_df = get_response_df(trial_df, var_list)

    return response_df

def compute_bias_ols(data, x='relative_mean', y='relative_error'):
    """OLS regression of y on x + intercept

    Args:
        data ([type]): [description]
        x (str, optional): [description]. Defaults to 'relative_mean'.
        y (str, optional): [description]. Defaults to 'relative_error'.

    Returns:
        [type]: [description]
    """
    y, X = dmatrices(f'{y} ~ {x}', data=data, return_type='dataframe')
    model = sm.OLS(y, X)
    
    return model.fit()

def compute_pairwise_corr(data, var='reported_rad'):
    """Returns dict of two [set_size x set_size] arrays. 'corr' contains pairwise circular correlations 
    between specified `var` (within-trial) and 'p_val' contains the result of significance tests.
    Input data (pd.DataFrame): data from only one condition, modality, and set_size
    """
    if data.set_size.nunique() > 1:
        raise Exception('Must pass df with only one set size')

    trial_df = get_trial_df(data, [var])    
    set_size = data.reset_index().set_size[0]
    corr = np.empty((set_size, set_size))
    p_val = np.empty((set_size, set_size))

    for row in range(set_size):
        for col in range(set_size):
            r, p = circ_corr(trial_df[f'{var}_{row + 1}'], 
                            trial_df[f'{var}_{col + 1}'])
            corr[row, col] = r 
            p_val[row, col] = p

    return {'corr': corr, 'p_val': p_val}

def calc_kl_div(x, y, kl_bins=18, base=2):
    """Computes KL divergence between empirical distributions of `x` and `y`.
    Assumes both `x` and `y` are in [-pi, pi]. `bins` determines # evenly-spaced bins to use.
    Logarithm base set by `base` (if 2, KL is in bits; if e, KL is in nats)
    """
    counts, _ = np.histogram(x, bins=kl_bins, range=[-np.pi, np.pi])
    P_x = counts / counts.sum()

    counts, _ = np.histogram(y, bins=kl_bins, range=[-np.pi, np.pi])
    P_y = counts / counts.sum()

    return stats.entropy(P_x, P_y, base=base)

def sample_uniform_circ(n, bins=None):
    """Returns `n` samples from the uniform on [-pi, pi]
    """
    if bins is None:
        samples = np.random.rand(n) * 2 * np.pi - np.pi
    else:
        samples = np.random.randint(bins, size=n) + 1 
        samples = samples / bins * 2 * np.pi - np.pi

    return samples

def calc_kl_vs_uniform(x, kl_bins=18, empirical=False, **kwargs):
    """Performs one resampling of input `x` (assumed to be in [-pi, pi]) and returns KL divergence
       from an empirical uniform. `kl_bins` determine bin size for empirical dists.
    """
    
    x_bs = resample(x, replace=True)

    if empirical:
        uni_edges = np.linspace(-np.pi, np.pi, kl_bins + 1)
        y = uni_edges[:-1] + (np.diff(uni_edges) /2)
    else:
        n = len(x)
        y = sample_uniform_circ(n)

    return calc_kl_div(x, y, kl_bins=kl_bins, **kwargs)

def bootstrap_kl_vs_uniform(x, n_iters=1000, kl_bins=18, sample_bins=8, 
                            uniform=False, confidence=0.95, **kwargs):
    """Gets bootstrapped estimate of KL divergence between `x` (assumed to be in [-pi, pi]) and
    a uniform on [-pi, pi]. `n_iters` determines number of resamplings.`kl_bins` determine bin size 
    for empirical dists. `confidence` specifies upper/lower confidence bounds to return.
    Returns a dict with 'samples' (sample KL vals), 'mean' (sample mean) 'lower' (lower CI), 
    'upper'(upper CI).
    """
    
    kl = []

    while len(kl) < n_iters:
        if uniform:
            x = sample_uniform_circ(len(x), bins=sample_bins)

        temp_kl = calc_kl_vs_uniform(x, kl_bins=kl_bins, **kwargs)

        if temp_kl < 1e308:
            kl.append(temp_kl)
    
    samples = np.array(kl)
    mean = samples.mean()
    lower = np.percentile(samples, (1 - confidence) / 2 * 100)
    upper = np.percentile(samples, (1 - confidence) / 2 * 100 + confidence * 100)

    return {'samples': samples, 'mean': mean, 'lower': lower, 'upper': upper}