#!/usr/bin/env python

"""
Contains commonly-used plotting functions.
"""

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
import numpy as np
import pandas as pd
import pingouin as pg
import scipy.stats as stats
from scipy.io import loadmat

from utils import get_relative_dist_df, get_trial_df, circ_corr, compute_pairwise_corr
from utils import bootstrap_kl_vs_uniform, compute_ensemble_bias, get_kl_df, vonmises_pdf
from utils import compute_bias_ols
from models import phase_locked_sine, fit_phase_locked_sine, compute_relative_error, compute_bias_stats

from data import PROCESSED_DIR

sns.set()

# plot colors
FREE_COLOR = sns.color_palette()[1]
RAND_COLOR = sns.color_palette()[0]
FREE_ORIENT = sns.color_palette()[2]
RAND_ORIENT = sns.color_palette()[4]
KAPPA_OBS_PALETTE = sns.color_palette("rocket")[:3]
RESP_PALETTE = 'crest' # not used
SET_SIZE_PALETTE = 'crest_r'
DISC_EDGES = np.array([-np.pi, -2.74889357, -1.96349541, -1.17809725, -0.39269908,
                       0.39269908,  1.17809725,  1.96349541,  2.74889357,  np.pi])

# general-purpose functions

def set_axes_rad(fig, axes):
    """sets xticks/xticklabels/xlim of all axes to +/- pi

    Args:
        fig (matplotlib.pyplot.figure):  
        axes (array-like): flat array of matplotlib.pyplot.axis objects 
    """
    for ax in axes:
        ax.set_xlim([-np.pi, np.pi])
        ax.set_xticks([-np.pi, 0, np.pi])
        ax.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])

def set_axes_aspect(fig, axes, ratio=1):
    """Sets aspect ratio (in display units) of all axes.
    Expects axes input to be a flat array of axes.
    """
    for ax in axes:
        ax.set_aspect(1.0 / ax.get_data_ratio() * ratio)

def set_axes_invisible(fig, axes):
    """Sets visibility of provided axes to False (invisible).
    Expects axes input to be a flat array of axes.
    """
    for ax in axes: 
        ax.set_visible(False)

def set_axes_labels(fig, axes, remove_ticklabels=False):
    """removes all text from a figure

    Args:
        fig (matplotlib.pyplot.figure):  
        axes (array-like): flat array of matplotlib.pyplot.axis objects
        remove_ticklabels (bool): whether to remove ticklabels 
    """
    
    for ax in axes:
        ax.set_title('')
        ax.set_xlabel('')
        ax.set_ylabel('')

        if remove_ticklabels:
            ax.set_xticklabels('')
            ax.set_yticklabels('')

def set_box_widths(ax, fac):
    """
    Adjust the widths of a seaborn-generated boxplot. 
    Adapted from https://stackoverflow.com/a/56955897.

    Args:
        ax (matplotlib.pyplot.axis): axis containing a seaborne boxplot
        fac (float): factor to adjust width by.
    """
    # iterating through axes artists:
    for c in ax.get_children():

        # searching for PathPatches
        if isinstance(c, PathPatch):
            # getting current width of box:
            p = c.get_path()
            verts = p.vertices
            verts_sub = verts[:-1]
            xmin = np.min(verts_sub[:, 0])
            xmax = np.max(verts_sub[:, 0])
            xmid = 0.5*(xmin+xmax)
            xhalf = 0.5*(xmax - xmin)

            # setting new width of box
            xmin_new = xmid-fac*xhalf
            xmax_new = xmid+fac*xhalf
            verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
            verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

            # setting new width of median line
            for l in ax.lines:
                if np.all(l.get_xdata() == [xmin, xmax]):
                    l.set_xdata([xmin_new, xmax_new])

def set_export_style(fig, axes, remove_ticklabels=False, figsize=(7.2, 1.8)):
    """applies some standard fig formatting for export to png

    Args:
        fig (matplotlib.pyplot.figure): 
        axes (array-like): flat array of matplotlib.pyplot.axis objects 
        remove_ticklabels (bool): whether to remove ticklabels
        figsize (tuple of floats): figure size in inches (width, height). Defaults to (7.2, 1.8).
    """
    sns.set_style('ticks', {'axes.linewidth': 2, 'axes.edgecolor':'black'})
    set_axes_labels(fig, axes, remove_ticklabels=remove_ticklabels)
    fig.set_size_inches(figsize)

def get_axes_section(fig, axes, dim, section):
    """Reshapes a flat array of axes to the shape provided by `dim`, then
    returns only the axes specified by `section`. `section` refers to part of a 2d matrix, and 
    can be `upper`, `lower`, or `diag`. Output is a flat array of axes.
    """
    if section == 'upper':
        ind = np.triu_indices(n=dim[0], m=dim[1], k=1)
    elif section == 'lower':
        ind = np.tril_indices(n=dim[0], m=dim[1], k=-1)
    elif section == 'diag':
        ind = np.diag_indices(n=dim[0])
    else:
        raise Exception('possible section args: \'upper\', \'lower\', \'diag\'')

    return axes.reshape(dim)[ind]

def get_colorwheel(prefix='../'):
    """Fetches RGB values for Adam et al's color wheel.

    Args:
        prefix (str, optional): Appended to beginning of PROCESSED_DIR to allow use
                                from different directories. Default is '../'
    Returns:
        (np.ndarray; (360,3)): Array of RGB values between 0 and 1. 
    """    
    mat = loadmat(f'{prefix}{PROCESSED_DIR}/colorwheel360.mat')
    return mat['fullcolormatrix']/255

def plot_shaded_error(x, value, error, color, alpha=0.2, **kwargs):
    """Plots value and shaded error (+/- 1 error). 

    Args:
        x (array-like): vector of x-axis values 
        value (array-like): vector of values to (line) plot. Must have same shape as x. 
        error (array-like): vector of errors to shade between. Must have same shape as x. 
        color ([1x3] array-like): RGB color value to plot. 
        alpha (float): Opacity of error shading. Defaults to 0.2.
    
    Returns:
        fig (matplotlib.pyplot.figure): Figure handle.
        axes (array of matplotlib.pyplot.axis): Figure axes.]
    """    
    plt.plot(x, value, color=color, **kwargs)

    upper = value + (error)
    lower = value - (error)
    plt.fill_between(x, upper, lower, color=color, alpha=alpha)
    
    return plt.gcf(), plt.gca()

def mirror_pi_errors(data):
    """Plotting utility function that appends a vector of -pi to data. The appended vector has
    length len(data[data == np.pi]).
    This is for visualization purposes only; it is needed because discrete datasets contain 
    only positive pi errors.
    This is *only* used when plotting error histograms of discrete data, and in those cases
    histogram bar widths are halved to demonstrate the difference visually.

    Args:
        data (array-like): A pandas.Series or numpy.ndarray of error values in radians.

    Returns:
        data_mirrored (numpy.ndarray): An array containing data + additional -np.pi values. 

    """ 
    data_mirrored = np.array(data, copy=True)
    n = len(data_mirrored[data_mirrored == np.pi])
    data_mirrored = np.append(data_mirrored, np.ones(n) * - np.pi)

    return data_mirrored

def weighted_hist(data, split_pi=False, **kwargs):
    """Wrapper for plt.hist() that computes a normalizing weight vector for the inputs.
    For use with sns.FacetGrid().
    If split_pi is True, all elements of data equal to np.pi are "mirrored"; see `mirror_pi_errors()`.
    """
    if split_pi:
        data = mirror_pi_errors(data)

    n, edges, patches = plt.hist(data, weights=np.ones(len(data)) / len(data), **kwargs)

    return n, edges, patches

def paired_hist(x_upper, x_lower, histtype='stepfilled', paired_colors=None, **kwargs):
    """Wrapper for weighted_hist() that takes two arrays of data and plots paired histograms 
    on a mirrored y-axis. **kwargs passed to weighted_hist().

    Args:
        x_upper (array-like): data to be plotted on upper half of axis
        x_lower (array-like): data to be plotted on lower half of axis 
        histtype (str, optional): Defaults to 'stepfilled'.
        paired_colors (array-like): set of colors for plotting [upper, lower]
    """

    if paired_colors is not None:
        _, edges, patches = weighted_hist(x_lower, histtype=histtype, 
                                          color=paired_colors[1], **kwargs)
    else:
        _, edges, patches = weighted_hist(x_lower, histtype=histtype, **kwargs)
    ax = plt.gca()

    # invert bars
    for p in patches:
        if histtype == 'stepfilled' or histtype == 'step':
            xy = p.get_xy()
            xy[:,1] = -xy[:,1]
            p.set_xy(xy)
        else:
            p.set_height(-p.get_height())

    try:
        kwargs.pop('bins')
    except KeyError:
        pass
    
    if paired_colors is not None:
        weighted_hist(x_upper, histtype=histtype, bins=edges, color=paired_colors[0], **kwargs)
    else:
        weighted_hist(x_upper, histtype=histtype, bins=edges, **kwargs)

    ylim = (-ax.get_ylim()[1], ax.get_ylim()[1])
    ax.set_ylim(ylim)

    pos_ticks = np.array([t for t in ax.get_yticks() if t > 0])
    ticks = np.concatenate([-pos_ticks[::-1], [0], pos_ticks])
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{abs(t):.2f}' for t in ticks])

    return plt.gcf(), ax

# plot-specific functions

def plot_error_by_set_size(data, bins=37, density=False, histtype='stepfilled', **kwargs):
    """Accepts a tidy dataframe and uses seaborn.FacetGrid to plot histogram of cumulative
    errors (in rad) for every set size, collapsed across responses. 
    **kwargs are arguments for plt.hist
    `axes` output is a flat array of all axes.
    """
    g = sns.FacetGrid(data.dropna(), col='set_size', sharex=False, sharey=False)
    g.map(weighted_hist, 'error_rad', bins=bins, density=density, histtype=histtype,
          edgecolor='none', **kwargs)

    fig = plt.gcf() 
    axes = np.array(g.axes.flat)
    set_axes_rad(fig, axes)
    ylabel = 'probability density' if density else 'proportion'
    g.set_axis_labels('report error (rad)', ylabel)

    return fig, axes, g

def plot_paired_error_by_set_size(data_upper, data_lower, bins=90, histtype='stepfilled', 
                                  figsize=(10, 3), linewidth=0, **kwargs):
    """Plots paired histograms of report error for each set size in provided dataframes. 

    Args:
        data_upper (pandas.DataFrame): Dataframe containing entries for `set_size` and `error_rad`. 
                                       Plotted on upper axis.
        data_lower (pandas.DataFrame): Dataframe containing entries for `set_size` and `error_rad`. 
                                       Plotted on lower axis.
        
        bins (int, optional): Number of histogram bins. Defaults to 90.
        histtype (str, optional): Defaults to 'stepfilled'.
        figsize (tuple, optional): Inches. Defaults to (10, 3).
        linewidth (int, optional): Histogram outline width. Defaults to 0.

    Raises:
        Exception: Both dataframes must have the same set sizes.

    Returns:
        fig (matplotlib.pyplot.figure): Figure handle.
        axes (array of matplotlib.pyplot.axis): Figure axes.
    """
    set_sizes = np.sort(data_upper.set_size.unique())

    if not np.array_equal(set_sizes, np.sort(data_lower.set_size.unique())):
        raise Exception('Both dfs must have the same set sizes')
    
    fig, axes = plt.subplots(1, len(set_sizes), sharey=False, figsize=figsize)

    for ax, ss in zip(axes, set_sizes):
        plt.sca(ax)
        ax.set_title(f'set size {ss}')
        x_upper = data_upper[data_upper.set_size == ss].error_rad
        x_lower = data_lower[data_lower.set_size == ss].error_rad
        paired_hist(x_upper, x_lower, bins=bins, histtype=histtype,   
                    linewidth=linewidth, **kwargs)

    set_axes_aspect(fig, axes)
    set_axes_rad(fig, axes)

    axes[0].set_ylabel('proportion')
    axes[0].set_xlabel('error (rad)')

    return fig, axes

def plot_error_by_response(data, bins=37, density=False, histtype='stepfilled', legend=True,
                           palette=RESP_PALETTE, **kwargs):
    """Accepts a tidy dataframe and uses seaborn.FacetGrid to plot histogram of cumulative
    errors (in rad) for every set size and response number. Removes empty axes if all set sizes
    and responses are provided. 
    **kwargs are arguments for plt.hist
    `axes` output is a flat array of all axes.
    """
    g = sns.FacetGrid(data.dropna(), col='response', row='set_size', hue='response', 
                      palette=palette, sharex=False, sharey=False)
    g.map(weighted_hist, 'error_rad', bins=bins, density=density, histtype=histtype,
          edgecolor='none', **kwargs)
    if legend:
        g.add_legend()
    fig = plt.gcf()
    axes = np.array(g.axes.flat)
    axes_dim = [len(g.row_names), len(g.col_names)]

    set_axes_rad(fig, axes)
    ylabel = 'probability density' if density else 'proportion'
    g.set_axis_labels('report error (rad)', ylabel)

    if axes_dim == [5, 6]: # all set sizes/responses; remove empty axes
        axes_upper = get_axes_section(fig, axes, axes_dim, 'upper')
        set_axes_invisible(fig, axes_upper)
        axes[-1].set_visible(True)

    return fig, axes, g

def plot_paired_error_by_response(data_upper, data_lower, bins=90, histtype='stepfilled', 
                                  figsize=(10, 3), linewidth=0, subplot_fill=None, **kwargs):
    """Plots paired histograms of report error for each response in provided dataframes. 

    Args:
        data_upper (pandas.DataFrame): Dataframe containing entries for `set_size`, `response` 
                                       and `error_rad`. Plotted on upper axis.
        data_lower (pandas.DataFrame): Dataframe containing entries for `set_size`, `response` 
                                       and `error_rad`. Plotted on lower axis.
        
        bins (int, optional): Number of histogram bins. Defaults to 90.
        histtype (str, optional): Defaults to 'stepfilled'.
        figsize (tuple, optional): Inches. Defaults to (10, 3).
        linewidth (int, optional): Histogram outline width. Defaults to 0.
        subplot_fill (int, optional): If provided, fills subplot with empty panes (for sizing).
        **kwargs: Passed to paired_hist().

    Raises:
        Exception: Both dataframes must have a single set size (and both must be the same).

    Returns:
        fig (matplotlib.pyplot.figure): Figure handle.
        axes (array of matplotlib.pyplot.axis): Figure axes.
    """
    set_size = data_upper.reset_index()['set_size'][0]
    if data_upper.set_size.nunique() > 1 or data_lower.set_size.nunique() > 1:
        raise Exception('Must pass df with only one set size')
    elif set_size != data_lower.reset_index()['set_size'][0]:
        raise Exception('Both dfs must have the same set size')
    
    n_subplots = set_size if subplot_fill is None else subplot_fill
    
    fig, axes = plt.subplots(1, n_subplots, sharey=False, figsize=figsize)

    for ax, response in zip(axes[:set_size], np.arange(set_size) + 1):
        plt.sca(ax)
        ax.set_title(f'response {response}')
        x_upper = data_upper[data_upper.response == response].error_rad
        x_lower = data_lower[data_lower.response == response].error_rad
        paired_hist(x_upper, x_lower, bins=bins, histtype=histtype,   
                    linewidth=linewidth, **kwargs)

    set_axes_aspect(fig, axes)
    set_axes_rad(fig, axes)

    axes[0].set_ylabel('proportion')
    axes[0].set_xlabel('error (rad)')
    
    return fig, axes

def plot_error_by_response_stacked(data, bins=37, density=True, histtype='step', legend=True, 
                                   palette=RESP_PALETTE, **kwargs):
    """Accepts a tidy dataframe and uses seaborn.FacetGrid to plot histogram of cumulative
    errors (in rad) for every set size and response number. Responses for each set size are stacked
    onto one axis.
    **kwargs are arguments for plt.hist
    `axes` output is a flat array of all axes.
    """
    g = sns.FacetGrid(data.dropna(), col='set_size', hue='response', palette=palette, sharex=False)
    g.map(plt.hist, 'error_rad', bins=bins, density=density, histtype=histtype, 
          **kwargs)
    if legend:
        g.add_legend()

    fig = plt.gcf()
    axes = np.array(g.axes.flat)
    set_axes_rad(fig, axes)
    g.set_axis_labels('report error (rad)', 'probability density')

    return fig, axes, g

def plot_error_by_response_ecdf(data, palette=RESP_PALETTE, legend=True, **kwargs):
    """Accepts a tidy dataframe and uses seaborn.displot to plot ecdfs of errors.
    """
    g = sns.FacetGrid(data.dropna(), col='set_size', hue='response', palette=palette, sharex=False,
                      sharey=False)
    g.map(sns.ecdfplot, 'error_rad', 
          **kwargs)
    if legend:      
        g.add_legend()

    fig = plt.gcf()
    axes = np.array(g.axes.flat)
    set_axes_rad(fig, axes)
    g.set_axis_labels('report error (rad)', 'cumulative prob.')

    return fig, axes, g

def plot_mrvl_by_response(data, palette=SET_SIZE_PALETTE):
    mrvl_df = data.groupby(['subject','set_size', 'response'])['error_rad'].aggregate(pg.circ_r).reset_index()
    mrvl_df.rename(columns={'error_rad': 'mrvl'}, inplace=True)

    # mean points
    mrvl_df_mean = mrvl_df.groupby(['set_size', 'response'])['mrvl'].aggregate(np.mean).reset_index()
    ax = sns.scatterplot(data=mrvl_df_mean, x='response', y='mrvl', hue='set_size', palette=palette, legend=False)

    # mean line and error bars
    ax = sns.lineplot(data=mrvl_df, x='response', y='mrvl', hue='set_size', palette=palette)

    if any(mrvl_df['set_size'] == 1):
        ss_one = mrvl_df[mrvl_df['set_size'] == 1]['mrvl'].to_numpy()
        ss_one_ci = stats.t.interval(0.95, len(ss_one) - 1, loc=np.mean(ss_one), scale=stats.sem(ss_one))
        ci_color = sns.color_palette(palette, mrvl_df['set_size'].nunique())[0]
        plt.errorbar(1, np.mean(ss_one), yerr=np.diff(ss_one_ci), ecolor=ci_color)

    fig = plt.gcf()
    ax.set_xlabel('report #')
    ax.set_ylabel('MRVL')

    return plt.gcf(), ax

def plot_relative_distance(data, ref_response=1, var='reported', bins=37, density=False, 
                           histtype='stepfilled', wspace=0.2, palette=RESP_PALETTE, sharey=True, **kwargs):
    """Accepts a tidy dataframe (with only one set size), and plots distributions of relative distances
    between the provided response (`ref_response`) and all other responses in the trial. `var` argument can 
    be 'presented' (distance between true values of items selected) or 'reported' (distance between reported values)
    """
    relative_dist_df = get_relative_dist_df(data, var=var, ref_response=ref_response)
    g = sns.FacetGrid(relative_dist_df.dropna(), col='response', hue='response', palette=palette, sharey=sharey, sharex=False)
    g.map(weighted_hist, 'relative_dist', bins=bins, density=density, histtype=histtype, edgecolor='none', **kwargs)

    fig = plt.gcf() 
    axes = np.array(g.axes.flat)

    if not sharey:
        ylim = axes[0].get_ylim()
        for ax in axes[1:]:
            ax.set_ylim(ylim)

    set_axes_rad(fig, axes)
    set_axes_aspect(fig, axes)
    plt.subplots_adjust(hspace=0.4, wspace=wspace)
    fig.set_size_inches((12, 6))
    axes[0].set_ylabel('proportion')

    return fig, axes

def plot_paired_relative_distance(data_upper, data_lower, ref_response=1, var='reported', bins=90, 
                                  histtype='stepfilled', figsize=(10, 3), linewidth=0, 
                                  truncate_comparisons=False, **kwargs):
    """Plots paired histograms of relative distance to ref_response for all other responses in  
    provided dataframes. 

    Args:
        data_upper (pandas.DataFrame): Dataframe containing entries for `set_size`, `response` 
                                       and `error_rad`. Plotted on upper axis.
        data_lower (pandas.DataFrame): Dataframe containing entries for `set_size`, `response` 
                                       and `error_rad`. Plotted on lower axis.
        ref_response (int, optional): Response number to compare all other responses to. Defaults
                                      to 1.
        var (str, optional): Variable to compute relative distance for ('reported' or 'presented').
                             Defaults to 'reported'. See utils.get_relative_dist() for info.
        bins (int, optional): Number of histogram bins. Defaults to 90.
        histtype (str, optional): Defaults to 'stepfilled'.
        figsize (tuple, optional): Inches. Defaults to (10, 3).
        linewidth (int, optional): Histogram outline width. Defaults to 0.
        truncate_comparisons (bool, optional): If True, only comparisons to *later* responses are 
                                               shown. If False all comparisons are shown. 
                                               Defaults to False.
        **kwargs: Passed to paired_hist() 

    Raises:
        Exception: Both dataframes must have a single set size (and both must be the same).

    Returns:
        fig (matplotlib.pyplot.figure): Figure handle.
        axes (array of matplotlib.pyplot.axis): Figure axes.
    """ 
    set_size = data_upper.reset_index().set_size[0]
    if data_upper.set_size.nunique() > 1 or data_upper.set_size.nunique() > 1:
        raise Exception('Must pass df with only one set size')
    elif set_size != data_lower.reset_index().set_size[0]:
        raise Exception('Both dfs must have the same set size')

    dist_df_upper = get_relative_dist_df(data_upper, 
                                         var=var, ref_response=ref_response).dropna()
    dist_df_lower = get_relative_dist_df(data_lower, 
                                         var=var, ref_response=ref_response).dropna()

    fig, axes = plt.subplots(1, set_size - 1, sharey=False, figsize=figsize)

    responses = np.delete(np.arange(set_size) + 1, ref_response - 1)
    
    for ax, response in zip(axes, responses):
        plt.sca(ax)
        ax.set_title(f'report {ref_response} v. {response}')
        x_upper = dist_df_upper[dist_df_upper.response == response].relative_dist
        x_lower = dist_df_lower[dist_df_lower.response == response].relative_dist
        paired_hist(x_upper, x_lower, bins=bins, histtype=histtype,   
                    linewidth=linewidth, **kwargs)

    ylim = axes[0].get_ylim()
    [ax.set_ylim(ylim) for ax in axes]
    [ax.set_yticks([]) for ax in axes[1:]]

    set_axes_aspect(fig, axes)
    set_axes_rad(fig, axes)

    axes[0].set_ylabel('proportion')
    axes[0].set_xlabel('relative dist. (rad)')

    if truncate_comparisons:
        for ax in axes[:ref_response - 1]:
            ax.set_visible(False)
    
    return fig, axes

def plot_relative_distance_scatter(data, ref_response=1, var='reported_rad', size=10,
                                   palette=None, color=None, corr=True):
    """Accepts a tidy dataframe (with only one set size), and plots within-trial pairwise
    relationships of `var`. Response specified by `ref_response` plotted on y-axis.
    """
    set_size = data.reset_index().set_size[0]
    trial_df = get_trial_df(data, [var])
    cols = [col for col in list(trial_df.columns) if var in col]
    cols.remove(f'{var}_{ref_response}')

    fig, axes = plt.subplots(1, set_size -1, figsize=(12, 6), sharey=False)
    
    for ind, ax in enumerate(axes):
        ax.scatter(trial_df[f'{var}_{ref_response}'], trial_df[cols[ind]], 
                   s=size, marker='.', color=color, alpha=0.75, edgecolor='none')

        if corr:
            r, p = circ_corr(trial_df[f'{var}_{ref_response}'], trial_df[cols[ind]])
            ax.set_title(f'r = {np.round(r, 2)}  p = {np.round(p,2)}')

        ax.set(adjustable='box', aspect='equal')
        ax.set_xlim([-np.pi - 0.2, np.pi + 0.2])
        ax.set_xticks([-np.pi, 0, np.pi])
        ax.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
        ax.set_ylim([-np.pi - 0.2, np.pi + 0.2])
        ax.set_yticks([-np.pi, 0, np.pi])
        ax.set_yticklabels([r'$-\pi$', '0', r'$\pi$'])
        ax.set_xlabel(cols[ind])
    axes[0].set_ylabel(f'response_{ref_response}')

    return fig, axes

def plot_proportion_sig_corr(data, ref_response=1, var='reported_rad',
                             palette=RESP_PALETTE):
    """Plots bar chart w/ proportion of subjects that have significant (p<0.05)
    circular correlations between `ref_response` variable and all other responses.
    """
    set_size = data.reset_index().set_size[0]
    results = data.groupby(['subject']).apply(compute_pairwise_corr).to_frame('temp_dict')
    n_subs = len(results)
    results = results.temp_dict.apply(pd.Series)
    all_p_vals = np.stack(results['p_val'].to_numpy())

    total_sig = (all_p_vals[:,ref_response - 1,:] < 0.05).sum(axis=0)
    prop_sig = np.delete(total_sig, ref_response - 1) / n_subs

    plt.bar(range(len(prop_sig)), prop_sig, color=sns.color_palette('crest'))
    plt.title(f'Sig. circular correlation w/ reponse {ref_response}')
    plt.ylabel('Proportion of subs')
    plt.ylim(0, 1)
    xlabels = np.delete(np.arange(set_size) + 1, ref_response - 1)
    plt.xticks(np.arange(5), labels=xlabels)
    plt.xlabel('response')

    fig, ax = plt.gcf(), plt.gca()
    set_axes_aspect(fig, [ax])

    return fig, ax

def plot_bootstrapped_kl(data_free, data_rand, ref_response=1, var='reported', n_iters=1000, 
                         kl_bins=8, sample_bins=8, empirical=False, color=None, **kwargs):
    """Generates and plots bootstrapped estimates of KL divergence between relative distance
       distributions and circular uniform samples. 

    Args:
        data_free (pandas.DataFrame): Dataframe containing data from only one modality and
                                      set size. Condition expected to be 'free'.
        data_rand (pandas.DataFrame): Dataframe containing data from only one modality and
                                      set size. Condition expected to be 'rand'.
        ref_response (int, optional): Response to compare to when computing relative distances. 
                                      Defaults to 1.
        var (str, optional): Variable to compute relative distance for ('reported' or 'presented').
                             Defaults to 'reported'. See utils.get_relative_dist() for info.
        n_iters (int, optional): Number of bootstrap iterations to perform. Defaults to 1000.
        kl_bins (int, optional): Number of bins to use when estimating KL div. Ideally, use 360
                                 bins for continuous dataset and 8 for discrete. Defaults to 8.
        sample_bins (int, optional): Number of bins to sample from when generating uniform samples.
                                     Use 360 for continuous, 8 for discrete. Defaults to 8.
        color (list of 3 RGB values, optional): Colors for plotting free, random, and baseline 
                                                sammples, respectively. Defaults to None (sns colors).
        **kwargs: passed to sns.boxplot() and sns.stripplot().

    Raises:
        Exception: Both dataframes must have a single set size (and both must be the same).

    Returns:
        fig (matplotlib.pyplot.figure): Figure handle.
        ax (matplotlib.pyplot.axis): Figure axis.
    """    

    set_size = data_free.reset_index().set_size[0]
    if data_free.set_size.nunique() > 1 or data_rand.set_size.nunique() > 1:
        raise Exception('Must pass df with only one set size')
    elif set_size != data_rand.reset_index().set_size[0]:
        raise Exception('Both dfs must have the same set size')

    kl_df_free = get_kl_df(data_free, n_iters=n_iters, condition='free', kl_bins=kl_bins,
                           empirical=empirical)
    kl_df_base_free = get_kl_df(data_free, n_iters=n_iters, uniform=True, 
                                sample_bins=sample_bins, condition='baseline_free',
                                kl_bins=kl_bins, empirical=empirical)
    
    kl_df_rand = get_kl_df(data_rand, n_iters=n_iters, condition='rand', kl_bins=kl_bins,
                           empirical=empirical)
    kl_df_base_rand = get_kl_df(data_rand, n_iters=n_iters, uniform=True, 
                                sample_bins=sample_bins, condition='baseline_rand',
                                kl_bins=kl_bins, empirical=empirical)

    kl_df = pd.concat([kl_df_free, kl_df_base_free, kl_df_rand, kl_df_base_rand])

    if color is None:
        color = [FREE_COLOR, (0.3, 0.3, 0.3), RAND_COLOR, (0.3, 0.3, 0.3)]

    ax = sns.boxplot(x='response', y='samples', hue='condition', data=kl_df, 
                     palette=color, showfliers=False, showcaps=False, linewidth=1, **kwargs)

    # give boxes transparency
    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .5))

    ax = sns.stripplot(x='response', y='samples', hue='condition', 
                       palette=color, jitter=True, data=kl_df, size=2, dodge=True, **kwargs)

    ax.legend([],[])
    ax.set_ylabel('KL div. from uniform (bits)')
    ax.set_xlabel('report number')
    set_box_widths(ax, 0.9)
    
    return plt.gcf(), ax

def plot_bias_ols_by_response(data, x_var, y_var, x_range=[-np.pi, np.pi], stats=True, palette=[FREE_COLOR, RAND_COLOR],
                              scatter_kws={'s': 10, 'alpha': 0.5}, line_kws=None, **kwargs):
    """ Plot regression results by response for data
    One set size, free & rand combined
    """
    data = data[(data[x_var] > x_range[0]) & (data[x_var] < x_range[1])]

    g = sns.lmplot(data=data, x=x_var, y=y_var, col='response', row='condition', 
                   hue='condition', palette=palette, 
                   legend=False, scatter_kws=scatter_kws, line_kws=line_kws, **kwargs)

    fig = plt.gcf() 
    axes = np.array(g.axes.flat)

    if stats:
        responses = np.sort(data['response'].unique())
        
        for ax, response in zip(axes, responses):
            df_free = data[(data['response'] == response) & 
                           (data['condition'] == 'free')]
            results = compute_bias_ols(df_free, x=x_var, y=y_var)
            coeff_free = np.round(results.params[x_var], decimals=3)
            r_sq_free = np.round(results.rsquared, decimals=3)
            p_free = np.round(results.f_pvalue, decimals=3)

            df_rand = data[(data['response'] == response) & 
                           (data['condition'] == 'rand')]
            results = compute_bias_ols(df_rand, x=x_var, y=y_var)
            coeff_rand = np.round(results.params[x_var], decimals=3)
            r_sq_rand = np.round(results.rsquared, decimals=3)
            p_rand = np.round(results.f_pvalue, decimals=3)

            txt_1 = f'response {response} \n'
            txt_2 = f'free: coeff = {coeff_free} | R^2 = {r_sq_free} | p = {p_free} \n'
            txt_3 = f'rand: coeff = {coeff_rand} | R^2 = {r_sq_rand} | p = {p_rand}'
            ax.set_title(txt_1 + txt_2 + txt_3)
            

    return fig, axes, g

def plot_bias_ols_by_k(data, x_var, y_var, x_range=[-np.pi, np.pi], stats=True, 
                              scatter_kws={'s': 10, 'alpha': 0.5}, line_kws=None, **kwargs):
    """ Plot regression results by kappa_obs for model estimates
    One set size
    """
    data = data[(data[x_var] > x_range[0]) & (data[x_var] < x_range[1])]

    g = sns.lmplot(data=data, x=x_var, y=y_var, col='kappa_obs', 
                   hue='kappa_obs', palette=KAPPA_OBS_PALETTE, 
                   legend=False, scatter_kws=scatter_kws, line_kws=line_kws, **kwargs)

    fig = plt.gcf() 
    axes = np.array(g.axes.flat)

    if stats:
        kappas = np.sort(data['kappa_obs'].unique())
        
        for ax, k in zip(axes, kappas):
            df_temp = data[data['kappa_obs'] == k]
            results = compute_bias_ols(df_temp, x=x_var, y=y_var)
            coeff = np.round(results.params[x_var], decimals=3)
            r_sq = np.round(results.rsquared, decimals=3)
            p = np.round(results.f_pvalue, decimals=3)

            txt_1 = f'kappa_obs = {k} \n'
            txt_2 = f'coeff = {coeff} | R^2 = {r_sq} | p = {p}'
            ax.set_title(txt_1 + txt_2)
            

    return fig, axes, g

def plot_ensemble_bias_by_response(data, bins=37, density=False, histtype='stepfilled', 
                        palette=RESP_PALETTE, sharey=False, compute_bias=True, **kwargs):
    """later
    """
    if compute_bias:
        data = compute_ensemble_bias(data)

    g = sns.FacetGrid(data, col='response', hue='response', sharex=False, 
                      sharey=sharey, palette=palette)
    g.map(weighted_hist, 'bias', bins=bins, range=(-300, 300), density=density, 
          histtype=histtype, edgecolor='none', **kwargs)

    fig = plt.gcf() 
    axes = np.array(g.axes.flat)

    if not sharey:
        ylim = axes[0].get_ylim()
        for ax in axes[1:]:
            ax.set_ylim(ylim)
    
    for ax in axes:
        ax.vlines(0, ylim[0], 0.05, color='red', linestyles='dashed', linewidth=0.5)
    axes[0].set_ylabel('proportion')

    return fig, axes

def plot_ensemble_bias_by_set_size(data, bins=37, density=False, histtype='stepfilled',
sharey=False, compute_bias=True, **kwargs):
    """Accepts a tidy dataframe and uses seaborn.FacetGrid to plot histogram of cumulative ensemble bias for every set size, collapsed across responses. 
    **kwargs are arguments for plt.hist
    `axes` output is a flat array of all axes.
    """
    g = sns.FacetGrid(data.dropna(), col='set_size', sharex=False, sharey=sharey)
    g.map(weighted_hist, 'bias', bins=bins, range=(-300, 300), density=density,
          histtype=histtype, edgecolor='none', **kwargs)

    fig = plt.gcf() 
    axes = np.array(g.axes.flat)

    if not sharey:
        ylim = axes[0].get_ylim()
        for ax in axes[1:]:
            ax.set_ylim(ylim)
    
    for ax in axes:
        ax.vlines(0, ylim[0], 0.05, color='red', linestyles='dashed', linewidth=0.5)
        
    ylabel = 'probability density' if density else 'proportion'
    g.set_axis_labels('bias', ylabel)

    return fig, axes

def plot_hbm_illustration(presented, mus, kappas, ensemble_mean, ensemble_kappa,
                           ensemble_color='k', palette=None):
    """Plots an illustration of a one-level hierarchical bayesian model.

    Args:
        presented (array-like): Array of n presented values.
        mus (array-like): Array of n posterior means. 
        kappas (array-like): Array of n posterior kappas. 
        ensemble_mean (scalar): Mean of ensemble posterior.
        ensemble_kappa (scalar): Kappa of ensemble posterior. 
        ensemble_color (string or RGB triplet, optional): Color to plot ensemble posterior.
        palette ([n x 3] array of RGB triplets, optional): Colors to plot posteriors. 
                                                           Defaults to Adam et al colorwheel. 

    Returns:
        fig (matplotlib.pyplot.figure): Figure handle.
        ax (matplotlib.pyplot.axis): Figure axis. 
    """    

    if palette is None:
        colorwheel = get_colorwheel()
        palette = colorwheel[np.round(np.degrees(presented) + 180).astype(int)]

    # plot individual posteriors
    for stim in range(len(presented)):
        plt.plot([presented[stim], presented[stim]], [0, 1], '--', c=palette[stim],
                 linewidth=1)
        x = np.linspace(-np.pi, np.pi, 360)
        plt.plot(x, vonmises_pdf(x, mus[stim], kappas[stim]), c=palette[stim])

    # plot top-level posterior
    pdf = vonmises_pdf(x, ensemble_mean, ensemble_kappa)
    plt.fill_between(x, pdf, pdf-pdf, color='k', alpha=0.2)

    fig = plt.gcf()
    ax = plt.gca()

    set_axes_rad(fig, [ax])

    return fig, ax

def plot_bfmm_illustration(presented, mus, kappas, cluster_mus, cluster_kappas, w,
                           ensemble_color='k', palette=None):
    """Plots an illustration of a bayesian finite mixture model.

    Args:
        presented (array-like): Array of n presented values.
        mus (array-like): Array of n posterior means. 
        kappas (array-like): Array of n posterior kappas. 
        cluster_mus (array-like): Means of cluster posteriors.
        cluster_kappas (array-like): Kappas of cluster posteriors. 
        w (array-like): Cluster weights.
        ensemble_color (string or RGB triplet, optional): Color to plot ensemble posterior.
        palette ([n x 3] array of RGB triplets, optional): Colors to plot posteriors. 
                                                           Defaults to Adam et al colorwheel. 

    Returns:
        fig (matplotlib.pyplot.figure): Figure handle.
        ax (matplotlib.pyplot.axis): Figure axis. 
    """    

    if palette is None:
        colorwheel = get_colorwheel()
        palette = colorwheel[np.round(np.degrees(presented) + 180).astype(int)]

    # plot individual posteriors
    for stim in range(len(presented)):
        plt.plot([presented[stim], presented[stim]], [0, 1], '--', c=palette[stim],
                 linewidth=1)
        x = np.linspace(-np.pi, np.pi, 360)
        plt.plot(x, vonmises_pdf(x, mus[stim], kappas[stim]), c=palette[stim])

    # plot cluster posteriors
    for cluster in range(len(cluster_mus)):
        pdf = w[cluster] * vonmises_pdf(x, cluster_mus[cluster], cluster_kappas[cluster])
        plt.fill_between(x, pdf, pdf-pdf, color='k', alpha=0.2)

    fig = plt.gcf()
    ax = plt.gca()

    set_axes_rad(fig, [ax])

    return fig, ax

def plot_model_bias(data, bins=36, var='theta_hat', bias_toward='presented_rad_mean', 
                      palette=KAPPA_OBS_PALETTE, fit_sin=False, discrete=False):

    if data.set_size.nunique() > 1:
        raise Exception('Must pass df with only one set size')
    
    set_size = data.reset_index()['set_size'][0]
    
    kappas = np.sort(data['kappa_obs'].unique())

    if discrete:
        bins = DISC_EDGES

    fig, ax = plt.subplots()
    for ind, kappa_obs in enumerate(kappas):
        df_temp = data[data['kappa_obs'] == kappa_obs].reset_index()
        
        # plot bias
        x, mean, std = compute_bias_stats(df_temp, var=var, 
                                          bias_toward=bias_toward, bins=bins)

        plot_shaded_error(x, mean, std, color=palette[ind])

        if fit_sin:
            X, Y = compute_relative_error(df_temp, var=var, bias_toward=bias_toward)
            a = fit_phase_locked_sine(X, Y)
            x_plot = np.linspace(-np.pi, np.pi, 100)
            ax.plot(x_plot, phase_locked_sine(x_plot, a), '--', c=palette[ind])

    ax.set_ylim([-np.pi/2, np.pi/2])
    ax.set_yticks([-np.pi/2, 0, np.pi/2])
    ax.set_yticklabels([r'$-\frac{\pi}{2}$', '0', r'$\frac{\pi}{2}$'])
    ax.set_xlabel('target distance from mean (rad)')
    ax.set_ylabel('report error (rad)')
    set_axes_rad(fig, [ax])
    
    return fig, ax

def plot_data_bias(data_free, data_rand, bins=36, var='reported_rad', 
                   bias_toward='presented_rad_mean', fit_sin=False, discrete=False, palette=[FREE_COLOR, RAND_COLOR]):

    set_size = data_free.reset_index().set_size[0]
    if data_free.set_size.nunique() > 1 or data_rand.set_size.nunique() > 1:
        raise Exception('Must pass df with only one set size')
    elif set_size != data_rand.reset_index().set_size[0]:
        raise Exception('Both dfs must have the same set size')

    if discrete:
        bins = DISC_EDGES

    x, mean, std = compute_bias_stats(data_free, var=var, 
                                      bias_toward=bias_toward, bins=bins)
    fig, ax = plot_shaded_error(x, mean, std, color=palette[0])

    x, mean, std = compute_bias_stats(data_rand, var=var, 
                                      bias_toward=bias_toward, bins=bins)
    plot_shaded_error(x, mean, std, color=palette[1])

    if fit_sin:
        x_plot = np.linspace(np.min(x), np.max(x), 100)
        X, Y = compute_relative_error(data_free, var=var, 
                                      bias_toward=bias_toward)
        a = fit_phase_locked_sine(X, Y)
        ax.plot(x_plot, phase_locked_sine(x_plot, a), '--', c=palette[0])

        X, Y = compute_relative_error(data_rand, var=var, 
                                      bias_toward=bias_toward)
        a = fit_phase_locked_sine(X, Y)
        ax.plot(x_plot, phase_locked_sine(x_plot, a), '--', c=palette[1])

    ax.set_ylim([-np.pi/2, np.pi/2])
    ax.set_yticks([-np.pi/2, 0, np.pi/2])
    ax.set_yticklabels([r'$-\frac{\pi}{2}$', '0', r'$\frac{\pi}{2}$'])
    ax.set_xlabel('target distance from mean (rad)')
    ax.set_ylabel('report error (rad)')
    set_axes_rad(fig, [ax])

    return fig, ax