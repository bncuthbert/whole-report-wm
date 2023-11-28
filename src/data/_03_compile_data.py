#!/usr/bin/env python

"""This script is intended to be called by the repo's makefile. It compiles the cleaned, pickled
dataframes in '/data/02_interim/' into a single tidy dataframe, and saves the result
to '/data/03_processed/'.
"""

import pandas as pd
import numpy as np
from utils import min_angle
from data import get_file_list, INTERIM_DIR, PROCESSED_DIR

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)            

DATASETS = ['discrete', 'continuous']
CONDITIONS = ['free', 'rand']
MODALITIES = ['color', 'orient']

def load_data(modality=None, condition=None, dataset='continuous', prefix=''):
    """Utility function for loading datasets.

    Args:
        modality (str, optional): 'color' or 'orient'. Defaults to None (both included)
        condition (str, optional): 'free' or 'rand'. Defaults to None (both included).
        dataset (str, optional): 'continuous' or 'discrete'. Defaults to 'continuous'.

    Returns:
        pandas.DataFrame: Filtered dataframe. Redundant columns dropped (ex if filtered by
                          'condition' then the condition column is dropped). 
    """
    
    df = pd.read_pickle(f'{prefix}../{PROCESSED_DIR}{dataset}_data.pickle')

    if modality is not None and condition is not None:
        df = df[(df.modality == modality) & (df.condition == condition)]
        df.drop(['modality', 'condition'], axis=1, inplace=True)
    elif modality is not None and condition is None:
        df = df[df.modality == modality]
        df.drop('modality', axis=1, inplace=True)
    elif modality is None and condition is not None:
        df = df[df.condition == condition]
        df.drop('condition', axis=1, inplace=True)
    
    if dataset == 'continuous': 
        # remove unnecessary columns 
        df.dropna(axis=1, how='all', inplace=True)

    return df

def get_within_trial_dist(lst0, lst1, lst2):
    def within_trial_dist(group):
        """To be applied as a pandas groupby operation. 
        Input `group` is a dataframe corresponding to one specific trial 
        (ie data grouped by ['subject','block','trial'])
        """

        for ind, response in enumerate(group['response']):
            temp_angle = np.array(group[group.response == response]['presented_deg'])
            temp_dist = min_angle(temp_angle, np.array(group['presented_deg']))
            temp_dist[ind] = np.nan
            lst0.append(temp_dist)

            temp_angle = np.array(group[group.response == response]['reported_deg'])
            temp_dist = min_angle(temp_angle, np.array(group['reported_deg']))
            temp_dist[ind] = np.nan
            lst1.append(temp_dist)

            lst2.append(group.index[ind])

        return 0
   
    return within_trial_dist

def compute_relative_dist(df):
    """Returns the dataframe with two additional columns of type 'object': 
    - 'presented_dist' contains numpy arrays with the relative angular distance (in degrees) 
        between each presented angle and all others in the trial
    - 'reported_dist' contains numpy arrays with the relative angular distance (in degrees) 
        between each reported angle and all others in the trial
    """

    presented_dist = []
    reported_dist = []
    index = []
    
    within_trial_dist = get_within_trial_dist(presented_dist, reported_dist, index)
    df.groupby(['subject', 'block', 'trial']).apply(within_trial_dist)

    presented_dist = np.array(presented_dist)[np.argsort(index)]
    reported_dist = np.array(reported_dist)[np.argsort(index)]

    df.insert(df.columns.get_loc('presented_deg') + 1, 'presented_dist', presented_dist)
    df.insert(df.columns.get_loc('reported_deg') + 1, 'reported_dist', reported_dist)
    
    return df

def drop_incomplete_trials(df, var='reported_deg'):
    """Finds all df rows (ie responses) where `var` is nan, 
    then drops all other rows/responses from the same trial.
    """
    df_no_response = df[df[var].isna()].copy()

    ind_bad = []
    inds_old = []

    for index, row in df_no_response.iterrows():
        trial_info = row[['subject', 'block', 'trial', 'set_size']]

        df_bad = df[(df['subject'] == trial_info['subject']) &
                    (df['block'] == trial_info['block']) &
                    (df['trial'] == trial_info['trial'])]
        
        inds = df_bad.index.values

        if not np.array_equal(inds, inds_old):
            ind_bad.append(inds)
            inds_old = inds

    print(f'    dropped {len(ind_bad)} incomplete trials')
    ind_bad = np.concatenate(ind_bad)

    df_complete = df.drop(index=ind_bad).copy()

    return df_complete.reset_index(drop=True)

if __name__ == '__main__':
    suffix = '_clean.pickle'
    df_names = get_file_list(INTERIM_DIR, suffix)
    exp_names = [df_name[:-len(suffix)] for df_name in df_names]

    ## continuous dataset ##
    conditions = ['free', 'free', 'rand', 'rand']
    modalities = ['color', 'orient', 'color', 'orient']

    dfs = {}
    for df_name, exp_name, condition, modality in zip(df_names[2:], exp_names[2:], 
                                                  conditions, modalities):
        dfs[exp_name] = pd.read_pickle(f'{INTERIM_DIR}{df_name}')
        dfs[exp_name]['condition'] = condition
        dfs[exp_name]['modality'] = modality

    n_subs_1a = len(dfs['exp_1a']['subject'].unique())
    dfs['exp_2a']['subject'] += n_subs_1a

    try:
        df_continuous = pd.concat(list(dfs.values())).reset_index(drop=True)
        cols = list(df_continuous.columns)
        new_cols = [cols[-3], cols[-2]] + cols[:7] + [cols[-1]] + cols[8:-3]
        df_continuous = df_continuous[new_cols]
    except ValueError as e:
        print(f'ValueError: {e}')

    print(' dropping incomplete trials...')
    df_continuous = drop_incomplete_trials(df_continuous)

    print(f' computing continuous dataset stats... (may take a couple of minutes)')
    
    df_continuous = compute_relative_dist(df_continuous)
    print(' complete')

    save_name = f'{PROCESSED_DIR}continuous_data.pickle'
    df_continuous.to_pickle(save_name)
    print(f' continuous dataset saved to {save_name}')

    ## discrete dataset ##
    df_free = pd.read_pickle(f'{INTERIM_DIR}discrete_free_clean.pickle')
    df_free.insert(0, 'modality', 'color')
    df_free.insert(0, 'condition', 'free')

    df_rand = pd.read_pickle(f'{INTERIM_DIR}discrete_rand_clean.pickle')
    
    df_rand.insert(0, 'modality', 'color')
    df_rand.insert(0, 'condition', 'rand')

    print(f' computing discrete dataset stats... (may take a couple of minutes)')
    df_free = compute_relative_dist(df_free)
    df_rand = compute_relative_dist(df_rand)

    df_discrete = pd.concat([df_free, df_rand])
    save_name = f'{PROCESSED_DIR}discrete_data.pickle'
    df_discrete.to_pickle(save_name)
    print(f' discrete dataset saved to {save_name}')