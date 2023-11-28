#!/usr/bin/env python

"""This script is intended to be called by the repo's makefile. It iterates through the raw
data folders in '/data/01_raw/' , extracts data from all raw .mat and .csv files in each folder,
and compiles them into a single pandas dataframe. Each dataframe is then written to 
'/data/02_interim/' as a binary (pickled) file.
"""

import pandas as pd
import numpy as np
import os
from scipy.io import loadmat
from utils import get_response_df

RAW_DIR = 'data/01_raw/'
INTERIM_DIR = 'data/02_interim/'
PROCESSED_DIR = 'data/03_processed/'
DISCRETE_PATH = RAW_DIR + 'discrete/discrete_WR_compiled_10-Apr-2018.mat'

COL_FEATURES = ['presentedColor', 'reportedColor']
ORIENT_FEATURES = ['presentedAngle', 'reportedAngle']

def get_file_list(directory, suffix):
    """Returns a list of all filenames in `directory` ending with the string
    'suffix'. Lists are sorted lexicographically.
    """
    file_list = []
    for _, _, files in os.walk(directory):
        for file in files:
            if file.endswith(suffix):
                file_list.append(file)
            
    file_list.sort()

    return file_list

def extract_mat_features(loaded_mat, condition):
    """Extracts features stored in a structured numpy array created by
    using `scipy.io.loadmat()` to import matlab structs. Condition determines
    feature names and saved format.  
    """
    
    if condition == 'color':
        set_size = loaded_mat['data'][0][0]['setSize'][0]
        trial_offset = loaded_mat['prefs'][0][0]['ind'].flatten()
        presented = loaded_mat['data'][0][0][COL_FEATURES[0]]
        reported = loaded_mat['data'][0][0][COL_FEATURES[1]]

    elif condition == 'orientation':
        set_size = loaded_mat['data'][0][0]['setSize'].flatten('F')
        trial_offset = np.empty_like(set_size, dtype=np.float32)
        trial_offset[:] = np.nan
        presented = loaded_mat['data'][0][0][ORIENT_FEATURES[0]]
        reported = loaded_mat['data'][0][0][ORIENT_FEATURES[1]]
        presented = presented.transpose(2, 0, 1).reshape(-1, presented.shape[1])
        reported = reported.transpose(2, 0, 1).reshape(-1, reported.shape[1])

    return presented, reported, set_size, trial_offset

def compile_mat_csv(raw_path, condition):
    """**for Adam et al (continuous) dataset only**
    compiles all .csv and .mat files in dir 'raw_path' into one pandas df"""

    csv_list = get_file_list(raw_path, '.csv')
    mat_list = get_file_list(raw_path, '.mat')

    # read csvs into dfs & add data from mats
    df = None
    for ind, csv_name in enumerate(csv_list):
        csv_path = raw_path + csv_name
        mat_name = mat_list[ind]
        mat_path = raw_path + mat_name
        
        temp_df = pd.read_csv(csv_path)
        temp_mat = loadmat(mat_path)

        # fill in missing subject_ids 
        subject_id = int(''.join(list(filter(str.isdigit, mat_name))))
        temp_df['Subject'] = subject_id

        temp_presented, temp_reported, set_size, trial_offset = \
            extract_mat_features(temp_mat, condition)

        # pad missing offsets with nan
        # (not all trials were completed, so offsets were not always generated)
        n_trials = len(set_size)
        if len(trial_offset) < n_trials:
            pad = np.empty(n_trials - len(trial_offset))
            pad[:] = np.nan
            trial_offset = np.concatenate([trial_offset, pad])

        # task code saved offsets differently for different subjects
        # (see lines 113-117 in `Cont_Free_ALL.m` and lines 127-135 in `Cont_Rand_ALL.m`
        # to see what is being reverse-engineered here and below)
        if 'free' in mat_name:
            last_sub = 11
        else:
            last_sub = 8
        
        n_responses = sum(set_size) 
        presented = np.empty(n_responses)
        reported = np.empty(n_responses)
        response_offset = np.empty(n_responses)

        # flatten (trial x response) 2-D arrays into 1-D array of responses 
        for trial, ss in enumerate(set_size):
            ind = sum(set_size[:trial])
            presented[ind:ind + ss] = temp_presented[trial][:ss]
            reported[ind:ind + ss] = temp_reported[trial][:ss]

            if subject_id  <= last_sub: # offset was cumulative across trials
                response_offset[ind:ind + ss] = int(np.nansum(trial_offset[:trial + 1]))
            else: # offset reset every trial
                response_offset[ind:ind + ss] = trial_offset[trial]
        
        temp_df['offset'] = response_offset
        temp_df['presented'] = presented
        temp_df['reported'] = reported

        # compile
        if df is None:
            df = temp_df 
        else:
            df = pd.concat([df, temp_df], ignore_index=True)

    return df

def discrete_mat_to_trial_df(loaded_mat, drop_bad=True, drop_mid_task=True):
    """**For discrete whole-report dataset only**
    Loads discrete .mat file and compiles into a df with one row per trial.
    """
    df = pd.DataFrame(loaded_mat['masterParams'])
    
    # rename metadata columns
    metadata_names = ['subject', 'block', 'trial', 'condition', 'bad', 'set_size']
    metadata_inds = np.arange(6)
    metadata_dict = dict(zip(metadata_inds, metadata_names))
    df.rename(columns=metadata_dict, inplace=True)
    df[metadata_names] = df[metadata_names].astype('int32')

    # rename data columns (stim position, reported color, presented color, RT)
    set_size_max = 8 
    position_names = [f'position_{r}' for r in np.arange(set_size_max) + 1]
    reported_names = [f'reported_{r}' for r in np.arange(set_size_max) + 1]
    presented_names = [f'presented_{r}' for r in np.arange(set_size_max) + 1]
    rt_names = [f'rt_{r}' for r in np.arange(set_size_max) + 1]

    all_names = [position_names, reported_names, presented_names, rt_names]
    data_names = [name for sublist in all_names for name in sublist]

    data_inds = np.arange(6,38)
    data_dict = dict(zip(data_inds, data_names))
    df.rename(columns=data_dict, inplace=True)

    # reorder presented
    """The original data matrix stored both target position and reported colors in the
    order that they were reported, but idiosyncratically stored presented colors in
    order of *target position*. This re-orders presented colors by report order for
    consistancy with other variables (and separating by response downstream).
    """
    set_sizes = df['set_size']
    positions = df[position_names].to_numpy()
    reported = df[reported_names].to_numpy()
    presented = df[presented_names].to_numpy()
    presented_reordered = presented.copy()
    rts = df[rt_names].to_numpy()

    for row in np.arange(len(df)):
        # reorder presented
        temp_presented = presented[row, :set_sizes[row]]
        temp_ind = (positions[row, :set_sizes[row]] - 1).astype(int)
        presented_reordered[row, :set_sizes[row]] = temp_presented[temp_ind]

        # replace super-threshold (junk) values while we're here
        for var in [positions, reported, presented_reordered, rts]:
            var[row, set_sizes[row]:] = np.nan

    # re-pack all data and drop vestigial columns and rows
    df[position_names] = positions
    df[reported_names] = reported
    df[presented_names] = presented_reordered
    df[rt_names] = rts
    df.drop(np.arange(38,54), axis=1, inplace=True)

    if drop_bad:
        print(f'dropping {len(df[df.bad == 1])} discrete trials (marked as bad in .mat file)')
        df.drop(df[df.bad == 1].index, inplace=True)
        df.drop('bad', axis=1, inplace=True)
        
    if drop_mid_task:
        df.drop(df[df.condition == 7].index, inplace=True)

    return df.reset_index(drop=True)

def discrete_trial_to_response_df(trial_df):
    """**For distrete whole-report dataset only**
    Splits trial_df by set size, converts each to a response_df, and concatenates"""

    set_sizes = np.sort(trial_df['set_size'].unique())
    split_dfs = []

    for ss in set_sizes:
        temp_df = trial_df[trial_df['set_size'] == ss].copy()
        for col in temp_df.columns:
            if col[-1].isdigit():
                if int(col[-1]) > ss:
                   temp_df.drop(col, axis=1, inplace=True)
        
        temp_df = get_response_df(temp_df, var_list=['position', 'reported',
                                  'presented', 'rt'])
        split_dfs.append(temp_df)

    return pd.concat(split_dfs).reset_index(drop=True)


if __name__ == '__main__':

    # Adam et al (continuous) data
    exp_list = []
    for root, dirs, files in os.walk(RAW_DIR):
        for dir_name in dirs:
            if dir_name.startswith('exp'):
                exp_list.append(dir_name)
    exp_list.sort()
  
    for exp in exp_list:
        raw_path = f'{RAW_DIR}{exp}/'
        save_name = f'{exp}_dirty.pickle'
        
        # compile dataframe 
        if exp.endswith('a'):
            condition = 'color'
        else: 
            condition = 'orientation' 

        df = compile_mat_csv(raw_path, condition)

        # save
        save_path = f'{INTERIM_DIR}{save_name}'
        df.to_pickle(save_path)
        print(f' {exp} .csv and .mat files extracted and saved to {save_path}')

    # discrete data
    temp_mat = loadmat(DISCRETE_PATH)
    trial_df = discrete_mat_to_trial_df(temp_mat)
    trial_df_free = trial_df[trial_df['condition'] == 1].copy()
    trial_df_rand = trial_df[trial_df['condition'] == 8].copy()

    response_df_free = discrete_trial_to_response_df(trial_df_free)
    save_name = 'discrete_free_dirty.pickle'
    save_path = f'{INTERIM_DIR}{save_name}'
    response_df_free.to_pickle(save_path)
    print(f' discrete_free .mat file extracted and saved to {save_path}')

    response_df_rand = discrete_trial_to_response_df(trial_df_rand)
    save_name = 'discrete_rand_dirty.pickle'
    save_path = f'{INTERIM_DIR}{save_name}'
    response_df_rand.to_pickle(f'{INTERIM_DIR}{save_name}')
    print(f' discrete_rand .mat file extracted and saved to {save_path}')
