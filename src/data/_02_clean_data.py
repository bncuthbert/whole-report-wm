#!/usr/bin/env python

"""This script is intended to be called by the repo's makefile. It iterates through the 
pickled dataframes in '/data/02_interim/' and cleans them (for details see 'clean_df()').
It also combines data from two experiments (exp_1b and exp_3) because they are effectively the same.

Missing data is left in place (as np.nan), and cleaned dataframes are saved to '/data/02_interim/'
as '{original filename}_clean.pickle'
"""

import pandas as pd
import numpy as np
from data import get_file_list, INTERIM_DIR
from utils import min_angle

def clean_df(df, condition):
    """Accepts a "dirty" dataframe (output of 'extract_data.py') and performs numerous
    cleaning operations on the contents, including:
        - changing column names to snake_case and removing vestigial columns
        - replacing '9999' and eps values with np.nan and 0
        - correcting the range of all presented/reported angles
        - creating columns with presented/reported/error angles in both radians and degrees 
        - condition-specific cleaning (ex accounting for random color wheel offset)
        - correcting reaction times with 'correct_rts()'
    The resulting dataframe is verified (by comparison with raw csv data) and returned.
    """

    df.columns = df.columns.str.lower().str.replace(' ','_')
    df[df == 9999] = np.nan   

    df['presented_deg'] = np.round(np.degrees(df['presented']))
    df['reported_deg'] = np.round(np.degrees(df['reported']))

    df['error_deg'] = df['offset_error']
    df.loc[df['error_deg'] == -180, 'error_deg'] = 180
    tol = 1e-10
    df.loc[abs(df['error_deg']) < tol, 'error_deg'] = 0

    # condition-specific cleaning 
    if condition == 'orientation': # need to correct degree range
        df['presented_deg'] = (df['presented_deg'] % 360) - 179
        df['reported_deg'] = (df['reported_deg'] % 360) - 179
    elif condition == 'color': # need to account for random color wheel offset
        df['presented_deg'] = (df['presented_deg'] + df['offset']) % 360 - 179      
        df['reported_deg'] = (df['reported_deg'] + df['offset']) % 360 - 179
    
    # compare against raw errors to verify
    bad, _ = check_errors(df)
    if any(bad):
        raise Exception('Raw and computed offset_error mismatch')

    # add radians and remove unneeded columns
    df.insert(df.columns.get_loc('error_deg'), 
        'error_rad', np.radians(df['error_deg']))
    df.insert(df.columns.get_loc('presented_deg'), 
        'presented_rad', np.radians(df['presented_deg']))
    df.insert(df.columns.get_loc('reported_deg'), 
        'reported_rad', np.radians(df['reported_deg']))
    df.drop(['offset_error', 'offset', 'presented', 'reported'], axis=1, inplace=True)
    
    return df

def clean_discrete_df(df):
    """**for discrete whole-report dataset only**
    Performs the following cleaning operations:
    - converts position to int
    - converts presented/reported indices to rad/deg and adds both as columns
    - computes error in both rad/deg and adds columns

    """
    df[['position', 'reported', 'presented']] = df[['position', 
                                                'reported', 'presented']].astype(int)

    angles_rad = np.linspace(- 3 / 4 * np.pi, np.pi, 8)
    angles_deg = np.degrees(angles_rad)

    df.insert(len(df.columns), 'presented_rad', angles_rad[df.presented - 1])
    df.insert(len(df.columns), 'presented_deg', angles_deg[df.presented - 1])
    df.insert(len(df.columns), 'reported_rad', angles_rad[df.reported - 1])
    df.insert(len(df.columns), 'reported_deg', angles_deg[df.reported - 1])

    df.drop(['reported', 'presented'], axis=1, inplace=True)

    df['error_rad'] = min_angle(df['presented_rad'], df['reported_rad'], radians=True)
    df['error_deg'] = min_angle(df['presented_deg'], df['reported_deg'])

    return df

def correct_rts(df):
    """Accepts a dataframe with columns ['rt_1', 'rt_2'] or 'rt' containing reaction
    times relative to trial start. Returns a datafram with the same columns corrected
    to contain reaction times relative to the last response.
    """

    rts = df.columns[df.columns.str.startswith('rt')]
    ilocs = df.index[df['response'] > 1]

    if len(rts) > 1:
        rt_1_loc = df.columns.get_loc('rt_1')
        rt_2_loc = df.columns.get_loc('rt_2')
        after = df.iloc[ilocs, rt_1_loc].reset_index(drop=True)
        before = df.iloc[ilocs - 1, rt_2_loc].reset_index(drop=True) 
        df['rt_2'] = df['rt_2'] - df['rt_1']
        df.iloc[ilocs, rt_1_loc] = (after - before).to_numpy()

    else:
        rt_loc = df.columns.get_loc('rt')
        after = df.iloc[ilocs, rt_loc].reset_index(drop=True)
        before = df.iloc[ilocs - 1, rt_loc].reset_index(drop=True)
        df.iloc[ilocs, rt_loc] = (after - before).to_numpy()

    return df

def check_errors(df):
    """Accepts a dataframe with columns 'presented_deg', 'reported_deg', and
    'error_deg'. Computes min angle between 'presented' and 'reported', and compares
    the result to 'error_deg'. Returns 'bad' (Boolean index of mismatches) and 
    'test_error' (pandas.Series of min angles)
    """

    df = df.dropna()
    test_error = min_angle(df['presented_deg'], df['reported_deg'])
    bad = df['error_deg'] != test_error
    
    return bad, test_error

if __name__ == '__main__':
    suffix = '_dirty.pickle'
    df_list = get_file_list(INTERIM_DIR, suffix)

    for df_file in df_list:
        if df_file.startswith('exp'): # continuous dataset
            exp = df_file[:-len(suffix)]
            if exp.endswith('a'):
                condition = 'color'
            else:
                condition = 'orientation'
            
            df_dirty = pd.read_pickle(f'{INTERIM_DIR}/{df_file}')
            df = clean_df(df_dirty, condition)

            if exp != 'exp_3':
                save_path = f'{INTERIM_DIR}{exp}_clean.pickle'
            elif exp == 'exp_3': # combine with exp_1b
                save_path = f'{INTERIM_DIR}exp_1b_clean.pickle'
                df2 = pd.read_pickle(save_path)
                df = pd.concat([df2, df], ignore_index=True) 

            df.to_pickle(save_path)
            print(f' {exp} dataframe cleaned and saved to {save_path}')

        elif df_file.startswith('discrete'): # discrete dataset
            exp = df_file[:-len(suffix)]
            df_dirty = pd.read_pickle(f'{INTERIM_DIR}/{df_file}')
            df = clean_discrete_df(df_dirty)
            save_path = f'{INTERIM_DIR}{exp}_clean.pickle' 
            df.to_pickle(save_path)
            print(f' {exp} dataframe cleaned and saved to {save_path}')