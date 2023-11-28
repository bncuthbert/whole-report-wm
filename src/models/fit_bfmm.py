#!/usr/bin/env python

"""Fit one-level hierarchical bayesian model and save results"""

from data import load_data
from utils import get_trial_df
from models import get_bfmm, fit_bfmm, SAVE_DIR
import pickle
import sys
import getopt
import numpy as np
import pandas as pd
from os import path

def parse_args(argv):
    n_to_fit = None

    try:
        opts, args = getopt.getopt(argv,"hm:c:d:s:k:K:n:")
    except getopt.GetoptError:
        print('use \'-h\' to see options')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('options: \n')
            print('    -m <modality> [\'color\' or \'orient\']')
            print('    -c <condition> [\'free\' or \'rand\']')
            print('    -d <dataset> [\'continuous\' or \'discrete\']')
            print('    -s <set_size> [int]')
            print('    -k <kappa_obs> [float]')
            print('    -K <K_components> [float]')
            print('    -n <n_to_fit> [int] (# of trials to fit; defaults to fitting all trials)')

            sys.exit(2)
        elif opt == '-m':
            modality = arg
        elif opt == '-c':
            condition = arg
        elif opt == '-d':
            dataset = arg
        elif opt == '-s':
            set_size = arg
        elif opt == '-k':
            kappa_obs = arg
        elif opt == '-K':
            K = arg
        elif opt == '-n':
            n_to_fit = arg

    data_params = {'modality': modality,
                   'condition': condition,
                   'dataset': dataset,
                   'set_size': int(set_size)}    
    
    model_params = {'kappa_obs': float(kappa_obs),
                    'K': int(K)}

    return data_params, model_params, n_to_fit

if __name__ == "__main__":

    data_params, model_params, n_to_fit = parse_args(sys.argv[1:])
    save_path = f"../{SAVE_DIR}{data_params['modality']}/{data_params['condition']}/{data_params['dataset']}/"
    fit_name = f"bfmm{model_params['K']}_ss_{data_params['set_size']}_k_{int(model_params['kappa_obs'])}"
    save_name = f'{save_path}{fit_name}.pickle'

    # load data 
    df_temp = load_data(modality=data_params['modality'], 
                        condition=data_params['condition'],
                        dataset=data_params['dataset'],
                        prefix='../')
    df_temp = df_temp[df_temp['set_size'] == data_params['set_size']].copy()
    df_to_fit = get_trial_df(df_temp, ['presented_rad', 'reported_rad']).reset_index()
    total_trials = len(df_to_fit)
    if n_to_fit is None:
        n_to_fit = total_trials
    else:
        n_to_fit = int(n_to_fit)
    
    # check if fit already exists
    if path.isfile(save_name):
        print(f'existing fit found at {save_name}')
        with open(save_name, 'rb') as p:
            df_fit_old = pickle.load(p)

        finished_trials = len(df_fit_old)
        print(f'{finished_trials} / {total_trials} trials already complete')
        
    else:
        print(f'no existing fit found at {save_name}, starting from scratch')
        df_fit_old = None

    # determine which trials need to be fit
    if df_fit_old is not None:

        if n_to_fit > (total_trials - finished_trials):
            n_to_fit = total_trials - finished_trials

        start_ind = finished_trials
        stop_ind = finished_trials + n_to_fit
        df_to_fit = df_to_fit[start_ind:stop_ind].copy()

    else:
        df_to_fit = df_to_fit[:n_to_fit].copy()

    print(f'fitting {n_to_fit} trials...')

    # get model and fit
    bfmm = get_bfmm(set_size=data_params['set_size'],
                    K=model_params['K'],
                    kappa_obs=model_params['kappa_obs'])

    df_fit = fit_bfmm(df_to_fit, bfmm, model_params['kappa_obs'])
    
    # concatenate old + new fits
    if df_fit_old is not None:
        df_fit = pd.concat([df_fit_old, df_fit]).reset_index(drop=True)
    
    df_fit.attrs = {'data_params':data_params, 'model_params':model_params}

    print(f'\n Done fitting, saving to {save_name}')

    with open(save_name, 'wb') as p:
        pickle.dump(df_fit, p)
    print('\n Saved')