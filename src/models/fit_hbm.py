#!/usr/bin/env python

"""Fit two-level hierarchical bayesian model and save results"""

from data import load_data
from utils import get_trial_df
from models import get_hbm, fit_hbm, SAVE_DIR
import pickle
import sys
import getopt

def parse_args(argv):
    try:
        opts, args = getopt.getopt(argv,"hm:c:d:s:k:")
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

    data_params = {'modality': modality,
                   'condition': condition,
                   'dataset': dataset,
                   'set_size': int(set_size)}    
    
    model_params = {'kappa_obs': float(kappa_obs)}

    return data_params, model_params

if __name__ == "__main__":

    data_params, model_params = parse_args(sys.argv[1:])

    # fit
    df_temp = load_data(modality=data_params['modality'], 
                        condition=data_params['condition'],
                        dataset=data_params['dataset'],
                        prefix='../')
    df_temp = df_temp[df_temp['set_size'] == data_params['set_size']].copy()
    df_to_fit = get_trial_df(df_temp, ['presented_rad'])

    hbm = get_hbm(set_size=data_params['set_size'], 
                            kappa_obs=model_params['kappa_obs'])
    
    df_fit = fit_hbm(df_to_fit, hbm, model_params['kappa_obs'])
    df_fit.attrs = {'data_params':data_params, 'model_params':model_params}
    
    save_path = f"../{SAVE_DIR}{data_params['modality']}/{data_params['condition']}/{data_params['dataset']}/"
    fit_name = f"hbm_ss_{data_params['set_size']}_k_{int(model_params['kappa_obs'])}"
    save_name = f'{save_path}{fit_name}.pickle'
    print(f'\n Done fitting, saving to {save_name}')

    with open(save_name, 'wb') as p:
        pickle.dump(df_fit, p)
    print('\n Saved')