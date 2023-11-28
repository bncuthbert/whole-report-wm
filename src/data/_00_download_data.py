#!/usr/bin/env python

"""Uses `subprocess` to download required files from osf programmatically. Has
potentially useful python wrappers for fetching file lists and individual files
from osf, but slow osf-client command-line calls mean that it's still faster to 
download entire folders and delete the extras later.
"""

import subprocess
import os.path
from data import RAW_DIR, PROCESSED_DIR

OSF_CODES = ['u22ez', '2qyxp', '9agbw', 'y5rhb', '97n3w']
EXP_NAMES = ['exp_1a', 'exp_1b', 'exp_2a', 'exp_2b', 'exp_3']

def list_osf_files(osf_code):
    """Returns 2 lists of all files contained in the osf project
    specified by the string `osf_code`. `path_list` includes full paths, 
    `file_list` only contains filenames.
    """
    command = f'osf -p {osf_code} list'.split()

    try:
        byte_string = subprocess.check_output(command)
    except:
        raise Exception('Failed to access osf project {osf_code}')
    
    path_list = []
    file_list = []

    for file_path in byte_string.decode('utf-8').split('\n'):
        path_list.append(file_path)
        file_list.append(file_path.split('/')[-1])

    return path_list, file_list

def fetch_osf_file(osf_code, file_path, save_path=None):
    """Wraps osf cli to download a file specified by the strings `osf_code` and
    `file_path`. Saves file to pwd if `save_path` not provided.
    """
    command = f'osf -p {osf_code} fetch'.split()
    command.append(file_path)

    if save_path is not None:
        command.append(save_path)

    try:
        subprocess.run(command, capture_output=True)
    except:
        raise Exception('Failed to access {file_path} in osf project {osf_code}')

if __name__ == '__main__':
    
    for exp, osf_code in zip(EXP_NAMES, OSF_CODES):
        exp_path = f'{RAW_DIR}{exp}/'      

        if os.path.isdir(exp_path):
            print(f' {exp} already downloaded')
        else:
            print(f' {exp} not found')
            print(' downloading from OSF now... (this may take a while)')    

            # download osf folder
            command = f'osf -p {osf_code} clone {exp_path}'
            subprocess.run(command, shell=True)
            print(' download complete')

            # copy .csv files
            if exp == 'exp_1a': # exp 1a has different dir name
                csv_path = 'osfstorage/Individual\ Files\ -\ CSV/'

                # grab colorwheel.mat while we're here
                command = f'cp {exp_path}osfstorage/Task\ Code/colorwheel360.mat {PROCESSED_DIR}'
                subprocess.run(command, shell=True)

            else:
                csv_path = 'osfstorage/IndividualFiles\ -\ CSV/'
                
            command = f'cp {exp_path}{csv_path}*.csv {exp_path}'
            subprocess.run(command, shell=True)

            # unzip and copy .mat files
            command = f'python src/utils/unzip.py {exp_path}osfstorage/IndividualFiles_Matlab.zip {exp_path}'
            subprocess.run(command, shell=True)

            command = f'cp {exp_path}IndividualFiles_Matlab/*.mat {exp_path}'
            subprocess.run(command, shell=True)

            # clean up
            command = f'rm -rf {exp_path}osfstorage {exp_path}IndividualFiles_Matlab {exp_path}__MACOSX'
            subprocess.run(command, shell=True)

            print(f' .csv and .mat files saved to {exp_path}')