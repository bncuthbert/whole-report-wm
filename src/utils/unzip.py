#!/usr/bin/env python

"""Script to perform simple os-agnostic unzipping. 
Syntax: 'python unzip.py <path/to/archive> <path/to/unzip/to>
"""

import sys
import zipfile

if __name__ == '__main__':
    archive_path = sys.argv[1]
    target_path = sys.argv[2]
    
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        zip_ref.extractall(target_path)