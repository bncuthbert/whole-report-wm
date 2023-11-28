#!/usr/bin/env python

from setuptools import setup

setup(name='whole-report-wm',
      version='0.1',
      description='Code for Cuthbert et al. 2023',
      url='https://github.com/bncuthbert/whole-report-wm',
      author='Ben Cuthbert',
      author_email='0bec@queensu.ca',
      packages=['utils', 'vis'],
      package_dir={'':'src'},
      zip_safe=False)