'''
Oxono setup file
M1 Bio-info project
Daniel Calvez & Vincent Ducot
2025
'''

import zipfile
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Define dependencies
REQUIRED = [
    'colorama==0.4.6',
    'icecream==2.1.4',
    'matplotlib==3.10.0',
    'numpy==2.2.2',
    'pathlib==1.0.1',
    'pip==25.0.1',
    'pytest==8.3.4',
    'torch==2.6.0'
]


def extract_zip(filepath: str):
    ''' Unzip a file'''
    setup_dir = Path(__file__).parent.absolute()
    try:
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(setup_dir)
        print(f"Successfully extracted {filepath} to {setup_dir}")
    except IOError as e:
        print(f"Error extracting zip file: {e}")
        sys.exit(1)

# Unzip DQN model file
extract_zip('xonox.dqn.zip')

# Setup configuration
setup(
    name='OXONO',
    version='1.0',
    description='Oxono game, M1 BI project',
    author='Daniel Calvez & Vincent Ducot',
    url='https://github.com/Daniel-Calvez/Projet-Oxono',
    packages=find_packages(),
    install_requires=REQUIRED,
    include_package_data=True,
    python_requires=">=3.12",
)
