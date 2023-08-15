#!/bin/bash

source ~/.bashrc

# Setup conda environment.
CONDA_ENV_NAME=einet-ccle
conda env list | grep "${CONDA_ENV_NAME}" &> /dev/null

# If environment doesn't exit, create it. Then activate it.
if [ $? -eq 1 ]; then
    echo "Creating Conda environment 'einet-ccle'..."
    conda env create --file environment.yml
fi
conda activate ${CONDA_ENV_NAME}

# Download datasets for training to the DFS.
cd src
python datasets.py
