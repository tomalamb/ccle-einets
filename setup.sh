#!/bin/bash

source ~/.bashrc

# Setup conda environment.
CONDA_ENV_NAME=einet_ccle
conda env list | grep "${CONDA_ENV_NAME}" &> /dev/null

# If environment doesn't exit, create it. Then activate it.
if [ $? -eq 1 ]; then
    echo "Creating Conda environment 'einet_ccle'..."
    conda env create --file environment.yml
fi
conda activate ${CONDA_ENV_NAME}

# Install requirements and dependencies for GPU useage.
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# Download datasets for training to the DFS.
cd src
python datasets.py
