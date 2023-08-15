#!/bin/sh

source ~/.bashrc

set -e 

# Set gpu device
export CUDA_VISIBLE_DEVICES=$1

# Setup conda environment.
echo "Setting up bash enviroment"
CONDA_ENV_NAME=pcs
echo "Activating conda environment: ${CONDA_ENV_NAME}"
conda activate ${CONDA_ENV_NAME}

# Set up scratch disk directory and move input data there.
echo "Moving input data to the compute node's scratch space: $SCRATCH_DISK"
SCRATCH_DISK=/disk/scratch/${USER}
SCRATCH_HOME=${SCRATCH_DISK}
mkdir -p ${SCRATCH_HOME}
src_path=~/EiNets/data/datasets
dest_path=${SCRATCH_HOME}/EiNets/data/datasets
mkdir -p ${dest_path}
rsync --archive --update --compress --progress ${src_path}/ ${dest_path}

# Move WandB log directory to scratch disk.
export WANDB_DIR=${SCRATCH_HOME}

# make ouput directory on the scratch disk of the node if required.
mkdir -p ${SCRATCH_HOME}/EiNets/data/output/cll_training

# Run evaluation script.
echo "Running evaluation script."
eval python curriculum_training.py -i /disk/scratch/${USER}/EiNets/data/datasets -K 32 -o /disk/scratch/${USER}/EiNets/data/output --pd_deltas 7,28 --dataset mnist --wandb_online --setting_epochs 32 --lr 0.01

# Move the output of experiments back to home directory in the DFS.
echo "Moving output data back to home storage from scratch scaling."
src_path=${SCRATCH_HOME}/EiNets/data/output/
dest_path=~/EiNets/data/output/
mkdir -p ${dest_path}
rsync --archive --update --compress --progress --exclude="test_clls.csv" --exclude="fid_samples/" --exclude="fid_inpaint_samples" --exclude="samples/" --exclude="inpaint_images/" --exclude models/ ${src_path}/ ${dest_path}

echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"
