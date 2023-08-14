#!/bin/sh

# Bash script for carrying out training on cluster. Is adapted
# from the scripts provided by the Univeristy of Edinburgh for 
# the use of the EDDIE cluster.


source ~/.bashrc

set -e 

# Set gpu device
export CUDA_VISIBLE_DEVICES=$1

# Setup conda environment.
echo "Setting up bash enviroment"
CONDA_ENV_NAME=einet_ccle
echo "Activating conda environment: ${CONDA_ENV_NAME}"
conda activate ${CONDA_ENV_NAME}

# Set up scratch disk directory and move input data there.
echo "Moving input data to the compute node's scratch space: $SCRATCH_DISK"
SCRATCH_DISK=/disk/scratch/${USER}
SCRATCH_HOME=${SCRATCH_DISK}
mkdir -p ${SCRATCH_HOME}
src_path=~/ccle_einets/data/datasets
dest_path=${SCRATCH_HOME}/ccle_einets/data/datasets
mkdir -p ${dest_path}
rsync --archive --update --compress --progress ${src_path}/ ${dest_path}

# Move WandB log directory to scratch disk.
export WANDB_DIR=${SCRATCH_HOME}

# make ouput directory on the scratch disk of the node if required.
mkdir -p ${SCRATCH_HOME}/ccle_einets/data/output/cclle_training

# Change command line arguments based on the training we want to carry out.
DATASET=${3:-mnist}
patch_prob=${4:-1.0}
WINDOW_DIMS=${5:-16}
# NUM_BIN_BISECTIONS=${6:-1}
# GRID_PROB=${6:-0.889}
WANDB_PROJ=EiNets_CCLE_vs_MLE

# Run EM, MLE or MCCLE training.
if [ "$2" == "em" ]; then  
    echo "Running EM training on dataset: ${DATASET}"
    eval python ~/ccle_einets/src/training.py -i /disk/scratch/${USER}/ccle_einets/data/datasets -o /disk/scratch/${USER}/ccle_einets/data/output -K 32 --max_num_epochs 64 --batch_size 100 --wandb_online --wandb_project ${WANDB_PROJ} --dataset ${DATASET} --patience 8 --pd_deltas 7,28 --use_em
    src_path=${SCRATCH_HOME}/ccle_einets/data/output/baseline_training
    dest_path=~/ccle_einets/data/output/baseline_training
    mkdir -p ${dest_path}
elif [ "$2" == "mle" ]; then
    echo "Running MLE training on dataset: ${DATASET}"
    eval python ~/ccle_einets/src/training.py -i /disk/scratch/${USER}/ccle_einets/data/datasets -o /disk/scratch/${USER}/ccle_einets/data/output -K 32 --max_num_epochs 64 --batch_size 100 --wandb_online --wandb_project ${WANDB_PROJ} --lr 0.01 --dataset ${DATASET} --patience 8 --pd_deltas 7,28
    src_path=${SCRATCH_HOME}/ccle_einets/data/output/baseline_training
    dest_path=~/ccle_einets/data/output/baseline_training
    mkdir -p ${dest_path}
elif [ "$2" == "ccle" ]; then
    echo "Running MCCLE training on dataset: ${DATASET}"
    eval python ~/ccle_einets/src/training.py -i /disk/scratch/${USER}/ccle_einets/data/datasets -o /disk/scratch/${USER}/ccle_einets/data/output -K 32 --max_num_epochs 64 --batch_size 100 --wandb_online --wandb_project ${WANDB_PROJ} --mccle --lr 0.01 --dataset ${DATASET} --patience 8 --pd_deltas 7,28 --patch_size ${WINDOW_DIMS} --patch_prob ${patch_prob}
    src_path=${SCRATCH_HOME}/ccle_einets/data/output/cclle_training
    dest_path=~/ccle_einets/data/output/cclle_training
    mkdir -p ${dest_path}
else
    echo "Invalid argument: $2"
fi
echo "Command ran successfully!"

# Move the output of experiments back to home directory in the DFS.
echo "Moving output data back to home storage from scratch scaling."
mkdir -p ${dest_path}
rsync --archive --update --compress --progress --exclude models/ ${src_path}/ ${dest_path}

echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"


