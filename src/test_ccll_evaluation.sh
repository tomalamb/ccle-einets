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
CONDA_ENV_NAME=einet-ccle
echo "Activating conda environment: ${CONDA_ENV_NAME}"
conda activate ${CONDA_ENV_NAME}

# Set up scratch disk directory and move input data there.
echo "Moving input data to the compute node's scratch space: $SCRATCH_DISK"
SCRATCH_DISK=/disk/scratch/${USER}
SCRATCH_HOME=${SCRATCH_DISK}
mkdir -p ${SCRATCH_HOME}
src_path=~/ccle-einets/data/datasets
dest_path=${SCRATCH_HOME}/ccle-einets/data/datasets
mkdir -p ${dest_path}
rsync --archive --update --compress --progress ${src_path}/ ${dest_path}

# Move WandB log directory to scratch disk.
export WANDB_DIR=${SCRATCH_HOME}

# Change the default torch cache directory for downloading inception model.
export TORCH_HOME="${SCRATCH_HOME}/.cache/torch"

# Change command line arguments based on the evaluation we want to carry out.
DATASET=${2:-"mnist"}
PATCH_DIMS=${3:-16}
CCLL_TEST=${4:-"y"}
CCLE=${5:-"y"}
INPAINTING=${6:-"n"}
# GRID_PROB=${7:-0.15}
# NUM_BIN_BISECTIONS=${7:-1}
# NUM_CONDITIONALS=${7:-1}
 
# Run evaluation script.
if [ ${CCLL_TEST} == "y" ]; then 
    # Run ccll test evaluation script.
    if [ ${CCLE} == "y" ]; then 
        echo "Running test ccll for cclle model"
        eval python test_ccll_evaluation.py --data_i /disk/scratch/${USER}/ccle-einets/data/datasets --model_i /disk/scratch/${USER}/ccle-einets/data/output -K 32 -o /disk/scratch/${USER}/ccle-einets/data/output --pd_deltas 7,28 --dataset ${DATASET} --patch_size ${PATCH_DIMS} --ccll_test
        src_path=${SCRATCH_HOME}/ccle-einets/data/output/mccle_evaluation
        dest_path=~/ccle-einets/data/output/mccle_evaluation
        mkdir -p ${dest_path}
    elif [ ${CCLE} == "n" ]; then
        # Note that need to change depending on whether you want to evaluate EM or SGD baseline.
        echo "Running test ccll for MLE model"
        eval python test_ccll_evaluation.py --data_i /disk/scratch/${USER}/ccle-einets/data/datasets --model_i /disk/scratch/${USER}/ccle-einets/data/output -K 32 -o /disk/scratch/${USER}/ccle-einets/data/output --pd_deltas 7,28 --dataset ${DATASET} --em --ccll_test
        src_path=${SCRATCH_HOME}/ccle-einets/data/output/baseline_evaluation
        dest_path=~/ccle-einets/data/output/baseline_evaluation
        mkdir -p ${dest_path}
    else
        echo "Invalid argument."
    fi
elif [ ${CCLL_TEST} == "n" ]; then 
    # Run FID evaluation script.
    if [ ${CCLE} == "y" ]; then 
        if [ ${INPAINTING} == "n" ]; then
            # Run FID evaluation script.
            echo "Running fid script for ccle model"
            eval python test_ccll_evaluation.py --data_i /disk/scratch/${USER}/ccle-einets/data/datasets --model_i /disk/scratch/${USER}/ccle-einets/data/output -K 32 -o /disk/scratch/${USER}/ccle-einets/data/output --pd_deltas 7,28 --dataset ${DATASET} --patch_size ${PATCH_DIMS} --fid
            src_path=${SCRATCH_HOME}/ccle-einets/data/output/mccle_evaluation
            dest_path=~/ccle-einets/data/output/mccle_evaluation/
            mkdir -p ${dest_path}
        elif [ ${INPAINTING} == "y" ]; then
            # Run FID inpainting evaluation script.
            echo "Running fid inpainting script for ccle mode"
            eval python test_ccll_evaluation.py --data_i /disk/scratch/${USER}/ccle-einets/data/datasets --model_i /disk/scratch/${USER}/ccle-einets/data/output -K 32 -o /disk/scratch/${USER}/ccle-einets/data/output --pd_deltas 7,28 --dataset ${DATASET} --patch_size ${PATCH_DIMS} --fid_inpaint
            src_path=${SCRATCH_HOME}/ccle-einets/data/output/mccle_evaluation
            dest_path=~/ccle-einets/data/output/mccle_evaluation
            mkdir -p ${dest_path}
        else
            echo "Invalid argument: $6"
        fi
    elif [ ${CCLE} == "n" ]; then 
        # Run FID evaluation script for baselien models
        if [ ${INPAINTING} == "n" ]; then
            echo "Running fid script for MLE model"
            # Again note that need to change depending on whether you want to evaluate EM or SGD baseline.
            eval python test_ccll_evaluation.py --data_i /disk/scratch/${USER}/ccle-einets/data/datasets/ --model_i /disk/scratch/${USER}/ccle-einets/data/output -K 32 -o /disk/scratch/${USER}/ccle-einets/data/output --pd_deltas 7,28 --dataset ${DATASET} --em --fid
            src_path=${SCRATCH_HOME}/ccle-einets/data/output
            dest_path=~/ccle-einets/data/output
            mkdir -p ${dest_path}
        elif [ ${INPAINTING} == "y" ]; then
            # Run FID inpainting evaluation script.
            echo "Running fid inpainting script for ccle mode"
            eval python test_ccll_evaluation.py --data_i /disk/scratch/${USER}/ccle-einets/data/datasets/ --model_i /disk/scratch/${USER}/ccle-einets/data/output -K 32 -o /disk/scratch/${USER}/ccle-einets/data/output --pd_deltas 7,28 --dataset ${DATASET} --em --fid_inpaint
            src_path=${SCRATCH_HOME}/ccle-einets/data/output/baseline_evaluation
            dest_path=~/ccle-einets/data/output/baseline_evaluation
            mkdir -p ${dest_path}
        else
            echo "Invalid argument: $6"
        fi
    else
        echo "Invalid argument: $6"
    fi
else
    echo "Invalid argument: $5"
fi


# Move the output of experiments back to home directory in the DFS.
echo "Moving output data back to home storage from scratch scaling."
mkdir -p ${dest_path}
rsync --archive --update --compress --progress --exclude models/ ${src_path}/ ${dest_path}

echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"
