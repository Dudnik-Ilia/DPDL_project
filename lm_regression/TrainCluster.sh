#!/bin/bash -l
#SBATCH --job-name=IIML_Tut
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH --mail-type=end,fail 
#SBATCH --time=00:15:00
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

source ~/.bashrc
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

module purge
module load python

# Conda
source activate seminar

# create a temporary job dir on $WORK
mkdir ${WORK}/$SLURM_JOB_ID
cd ${WORK}/$SLURM_JOB_ID

# copy input file from location where job was submitted, and run 
cp -r ${SLURM_SUBMIT_DIR}/. .
mkdir -p output/logs/
mkdir -p output/checkpoints/

srun python src/Training.py 

mkdir ${HOME}/$SLURM_JOB_ID
cp -r ./output/. ${HOME}/$SLURM_JOB_ID
