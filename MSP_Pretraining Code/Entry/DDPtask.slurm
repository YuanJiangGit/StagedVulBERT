#!/bin/bash

#SBATCH --job-name=Pretrain_SCB
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:5
#SBATCH --output=%j.out
#SBATCH --error=%j.err

module load miniconda3

source activate GraphCodeBert

python -m torch.distributed.launch --nproc_per_node=5 --nnodes=1 PreTrain_nl_DDP.py