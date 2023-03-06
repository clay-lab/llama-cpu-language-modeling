#!/bin/bash


#SBATCH --job-name=test-llama-lm
#SBATCH --output=joblogs/test-llama-lm.log
#SBATCH --mem=32G
#SBATCH --partition=day
#SBATCH --time=02:00:00


module load miniconda

source activate /gpfs/gibbs/project/frank/ref4/conda_envs/llama

torchrun \
	--nproc_per_node 1 \
	run_LM.py \
	--ckpt_dir llama-checkpoints/7B \
	--tokenizer_path llama-checkpoints/tokenizer.model
