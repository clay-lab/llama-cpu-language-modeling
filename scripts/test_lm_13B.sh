#!/bin/bash


#SBATCH --job-name=test-llama-lm-13B
#SBATCH --output=joblogs/test-llama-lm-13B.log
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=64G
#SBATCH --partition=day
#SBATCH --time=01-00:00:00


module load miniconda

source activate /gpfs/gibbs/project/frank/ref4/conda_envs/llama

echo `date`

torchrun \
	--nproc_per_node 2 \
	run_LM.py \
	--ckpt_dir llama-checkpoints/13B \
	--tokenizer_path llama-checkpoints/tokenizer.model \
    --dataset_path data/en_BC_92-RCPP/en_BC_92-RCPP.txt.gz \
    --max_batch_size 2

# to give us an idea of how long it takes to run
echo `date`
