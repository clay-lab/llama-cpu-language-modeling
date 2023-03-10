#!/bin/bash


#SBATCH --job-name=test-llama-lm-7B
#SBATCH --output=joblogs/test-llama-lm-7B.log
#SBATCH --mem=32G
#SBATCH --partition=day
#SBATCH --time=01-00:00:00


module load miniconda

source activate /gpfs/gibbs/project/frank/ref4/conda_envs/llama

echo `date`

python run_LM.py \
	--ckpt_dir llama-checkpoints/7B \
	--tokenizer_path llama-checkpoints/tokenizer.model \
    --dataset_path data/en_BC_92-RCPP/en_BC_92-RCPP.txt.gz \
    --max_batch_size 2

# to give us an idea of how long it takes to run
echo `date`
