#!/bin/bash


#SBATCH --job-name=test-llama-13B
#SBATCH --output=joblogs/test-llama-13B.log
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=128G
#SBATCH --partition=day
#SBATCH --time=01-00:00:00


module load miniconda

source activate /gpfs/gibbs/project/frank/ref4/conda_envs/llama

echo `date`

python \
	example.py \
	--ckpt_dir llama-checkpoints/13B \
	--tokenizer_path llama-checkpoints/tokenizer.model \
    --max_batch_size 2

# to give us an idea of how long it takes to run
echo `date`
