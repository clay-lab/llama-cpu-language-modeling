#!/bin/bash

#SBATCH --job-name=llama-30B-en_WLP_09-exp3
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --partition=bigmem
#SBATCH --mem=169G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate /gpfs/gibbs/project/frank/ref4/conda_envs/llama

echo "Starting job at `date`"
echo

time python run_LM.py \
	--ckpt_dir llama-checkpoints/30B \
	--tokenizer_path llama-checkpoints/tokenizer.model \
	--dataset_path data/en_WLP_09-exp3/en_WLP_09-exp3.txt.gz \
	--max_batch_size 2

echo
echo "Finished job at `date`"
