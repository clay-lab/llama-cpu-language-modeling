#!/bin/bash

#SBATCH --job-name=llama-65B-en_PGB_99
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --partition=bigmem
#SBATCH --mem=288G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate /gpfs/gibbs/project/frank/ref4/conda_envs/llama

echo "Starting job at `date`"
echo

time python run_LM.py \
	--ckpt_dir llama-checkpoints/65B \
	--tokenizer_path llama-checkpoints/tokenizer.model \
	--dataset_path data/en_PGB_99/en_PGB_99.txt.gz \
	--max_batch_size 2

echo
echo "Finished job at `date`"
