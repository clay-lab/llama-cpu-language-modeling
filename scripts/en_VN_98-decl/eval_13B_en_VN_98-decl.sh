#!/bin/bash

#SBATCH --job-name=llama-13B-en_VN_98-decl
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --partition=day
#SBATCH --mem=128
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate /gpfs/gibbs/project/frank/ref4/conda_envs/llama

echo "Starting job at `date`"
echo

time python run_LM.py \
	--ckpt_dir llama-checkpoints/13B \
	--tokenizer_path llama-checkpoints/tokenizer.model \
	--test_file data/en_VN_98-decl/en_VN_98-decl.txt.gz \
	--max_batch_size 2

echo
echo "Finished job at `date`"
