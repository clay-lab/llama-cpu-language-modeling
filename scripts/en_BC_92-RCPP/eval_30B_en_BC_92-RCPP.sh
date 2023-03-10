#!/bin/bash

#SBATCH --job-name=llama-30B-en_BC_92-RCPP
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --partition=bigmem
#SBATCH --mem=320
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate /gpfs/gibbs/project/frank/ref4/conda_envs/llama

echo "Starting job at `date`"
echo

time python run_LM.py \
	--ckpt_dir llama-checkpoints/30B \
	--tokenizer_path llama-checkpoints/tokenizer.model \
	--test_file data/en_BC_92-RCPP/en_BC_92-RCPP.txt.gz \
	--max_batch_size 2

echo
echo "Finished job at `date`"
