import os
import re

from tqdm import tqdm
from typing import Set, Dict

# there are also MANY more T5 ablations,
# but we can't store every model
MODELS: Set[str] = set(
	f'llama-checkpoints/{i}B' for i in (7, 13, 30, 65)
)

MODEL_PARAMS: Dict[str,Dict[str,str]] = {
	'llama-checkpoints/7B': {
		'MEM': 48,
		'PARTITION': 'day',
		'TIME': '01:00:00',
	},
	'llama-checkpoints/13B': {
		'MEM': 78,
		'PARTITION': 'day',
		'TIME': '02:00:00',
	},
	'llama-checkpoints/30B': {
		'MEM': 169,
		'PARTITION': 'bigmem',
		'TIME': '04:00:00',
	},
	'llama-checkpoints/65B': {
		'MEM': 288,
		'PARTITION': 'bigmem',
		'TIME': '20:00:00',
	}
}

DATASETS: Set[str] = {
	'en_BC_92-CCPP_SL',
	'en_BC_92-RCPP',
	'en_FMP_23-NPI-exp1',
	'en_FMP_23-NPI-exp3',
	'en_FMP_23-NPI-exp5',
	'en_FVN_02',
	'en_FVN_02-simple',
	'en_PGB_99',
	'en_VN_98-decl',
	'en_WLP_09-exp1',
	'en_WLP_09-exp3',
	'en_WLP_09-exp4',
	'en_WLP_09-exp5',
}

SCRIPT_TEMPLATE: str = '\n'.join([
	'#!/bin/bash',
	'',
	'#SBATCH --job-name=llama-{MODEL_BASENAME}-{DATASET}',
	'#SBATCH --output=joblogs/%x_%j.txt',
	'#SBATCH --partition={PARTITION}',
	'#SBATCH --mem={MEM}G',
	'#SBATCH --time={TIME}',
	'#SBATCH --mail-type=END,FAIL,INVALID_DEPEND',
	'',
	'module load miniconda',
	'',
	'source activate /gpfs/gibbs/project/frank/ref4/conda_envs/llama',
	'',
	'echo "Starting job at `date`"',
	'echo',
	'',
	'time python run_LM.py \\',
	'\t--ckpt_dir {MODEL} \\',
	'\t--tokenizer_path llama-checkpoints/tokenizer.model \\',
	'\t--dataset_path data/{DATASET}/{DATASET}.txt.gz \\',
	'\t--max_batch_size 2',
	'',
	'echo',
	'echo "Finished job at `date`"',
	'',
])

def create_scripts() -> None:
	with tqdm(total=len(DATASETS) * len(MODELS)) as pbar:
		for dataset in DATASETS:
			script_dirname = dataset
			os.makedirs(script_dirname, exist_ok=True)
			for model in MODELS:
				# deal with slashes in model names
				model_basename = os.path.basename(model)
				
				script = SCRIPT_TEMPLATE.format(
							MODEL_BASENAME=model_basename,
							DATASET=dataset,
							PARTITION=MODEL_PARAMS[model]['PARTITION'],
							MEM=MODEL_PARAMS[model]['MEM'],
							TIME=MODEL_PARAMS[model]['TIME'],
							MODEL=model,
						)
				
				script_filename = os.path.join(script_dirname, f'eval_{model_basename}_{dataset}.sh')
				with open(script_filename, 'wt') as out_file:
					_ = out_file.write(script)
				
				pbar.update(1)

if __name__ == '__main__':
	create_scripts()
