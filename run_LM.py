# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import *
import os
import sys
import gzip
import torch
import torch.nn.functional as F
import fire
import time
import json

import pandas as pd

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA

import logging

logger = logging.getLogger(__name__)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)

# this marks the position of interest in our datasets
# sentences will be stripped of everything including
# and after this word. (we use these datasets also for
# MLM and seq2seq tasks where the following context is
# not stripped)
MASK_TOKEN: str = '<extra_id_0>'

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    # torch.distributed.init_process_group("gloo")
    # initialize_model_parallel(world_size)
    # torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size

def load(
    ckpt_dir: str, 
    tokenizer_path: str, 
    local_rank: int, 
    world_size: int,
    max_seq_len: int = 32, #TODO: set this dynamically
    max_batch_size: int = 1,
) -> LLaMA:
    start_time = time.time()

    logger.info("Locating checkpoints")
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert (
        world_size == len(checkpoints)
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"

    logger.info(f"Found MP={len(checkpoints)} checkpoints")
    ckpt_path = checkpoints[local_rank]

    logger.info("Creating checkpoint instance...")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    logger.info("Grabbing params...")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    logger.info("Loading model arguments...")
    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params)

    logger.info("Creating tokenizer...")
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words

    logger.info("Creating transformer...")
    torch.set_default_tensor_type(torch.BFloat16Tensor)
    model = Transformer(model_args)

    logger.info("Loading checkpoint to model...")
    _start_time = time.time()
    torch.set_default_tensor_type(torch.BFloat16Tensor)
    model.load_state_dict(checkpoint, strict=False)
    logger.info(f"Loaded in {time.time() - _start_time:.2f} seconds")

    _start_time = time.time()
    logger.info("Creating LLaMA generator...")
    generator = LLaMA(model, tokenizer)
    logger.info(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator

def load_dataset(
    dataset_path: str,
) -> List[str]:
    '''
    To be replaced with a function
    that loads the data from disk
    '''
    with gzip.open(dataset_path, 'rt') as in_file:
        dataset = in_file.readlines()
    
    # remove everything including + after the position of interest,
    # since LLaMA models predict the next word
    dataset = [example.split(MASK_TOKEN, 1)[0].strip() for example in dataset]
    
    return dataset

def preprocess_dataset(
    dataset: List[str], 
    tokenizer: Tokenizer
) -> 'Dataset':
    '''
    Formats the dataset for use with a T5ForConditionalGeneration model.
    
    params:
        dataset (Dataset)           : a list of strings to use as prompts for the model
        tokenizer (Tokenizer)   	: the tokenizer to use to prepare the examples for the model.
    
    returns:
        Dataset                     : the dataset formatted for use with a T5ForConditionalGeneration model.
    ''' 
    def preprocess_function(example: str) -> torch.Tensor:
        '''Tokenizes a string input.'''
        model_inputs = tokenizer.encode(example, bos=True, eos=False)
        return model_inputs
    
    return torch.Tensor([preprocess_function(example) for example in dataset]).long()

def evaluate_language_modeling(
    generator: LLaMA, 
    dataset_path: str, 
    output_dir: str,
):
    output_file = os.path.join(output_dir, 'language_modeling_results.csv.gz')
    
    if os.path.exists(output_file):
        # return
        pass
    
    dataset = load_dataset(dataset_path)
    
    with gzip.open(dataset_path.replace('.txt.gz', '_metadata.json.gz'), 'rt', encoding='utf-8') as in_file:
        metadata = [json.loads(l) for l in in_file.readlines()]
    
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = []
    for i, (example, ex_metadata) in enumerate(zip(dataset, metadata)):
        eval_tokens = ex_metadata['eval_tokens']
        
        metrics.extend(evaluate_example(
            generator=generator,
            inputs=[example],
            input_nums=[i],
            eval_tokens=[eval_tokens],
            batch_metadata=[ex_metadata]
        ))
    
    metrics = pd.DataFrame(metrics)
    
    metrics = metrics.assign(
        n_test_examples=len(dataset)
    )
    
    metrics.to_csv(output_file, index=False, na_rep='NaN')

def evaluate_example(
    generator: LLaMA,
    inputs: torch.Tensor,
    input_nums: List[int] = None,
    eval_tokens: List[List[str]] = None,
    batch_metadata: List[Dict] = None,
): 
    '''Evaluate a single batch of inputs on the eval tokens, depending on the model type.'''
    if input_nums is None:
        input_nums = range(len(inputs))
    
    if eval_tokens is None:
        raise ValueError(f'No tokens were provided for evaluation.')
    
    if len(eval_tokens) != len(inputs):
        raise ValueError(
            f'{len(eval_tokens)} sets of eval tokens were '
            f'provided for {len(inputs)} sentences.'
        )
        
    if batch_metadata is None:
        batch_metadata = {}
    
    eval_token_ids = get_eval_token_ids(
        tokenizer=generator.tokenizer,
        eval_tokens=eval_tokens
    )
    
    return evaluate_lm_batch(
        generator=generator,
        inputs=inputs,
        input_nums=input_nums,
        eval_tokens=eval_tokens,
        eval_token_ids=eval_token_ids,
        batch_metadata=batch_metadata,
    )

def get_eval_token_ids(
    tokenizer: Tokenizer,
    eval_tokens: List[List[str]]
) -> List[List[int]]:
    '''
    Get the eval token ids depending on their position
    in the input sequence (beginning of sentence or not).
    '''
    eval_token_ids = [[tokenizer.encode(t, bos=False, eos=False) for t in tokens] for tokens in eval_tokens]
    
    # check that the eval tokens are single tokens
    check_ids(eval_tokens=eval_tokens, eval_token_ids=eval_token_ids)
    
    eval_token_ids = [[id for t in token_ids for id in t] for token_ids in eval_token_ids]
    
    return eval_token_ids

def check_ids(
	eval_tokens: List[List[str]],
    eval_token_ids: List[List[int]],
) -> None:
    # check that eval tokens make sense
    for tokens, token_ids in zip(eval_tokens, eval_token_ids):
        if (any(len(token_id) > 1 for token_id in token_ids)):
            raise ValueError(
                f'Some tokens used for evaluation are not tokenized as single words!:\n\n' +
                "\n".join(
                    [str(t) for t in zip(tokens, token_ids) if len(t[-1]) > 1]
                )
            )

def evaluate_lm_batch(
    generator: LLaMA,
    inputs: torch.Tensor,
    input_nums: List[int],
    eval_tokens: List[List[str]],
    eval_token_ids: List[List[int]],
    batch_metadata: List[Dict],
) -> List[Dict]:
    with torch.no_grad():
        batch_outputs = generator(tokens=inputs, start_pos=0)
    
    batch_scores = torch.stack([t[:generator.tokenizer.n_words] for t in batch_outputs])
    batch_logprobs = F.log_softmax(batch_scores, dim=-1)
    
    metrics = []
    records = zip(input_nums, inputs, batch_outputs, eval_tokens, eval_token_ids, batch_logprobs, batch_metadata)
    
    for input_num, input_seq, pred_token, tokens, token_ids, score, example_metadata in records:
        metrics.extend(
            [
                {
                    'item': input_num,
                    'input_text': generator.tokenizer.decode(input_seq.tolist()),
                    'pred_token': generator.tokenizer.decode(torch.argmax(pred_token, dim=-1).tolist()),
                    'token': token,
                    'token_id': token_id,
                    'logprob': score[token_id].item(),
                    **{k: v for k, v in example_metadata.items() if not k == 'eval_tokens'}
                } for token, token_id in zip(tokens, token_ids)
            ]
        )
    
    return metrics

def main(
    ckpt_dir: str, 
    tokenizer_path: str, 
    dataset_path: str,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, 'w')

    generator = load(ckpt_dir, tokenizer_path, local_rank, world_size)
    
    output_dir = os.path.join('outputs', os.path.split(ckpt_dir)[-1])
    evaluate_language_modeling(generator=generator, dataset=dataset, output_dir=output_dir)
    
if __name__ == "__main__":
    fire.Fire(main)
