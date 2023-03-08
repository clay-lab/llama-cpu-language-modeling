# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import *
import os
import re
import sys
import gzip
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import fire
import time
import json

import pandas as pd

from tqdm import tqdm

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

class Timer():
    '''
    Prints a message about how long a block of code takes to run.
    
    Usage:
        with Timer(log_fn=...):
            ...
    '''
    def __init__(self, log_fn=print):
        self.log_fn = log_fn
    
    def __enter__(self):
        self._start_time = time.time()
    
    def __exit__(self, type, value, traceback):
        self.log_fn(
            f'Completed in {time.time() - self._start_time:.2f} seconds'
        )

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    # torch.distributed.init_process_group("gloo")
    # initialize_model_parallel(world_size)
    # torch.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size

def load_tokenizer(
    tokenizer_path: str,
) -> Tokenizer:
    '''
    Loads the LLaMA tokenizer.
    We do this separately so we can tokenize
    the dataset before loading the model, and thus
    dynamically adjust the necessary cache as required.
    '''
    logger.info('Creating tokenizer...')
    tokenizer = Tokenizer(model_path=tokenizer_path)
        
    return tokenizer

def load_llama(
    ckpt_dir: str,
    tokenizer: Tokenizer,
    local_rank: int, 
    world_size: int,
    max_seq_len: int,
    max_batch_size: int = 1,
) -> LLaMA:
    '''
    Loads and returns a LLaMA generator.
    
    params:
        ckpt_dir (str): the location of the directory containing the LLaMA checkpoint
        tokenizer (Tokenizer): the tokenizer for the model 
        dataset (List[str]): the dataset to be used for evaluation
                             this is used to dynamically set the max_seq_len, which must
                             be at least the length of the longest (tokenized) input seq
                             in words. we do this here to avoid setting an overly large
                             cache size when creating the ModelArgs
        local_rank (int): the local rank of the process
        world_size (int): the world size (i.e., how many processes total)
        max_seq_len (int): the maximum sequence length that can be generated.
                           must be at least the length of the longest (tokenized) input sequence
        max_batch_size (int): at most this many examples will be run in a batch
    
    returns:
        LLaMA: the LLaMA generator
    '''
    logger.info('Locating checkpoints')
    checkpoints = sorted(Path(ckpt_dir).glob('*.pth'))
    assert (
        world_size == len(checkpoints)
    ), f'Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}'
    
    logger.info(f'Found MP={len(checkpoints)} checkpoints')
    ckpt_path = checkpoints[local_rank]
    
    logger.info('Creating checkpoint instance...')
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    logger.info('Grabbing params...')
    with open(Path(ckpt_dir)/'params.json', 'r') as f:
        params = json.loads(f.read())
    
    logger.info('Loading model arguments...')
    model_args = ModelArgs(
        max_seq_len=max_seq_len, 
        max_batch_size=max_batch_size,
        **params
    )
    
    model_args.vocab_size = tokenizer.n_words
    
    logger.info('Creating transformer...')
    torch.set_default_tensor_type(torch.HalfTensor)
    model = Transformer(model_args)
    
    logger.info('Loading checkpoint to model...')
    with Timer(logger.info):
        torch.set_default_tensor_type(torch.FloatTensor)
        model.load_state_dict(checkpoint, strict=False)
    
    logger.info('Creating LLaMA generator...')
    with Timer(logger.info):
        generator = LLaMA(model, tokenizer)
    
    setattr(generator, 'model_name_or_path', ckpt_dir)
    
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
    
    # return dataset
    # for debugging
    return ['The key to the cabinets on the table', 'The key to the cabinets']

def preprocess_dataset(
    dataset: List[str], 
    tokenizer: Tokenizer
) -> torch.Tensor:
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
    
    def pad_tensor(t: torch.Tensor, pad: int, dim: int = -1) -> torch.Tensor:
        '''
        Pads a tensor to length pad in dim dim.
        From https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/8
        
            params:
                t (torch.Tensor): tensor to pad
                pad (int)       : the size to pad to
                dim (int)       : dimension to pad
            
            returns:
                a new torch.Tensor padded to 'pad' in dimension 'dim'
        '''
        pad_size = list(t.shape)
        pad_size[dim] = pad - t.size(dim)
        return torch.cat([
            t, 
            torch.full(
                size=pad_size, 
                fill_value=tokenizer.pad_id, 
                dtype=t.dtype, 
                device=t.device
            )
        ], dim=dim)
    
    dataset = [torch.tensor(preprocess_function(example)) for example in dataset]
    
    max_seq_len = max(len(example) for example in dataset) + 1
    logger.info(f'Maximum sequence length in dataset is {max_seq_len - 1} tokens')
    
    dataset = torch.stack([pad_tensor(t=t, pad=max_seq_len, dim=-1) for t in dataset])
    
    return dataset

def evaluate_language_modeling(
    generator: LLaMA, 
    dataset: torch.Tensor,
    dataset_path: str,
    output_dir: str,
    max_batch_size: int = 1,
) -> None:
    output_file = os.path.join(output_dir, f'{os.path.split(generator.model_name_or_path)[-1]}.lm_results.csv.gz')
       
    if os.path.exists(output_file):
        # return
        pass
    
    with gzip.open(dataset_path.replace('.txt.gz', '_metadata.json.gz'), 'rt', encoding='utf-8') as in_file:
        metadata = [json.loads(l) for l in in_file.readlines()]
    
    os.makedirs(output_dir, exist_ok=True)
    
    dataloader = DataLoader(dataset, batch_size=max_batch_size)
    
    n_observed_examples = 0
    metrics = []
    for inputs in tqdm(dataloader):
        n_examples_in_batch = inputs.shape[0]
        
        # use this as a unique input identifier
        input_nums = range(n_observed_examples, n_observed_examples + n_examples_in_batch)
        n_observed_examples += n_examples_in_batch
        
        batch_metadata = metadata[(n_observed_examples - n_examples_in_batch):n_observed_examples]
        eval_tokens = [d['eval_tokens'] for d in batch_metadata]
        
        metrics.extend(evaluate_batch(
            generator=generator,
            inputs=inputs,
            input_nums=input_nums,
            eval_tokens=eval_tokens,
            batch_metadata=batch_metadata
        ))
    
    metrics = pd.DataFrame(metrics)
    
    metrics = metrics.assign(
        model_name=re.sub('["\']', '', generator.model_name_or_path),
        n_params=f'{round(sum(p.numel() for p in generator.model.parameters() if p.requires_grad)/1000000000)}B',
        test_dataset=os.path.basename(dataset_path).replace('.txt.gz', ''),
        n_test_examples=len(dataset)
    )
    
    metrics.to_csv(output_file, index=False, na_rep='NaN')

def evaluate_batch(
    generator: LLaMA,
    inputs: torch.Tensor,
    input_nums: List[int] = None,
    eval_tokens: List[List[str]] = None,
    batch_metadata: List[Dict] = None,
) -> List[Dict]: 
    '''Evaluate a single batch of inputs on the eval tokens.'''
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
        batch_metadata = [{} for _ in range(len(inputs))]
    
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
        batch_outputs = generator(tokens=inputs)
    
    batch_scores = torch.stack([t[:generator.tokenizer.n_words] for t in batch_outputs])
    batch_logprobs = F.log_softmax(batch_scores, dim=-1)
    
    metrics = []
    records = zip(input_nums, inputs, batch_outputs, eval_tokens, eval_token_ids, batch_logprobs, batch_metadata)
    
    for input_num, input_seq, pred_token, tokens, token_ids, score, example_metadata in records:
        metrics.extend(
            [
                {
                    'item': input_num,
                    'input_text': generator.tokenizer.decode(input_seq[torch.where((input_seq != generator.tokenizer.pad_id).nonzero(as_tuple=True)[0])].tolist()),
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
    max_batch_size: int,
) -> None:
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, 'w')
    
    dataset = load_dataset(dataset_path=dataset_path)
    tokenizer = load_tokenizer(tokenizer_path=tokenizer_path)
    breakpoint()
    dataset = preprocess_dataset(dataset=dataset, tokenizer=tokenizer)
    
    max_seq_len = dataset.size()[-1]
    
    generator = load_llama(
        ckpt_dir=ckpt_dir, 
        tokenizer=tokenizer, 
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        local_rank=local_rank, 
        world_size=world_size,
    )
    
    output_dir = os.path.join('outputs', os.path.split(dataset_path)[-1].replace('.txt.gz', ''))
    evaluate_language_modeling(
        generator=generator, 
        dataset=dataset, 
        dataset_path=dataset_path,
        max_batch_size=max_batch_size,
        output_dir=output_dir,
    )
    
if __name__ == '__main__':
    fire.Fire(main)
