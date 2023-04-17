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
    
    Result:
        After code is run, runs log_fn with a string
        "Completed in n seconds"
    '''
    def __init__(self, log_fn=print):
        self.log_fn = log_fn
    
    def __enter__(self):
        self._start_time = time.time()
    
    def __exit__(self, type, value, traceback):
        self.log_fn(
            f'Completed in {time.time() - self._start_time:.2f} seconds'
        )

def load_tokenizer(
    tokenizer_path: str,
) -> Tokenizer:
    '''
    Loads the LLaMA tokenizer.
    We do this separately so we can tokenize
    the dataset before loading the model, and thus
    automatically adjust the maximum sequence length.
    '''
    logger.info('Creating tokenizer...')
    tokenizer = Tokenizer(model_path=tokenizer_path)
    
    return tokenizer

def load_llama(
    ckpt_dir: str,
    tokenizer: Tokenizer,
    max_seq_len: int,
    max_batch_size: int = 1,
) -> LLaMA:
    '''
    Loads and returns a LLaMA generator.
    
    params:
        ckpt_dir (str): the location of the directory containing the LLaMA checkpoint
        tokenizer (Tokenizer): the tokenizer for the model
        max_seq_len (int): the maximum sequence length that can be generated.
                           must be at least the length of the longest (tokenized) input sequence
        max_batch_size (int): at most this many examples will be run in a batch
    
    returns:
        LLaMA: the LLaMA generator
    '''
    checkpoints = sorted(Path(ckpt_dir).glob('*.pth'))
    
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())
    
    model_args = ModelArgs(
        max_seq_len=max_seq_len, 
        max_batch_size=max_batch_size,
        **params
    )
    
    model_args.vocab_size = tokenizer.n_words
    
    logger.info('Creating transformer...')
    model = Transformer(model_args)
    
    # Original copyright by tloen
    # https://github.com/tloen/llama-int8/blob/main/example.py
    key_to_dim = {
        "w1": 0,
        "w2": -1,
        "w3": 0,
        "wo": -1,
        "wq": 0,
        "wk": 0,
        "wv": 0,
        "output": 0,
        "tok_embeddings": -1,
        "ffn_norm": None,
        "attention_norm": None,
        "norm": None,
        "rope": None,
    }
    
    logger.info('Loading checkpoints to model...')
    with Timer(logger.info):
        for i, ckpt in tqdm(enumerate(checkpoints), total=len(checkpoints)):
            checkpoint = torch.load(ckpt, map_location="cpu")
            for parameter_name, parameter in model.named_parameters():
                short_name = parameter_name.split(".")[-2]
                if key_to_dim[short_name] is None and i == 0:
                    parameter.data = checkpoint[parameter_name]
                elif key_to_dim[short_name] == 0:
                    size = checkpoint[parameter_name].size(0)
                    parameter.data[size * i: size * (i + 1), :] = checkpoint[
                        parameter_name
                    ]
                elif key_to_dim[short_name] == -1:
                    size = checkpoint[parameter_name].size(-1)
                    parameter.data[:, size * i: size * (i + 1)] = checkpoint[
                        parameter_name
                    ]
                del checkpoint[parameter_name]
            del checkpoint
        
        model.to("cpu")
    
    generator = LLaMA(model, tokenizer)
    
    setattr(generator, 'model_name_or_path', ckpt_dir)
    
    return generator

def load_dataset(
    dataset_path: str,
) -> List[str]:
    '''
    Loads a dataset from disk. This should be a txt.gz file with one
    example to run per line. Positions of interest should be marked
    with MASK_TOKEN (defined above). Because of how generation for
    decoder-only models works, only one position of interest
    can be defined per sentence.
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
) -> torch.Tensor:
    '''
    Formats the dataset for use with a LLaMA model.
    
    params:
        dataset (Dataset)           : a list of strings to use as prompts for the model
        tokenizer (Tokenizer)   	: the tokenizer to use to prepare the examples for the model.
    
    returns:
        Dataset                     : the dataset formatted for use with a LLaMA model.
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
    
    # we increase the maximum sequence length by 1 so we can generate the next token
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
    '''
    Evaluates a LLaMA model on the language modeling task defined
    by the dataset.
    
    params:
        generator (LLaMA): the LLaMA generator to evaluate
        dataset (torch.Tensor): a dataset formatted for use with a LLaMA model.
                                this should be a tensor of shape (num_examples, max_seq_len)
        dataset_path (str): the path to the dataset file. used to find the metadata for the dataset
                            the metadata contains information about which tokens to evaluate for
                            each example
        output_dir (str): where to save the results of the evaluation
        max_batch_size (int): the maximum batch size
    '''
    output_file = os.path.join(output_dir, f'{os.path.split(generator.model_name_or_path)[-1]}.lm_results.csv.gz')
       
    if os.path.exists(output_file):
        # return commented out for testing
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
        
        metrics.extend(evaluate_batch(
            generator=generator,
            inputs=inputs,
            input_nums=input_nums,
            batch_metadata=batch_metadata
        ))
    
    metrics = pd.DataFrame(metrics)
    
    metrics = metrics.assign(
        model_name=re.sub('["\']', '', generator.model_name_or_path),
        architecture='decoder-only',
        n_params=f'{round(sum(p.numel() for p in generator.model.parameters() if p.requires_grad)/1000000000)}B',
        test_dataset=os.path.basename(dataset_path).replace('.txt.gz', ''),
        n_test_examples=len(dataset)
    )
    
    metrics.to_csv(output_file, index=False, na_rep='NaN')

def evaluate_batch(
    generator: LLaMA,
    inputs: torch.Tensor,
    input_nums: List[int] = None,
    batch_metadata: List[Dict] = None,
) -> List[Dict]: 
    '''
    Evaluate a single batch of inputs on the eval tokens.
    
    params:
        generator (LLaMA): the LLaMA generator being evaluated
        inputs (torch.Tensor): the batch of inputs to evaluate. shape: (batch_size, max_seq_len)
        input_nums (List[int]): a list of unique numerical identifiers for the inputs
                                if not provided, range(len(inputs)) is used
        batch_metadata (List[Dict]): for each example, the metadata for that example
                                     must minimally contain a List of eval tokens for each example,
                                     under the key `eval_tokens`

    returns:
        List[Dict]: a List of dictionaries, each of which contains the results
                    of the evaluation for the corresponding example
    '''
    if input_nums is None:
        input_nums = range(len(inputs))
    
    eval_tokens = [d.get("eval_tokens") for d in batch_metadata]    
    
    if not any(eval_tokens):
        raise ValueError(f'No tokens were specified for evaluation.')
    
    if len(eval_tokens) != len(inputs):
        raise ValueError(
            f'{len(eval_tokens)} sets of eval tokens were '
            f'provided for {len(inputs)} sentences.'
        )
    
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
    Get the token ids for the eval tokens.
    
    params:
        tokenizer (Tokenizer): the tokenizer to use to encode the eval tokens
        eval_tokens (List[List[str]]): for each example in a batch, a list of eval_tokens
                                       to use to evaluate the corresponding example
    
    returns:
        List[List[int]]: the token ids of the eval tokens

    raises:
        ValueError via check_ids: if any tokens are not in the model vocabulary
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
    '''
    Verifies that all tokens in eval_token_ids are length 1.
    This ensure that the tokens exist in the tokenizer's vocabulary.
    
    params:
        eval_tokens (List[List[str]]): the list of eval tokens as strings. Used
                                       to make the ValueError message more useful
        eval_token_ids (List[List[int]]): the encodings of the tokens by the tokenizer to check
    
    raises:
        ValueError: if any token_ids are longer than 1 token, raises a ValueError
                    because that token is not a word in the tokenizer's vocabulary
    '''
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
    '''
    Evaluates a batch of examples on a language modeling (=next word prediction) task.
    
    params:
        generator (LLaMA): the generator to run on the inputs
        inputs (torch.Tensor): the tokenized batch to run next word predictions for
        input_nums (List[int]): a list of numerical input identifiers
        eval_tokens (List[List[str]]): for each example in the inputs, the tokens to extract
                                       log probabilities for
        eval_token_ids (List[List[int]]): for each example in the inputs: the encoded tokens
                                          to extract log probabilities for
        batch_metadata (List[Dict]): for each input, a dictionary containing metadata to add
                                     to the results
    
    returns:
        List[Dict]: for each example_i * eval_tokens_example_i, a dictionary containing
                    the next word predicted by the generator for that example,
                    and the log probability predicted for that eval token
    '''
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
                    'pred_seq': generator.tokenizer.decode(torch.argmax(pred_token, dim=-1).tolist()), # we use "pred_seq" for consistency with other models
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
    '''
    runs a language modeling task using a LLaMA generator
    
    params:
        ckpt_dir (str): the location of the LLaMA checkpoint (e.g., llama-checkpoints/7B)
        tokenizer_path (str): the location of the LLaMA tokenizer (e.g., llama-checkpoints/tokenizer.model)
        dataset_path (str): the location of the dataset to use for evaluation
                            datasets should consist of two files. one is a txt.gz file
                                that contains a single example per line, with the position
                                of interest marked with `<extra_id_0>`
                            the second is a file with the same name + `_metadata`, a .json.gz file
                                which for each line of the txt.gz file contains the metadata
                                for the corresponding example (minimally, a dictionary with key `eval_tokens`
                                containing a list of strings to extract log probabilities from the next word
                                predictions for the corresponding sentence
        max_batch_size (int): how many examples to run at the same time
                              since LLaMA models are large, unless you have a lot of resources,
                              best to keep this small
    
    Results are saved to disk in outputs/$dataset_name/$checkpoint_size.lm_results.csv.gz
    
    '''
    # we load the tokenizer and dataset first
    # to automatically determine the maximum sequence length
    dataset = load_dataset(dataset_path=dataset_path)
    tokenizer = load_tokenizer(tokenizer_path=tokenizer_path)
    
    dataset = preprocess_dataset(dataset=dataset, tokenizer=tokenizer)
    
    max_seq_len = dataset.size()[-1]
    
    generator = load_llama(
        ckpt_dir=ckpt_dir, 
        tokenizer=tokenizer, 
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    
    output_dir = os.path.join('outputs', os.path.split(dataset_path)[-1].replace('.txt.gz', ''))
    evaluate_language_modeling(
        generator=generator, 
        dataset=dataset, 
        dataset_path=dataset_path,
        output_dir=output_dir,
        max_batch_size=max_batch_size,
    )
    
if __name__ == '__main__':
    fire.Fire(main)
