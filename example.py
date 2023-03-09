# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json

from pathlib import Path

from llama import ModelArgs, Transformer, Tokenizer, LLaMA

def load(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int = 17,
    max_batch_size: int = 1,
) -> LLaMA:
    print("Creating model...")
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())
    
    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        **params,
    )
    
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    
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
     
    for i, ckpt in enumerate(checkpoints):
        print(f"Loading checkpoint {i}")
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
    print(f"Loaded model in {time.time() - start_time:.2f} seconds")
    return generator    


def main(
    ckpt_dir: str, 
    tokenizer_path: str, 
    temperature: float = 0.8, 
    top_p: float = 0.95,
    max_seq_len: int = 20,
    max_batch_size: int = 2,
):
    generator = load(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size)
    
    prompts = ['The thing I like best is', 'The key to the cabinets']
    
    results = generator.generate(
        prompts, max_gen_len=5, temperature=temperature, top_p=top_p
    )
    
    for result in results:
        print(result)
        print("\n==================================\n")
        

if __name__ == "__main__":
    fire.Fire(main)
