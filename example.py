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

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    # torch.distributed.init_process_group("gloo")
    # initialize_model_parallel(world_size)
    # torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(ckpt_dir: str, tokenizer_path: str, local_rank: int, world_size: int) -> LLaMA:
    start_time = time.time()

    print("Locating checkpoints")
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert (
        world_size == len(checkpoints)
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"

    print(f"Found MP={len(checkpoints)} checkpoints")
    ckpt_path = checkpoints[local_rank]

    print("Creating checkpoint instance...")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    print("Grabbing params...")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    print("Loading model arguments...")
    model_args: ModelArgs = ModelArgs(
        max_seq_len=20, max_batch_size=2, **params)

    print("Creating tokenizer...")
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words

    print("Creating transformer...")
    torch.set_default_tensor_type(torch.BFloat16Tensor)
    model = Transformer(model_args)

    print("Loading checkpoint to model...", end="")
    _start_time = time.time()
    torch.set_default_tensor_type(torch.BFloat16Tensor)
    model.load_state_dict(checkpoint, strict=False)
    print(f"done in {time.time() - _start_time:.2f} seconds")

    _start_time = time.time()
    print("Creating LLaMA generator...", end="")
    generator = LLaMA(model, tokenizer)
    print(f"done in {time.time() - _start_time:.2f} seconds")

    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(ckpt_dir: str, tokenizer_path: str, temperature: float = 0.8, top_p: float = 0.95):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, 'w')

    generator = load(ckpt_dir, tokenizer_path, local_rank, world_size)
    prompts = ['The key to the cabinets near the table', 'The key to the cabinets']
    start_time = time.time()
    results = generator.generate(
        prompts, max_gen_len=1, temperature=temperature, top_p=top_p)
    print(f"responded in {time.time() - start_time:.2f} seconds")

    for result in results:
        print(result)
        print("\n==================================\n")
        

if __name__ == "__main__":
    fire.Fire(main)
