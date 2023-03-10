# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List

import torch

from llama.tokenizer import Tokenizer
from llama.model import Transformer

import logging

logger = logging.getLogger(__name__)

class LLaMA:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    @torch.inference_mode()
    def __call__(self, tokens: torch.Tensor):
        '''
        Returns logits for the next token for each example
        in a batch of tokens.
        '''
        each_prompt_size = sorted([
            min((t == self.tokenizer.pad_id).nonzero(as_tuple=True)[0]).item() for t in tokens
        ])
        
        min_prompt_size = each_prompt_size[0]
        max_prompt_size = each_prompt_size[-1]
        
        total_len = max_prompt_size + 1
        
        input_mask = tokens != self.tokenizer.pad_id
        
        start_pos = min_prompt_size
        prev_pos = 0        
        
        keep_logits = torch.tensor(())
        
        for cur_pos in range(start_pos, total_len):            
            _logits = self.model.forward(tokens=tokens[:, prev_pos:cur_pos], start_pos=prev_pos)
            
            # this gets run the first time through the loop
            # to initialize the tensor we'll use to store
            # the logits for the next token for each example
            if keep_logits.shape[0] == 0:
                keep_logits = _logits
            
            # if the prompts are of unequal length,
            # we want to get the logits for the next position for each prompt
            if not torch.all(input_mask):
                # get the inputs for which we want the logits.
                # these are the ones where the input_mask is False,
                # because that corresponds to the previous token
                # being the end of the input
                keep = torch.where(
                    torch.all(
                        torch.stack(
                            (
                                input_mask[:, cur_pos-1],
                                input_mask[:, cur_pos] == False
                            ),
                            dim=-1
                        ), 
                        dim=-1
                    )
                )
                
                keep_logits[keep] = _logits[keep]
                
                next_token = torch.full((_logits.shape[0],), self.tokenizer.eos_id)
                    
                # only replace token if prompt has already been generated
                next_token = torch.where(
                    input_mask[:, cur_pos], tokens[:, cur_pos], next_token
                )
                tokens[:, cur_pos] = next_token
            
            prev_pos = cur_pos
        
        return keep_logits
    
    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        
        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])
        
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)
        
        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cpu().long()
        
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        
        start_pos = min_prompt_size
        prev_pos = 0
        
        for cur_pos in range(start_pos, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos
        
        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
