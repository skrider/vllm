import argparse
import time
import os
from typing import List, Tuple

import numpy as np
from tqdm import tqdm
import torch
from multiprocessing import Process, Pipe
from transformers import LlamaTokenizerFast
from vllm import LLM, SamplingParams

import pandas as pd

import ray

LOGPROBS_FANOUT=256

def main(args):
    oracle = ray.remote(num_gpus=1)(LLM).remote(
        model=args.oracle_model,
        tokenizer=args.tokenizer,
        max_num_seqs=args.batch_size * args.gamma,
        trust_remote_code=args.trust_remote_code,
        dtype=args.oracle_dtype,
        gpu_memory_utilization=0.9,
    )
    print("oracle actor: ", oracle)

    draft = ray.remote(num_gpus=1)(LLM).remote(
        model=args.draft_model,
        tokenizer=args.tokenizer,
        max_num_seqs=args.batch_size,
        trust_remote_code=args.trust_remote_code,
        dtype=args.draft_dtype,
        gpu_memory_utilization=0.9,
    )
    print("draft actor: ", draft)

    input_text = "The strangest city in California is San"
    tokenizer = LlamaTokenizerFast.from_pretrained(args.tokenizer)
    input_ids = tokenizer(input_text)["input_ids"]

    draft_sp = SamplingParams(
        n=1, temperature=0., max_tokens=args.gamma, logprobs=LOGPROBS_FANOUT
    )
    oracle_sp = SamplingParams(
        n=1, temperature=0., max_tokens=1, logprobs=LOGPROBS_FANOUT
    )

    # follow the algorithm from https://arxiv.org/pdf/2211.17192.pdf page 3
    def speculative_step_deterministic(prefix: List[int]) -> Tuple[List[int], bool]:
        draft_run = draft.generate.remote(
            prompt_token_ids=[prefix], sampling_params=draft_sp, use_tqdm=False
        )
        draft_run = ray.get(draft_run)
        draft_run = draft_run[0].outputs[0]

        xs = draft_run.token_ids

        oracle_prompts = [prefix]
        for i in range(len(xs)):
            oracle_prompts += [prefix + xs[: i + 1]]

        oracle_run = oracle.generate.remote(
            prompt_token_ids=oracle_prompts, sampling_params=oracle_sp, use_tqdm=False
        )
        oracle_run = ray.get(oracle_run)

        n = 0
        sampled_token = oracle_run[0].outputs[0].token_ids[0]
        for i, x in enumerate(xs):
            sampled_token = oracle_run[i].outputs[0].token_ids[0]
            if x == sampled_token:
                n += 1
            else:
                break

        done = oracle_run[n].outputs[0].finish_reason != "length"

        return xs[:n] + [sampled_token], done

    df = pd.read_parquet(args.dataset, engine="pyarrow")
    indexer = df.loc()

    accepted = 0
    total_generated = 0

    for i in range(1000):
        conversation = indexer[i].conversation_a
        prompt = conversation[0]["content"]
        input_ids = tokenizer(prompt)["input_ids"]
        prefix = list(input_ids)

        answer = conversation[1]["content"]
        max_tokens = len(tokenizer(answer)["input_ids"])

        done = False
        next_tokens = []
        while (not done) and len(prefix) < max_tokens:
            prefix += next_tokens
            next_tokens, done = speculative_step_deterministic(prefix)
            accepted += len(next_tokens) - 1
            total_generated += min(len(next_tokens), args.gamma)
        print("total accepted: ", accepted)
        print("total generated: ", total_generated)
        print("alpha: ", accepted / total_generated)

if __name__ == "__main__":
    runtime_env = {
        "env_vars": {"HUGGING_FACE_HUB_TOKEN": os.getenv("HUGGING_FACE_HUB_TOKEN")}
    }
    ray.init(runtime_env=runtime_env)

    parser = argparse.ArgumentParser(
        description="Benchmark the accuracy/alpha of a draft model attempting to predict an oracle"
        "model"
    )
    parser.add_argument("--oracle-model", type=str, default="facebook/opt-125m")
    parser.add_argument("--draft-model", type=str, default="facebook/opt-125m")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--dataset", type=str)
    parser.add_argument(
        "--gamma", type=int, default=5, help="number of tokens to sample ahead"
    )
    parser.add_argument(
        "--split", type=float, default=0.08, help="memory to give draft model"
    )
    parser.add_argument("--temperature", type=float, choices=[0.0, 1.0], default=0.0)
    parser.add_argument("--input-len", type=int, default=32)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--num-iters", type=int, default=3, help="Number of iterations to run."
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="trust remote code from huggingface",
    )
    parser.add_argument(
        "--draft-dtype",
        type=str,
        default="auto",
        choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
    )
    parser.add_argument(
        "--oracle-dtype",
        type=str,
        default="auto",
        choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
    )
    args = parser.parse_args()

    main(args)
