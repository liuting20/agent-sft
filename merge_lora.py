from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into the base model.")
    parser.add_argument("--adapter_path", default="outputs/qwen3-8b-thinking-whoandwhen")
    parser.add_argument("--output_dir", default="outputs/qwen3-8b-thinking-whoandwhen-merged")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.adapter_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True)

    merged_model = model.merge_and_unload()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Merged model saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
