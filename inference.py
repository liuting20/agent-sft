from __future__ import annotations

import argparse

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with the fine-tuned Qwen LoRA adapter.")
    parser.add_argument("--adapter_path", default="outputs/qwen3-8b-thinking-whoandwhen")
    parser.add_argument("--question", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
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

    messages = [
        {
            "role": "system",
            "content": "You are a careful historian and fact-checking assistant.",
        },
        {"role": "user", "content": args.question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[-1] :]
    answer = tokenizer.decode(generated, skip_special_tokens=True)
    print(answer.strip())


if __name__ == "__main__":
    main()
