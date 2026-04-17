from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path
from typing import Any

import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen on the Who and When dataset.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--dataset_path", default="data/processed")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output_dir", default="outputs/eval-qwen3-8b-thinking")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--attn_implementation", default=None)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("\u2019", "'")
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def extract_final_answer(text: str) -> str:
    cleaned = text.strip()
    patterns = [
        r"(?is)final answer\s*[:：]\s*(.+)$",
        r"(?is)answer\s*[:：]\s*(.+)$",
        r"(?is)<answer>\s*(.+?)\s*</answer>",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            return match.group(1).strip()

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[-1]


def exact_match(prediction: str, reference: str) -> bool:
    return normalize_text(prediction) == normalize_text(reference)


def substring_match(prediction: str, reference: str) -> bool:
    pred = normalize_text(prediction)
    ref = normalize_text(reference)
    return bool(pred and ref) and (pred in ref or ref in pred)


def batched(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def build_prompts(tokenizer: AutoTokenizer, examples: list[dict[str, Any]]) -> list[str]:
    prompts = []
    for example in examples:
        if "messages" in example and example["messages"]:
            messages = example["messages"][:-1]
        else:
            messages = [
                {
                    "role": "system",
                    "content": "You are a careful historian and fact-checking assistant.",
                },
                {"role": "user", "content": example["question"]},
            ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
    return prompts


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_from_disk(args.dataset_path)
    if args.split not in dataset:
        raise ValueError(f"Split '{args.split}' not found. Available splits: {list(dataset.keys())}")
    eval_dataset = dataset[args.split]

    torch_dtype = torch.bfloat16 if args.use_bf16 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    except Exception as exc:
        if args.attn_implementation == "flash_attention_2":
            print(
                "Warning: failed to enable flash_attention_2, falling back to the model default attention. "
                f"Original error: {exc}"
            )
            model_kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
        else:
            raise

    model.eval()

    records = [eval_dataset[index] for index in range(len(eval_dataset))]
    results: list[dict[str, Any]] = []

    for batch in tqdm(batched(records, args.batch_size), desc=f"Evaluating {args.split}"):
        prompts = build_prompts(tokenizer, batch)
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": args.temperature > 0,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }

        with torch.inference_mode():
            outputs = model.generate(**inputs, **generation_kwargs)

        prompt_length = inputs["input_ids"].shape[1]
        generated_sequences = outputs[:, prompt_length:]
        decoded = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)

        for example, raw_prediction in zip(batch, decoded):
            prediction = extract_final_answer(raw_prediction)
            reference = example["answer"]
            result = {
                "question": example["question"],
                "reference_answer": reference,
                "raw_prediction": raw_prediction.strip(),
                "final_prediction": prediction,
                "exact_match": exact_match(prediction, reference),
                "substring_match": substring_match(prediction, reference),
            }
            results.append(result)

    exact_accuracy = sum(item["exact_match"] for item in results) / len(results) if results else 0.0
    substring_accuracy = sum(item["substring_match"] for item in results) / len(results) if results else 0.0

    metrics = {
        "model_name_or_path": args.model_name_or_path,
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "split": args.split,
        "num_examples": len(results),
        "exact_match_accuracy": exact_accuracy,
        "substring_match_accuracy": substring_accuracy,
    }

    metrics_path = output_dir / "metrics.json"
    predictions_path = output_dir / "predictions.jsonl"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    with predictions_path.open("w", encoding="utf-8") as file:
        for item in results:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved metrics to: {metrics_path.resolve()}")
    print(f"Saved predictions to: {predictions_path.resolve()}")


if __name__ == "__main__":
    main()
