from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA SFT for Qwen on Who_and_When.")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-8B")
    parser.add_argument("--dataset_path", default="data/processed")
    parser.add_argument("--output_dir", default="outputs/qwen3-8b-thinking-whoandwhen")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--disable_4bit", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--attn_implementation", default=None)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def formatting_prompts_func(example: dict[str, list]) -> list[str]:
    raise NotImplementedError("This function is replaced at runtime with tokenizer-aware formatting.")


def main() -> None:
    args = parse_args()
    dataset = load_from_disk(args.dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def formatting_prompts_func(example: dict[str, list]) -> list[str]:
        texts = []
        for messages in example["messages"]:
            texts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
        return texts

    use_4bit = not args.disable_4bit
    quantization_config = None
    torch_dtype = torch.bfloat16 if args.use_bf16 else torch.float16

    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )

    model_kwargs = {
        "torch_dtype": torch_dtype,
        "quantization_config": quantization_config,
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
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=args.use_bf16,
        fp16=not args.use_bf16,
        report_to="none",
        dataset_text_field=None,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        formatting_func=formatting_prompts_func,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    final_path = Path(args.output_dir).resolve()
    print(f"Training complete. Adapter saved to: {final_path}")


if __name__ == "__main__":
    main()
