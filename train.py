import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

try:
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
except ImportError:
    LoraConfig = None
    PeftModel = None
    get_peft_model = None
    prepare_model_for_kbit_training = None


SYSTEM_PROMPT = "You are a helpful assistant skilled in analyzing conversations."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a causal language model for automated failure attribution."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the dataset JSON files, e.g. ../data/Algorithm-Generated",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base causal LM to fine-tune.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/qwen_fa_lora",
        help="Directory to save checkpoints and the final fine-tuned model.",
    )
    parser.add_argument(
        "--eval_ratio",
        type=float,
        default=0.2,
        help="Fraction of files used as eval set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data split and training.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=3.0,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Train batch size per device.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Eval batch size per device.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help="Warmup ratio.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Logging frequency.",
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="epoch",
        choices=["no", "steps", "epoch"],
        help="Checkpoint save strategy.",
    )
    parser.add_argument(
        "--eval_strategy",
        type=str,
        default="epoch",
        choices=["no", "steps", "epoch"],
        help="Trainer eval strategy.",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="Maximum number of checkpoints to keep.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=96,
        help="Maximum new tokens for evaluation generation.",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Enable LoRA fine-tuning.",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout.",
    )
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj",
        help="Comma-separated LoRA target modules.",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load base model in 4-bit for QLoRA. Requires bitsandbytes + peft.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 training if supported.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use float16 training.",
    )
    parser.add_argument(
        "--save_merged_model",
        action="store_true",
        help="After LoRA training, save a merged full model copy as well.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to Hugging Face loaders.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sorted_json_files(data_dir: str) -> List[str]:
    files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
    return sorted(files, key=lambda name: int("".join(filter(str.isdigit, name)) or 0))


def load_example(file_path: str) -> Dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_agent_key(example: Dict) -> str:
    history = example.get("history", [])
    if not history:
        return "name"
    if "name" in history[0]:
        return "name"
    return "role"


def build_user_prompt(example: Dict) -> str:
    agent_key = infer_agent_key(example)
    chat_history = example.get("history", [])
    problem = example.get("question", "")
    ground_truth = example.get("ground_truth", "")
    chat_content = "\n".join(
        f"{entry.get(agent_key, 'Unknown Agent')}: {entry.get('content', '')}"
        for entry in chat_history
    )

    return (
        "You are an AI assistant tasked with analyzing a multi-agent conversation history "
        "when solving a real world problem. "
        f"The problem is: {problem}\n"
        f"The Answer for the problem is: {ground_truth}\n"
        "Identify which agent made an error, at which step, and explain the reason for the error. "
        "Here's the conversation:\n\n"
        f"{chat_content}\n\n"
        "Based on this conversation, please predict the following:\n"
        "1. The name of the agent who made a mistake that should be directly responsible for the wrong solution to the real world problem. "
        "If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert.\n"
        "2. In which step the mistake agent first made mistake. For example, in a conversation structured as follows:\n"
        "{\n"
        "\"agent a\": \"xx\",\n"
        "\"agent b\": \"xxxx\",\n"
        "\"agent c\": \"xxxxx\",\n"
        "\"agent a\": \"xxxxxxx\"\n"
        "}\n"
        "each entry represents a 'step' where an agent provides input. The 'x' symbolizes the speech of each agent. "
        "If the mistake is in agent c's speech, the step number is 2. If the second speech by 'agent a' contains the mistake, "
        "the step number is 3, and so on. Please determine the step number where the first mistake occurred.\n"
        "3. The reason for your prediction.\n"
        "Please answer in the format:\n"
        "Agent Name: (Your prediction)\n"
        "Step Number: (Your prediction)\n"
        "Reason for Mistake: (Your reason)\n"
    )


def build_target_text(example: Dict) -> str:
    return (
        f"Agent Name: {example['mistake_agent']}\n"
        f"Step Number: {example['mistake_step']}\n"
        f"Reason for Mistake: {example.get('mistake_reason', '')}\n"
    )


def build_messages(example: Dict, include_target: bool) -> List[Dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(example)},
    ]
    if include_target:
        messages.append({"role": "assistant", "content": build_target_text(example)})
    return messages


def render_chat(
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    text = ""
    for message in messages:
        text += f"{message['role'].upper()}: {message['content']}\n"
    return text


class SupervisedChatDataset(Dataset):
    def __init__(self, examples: List[Dict], tokenizer: AutoTokenizer, max_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        example = self.examples[idx]
        full_text = render_chat(self.tokenizer, build_messages(example, include_target=True))
        prompt_text = render_chat(self.tokenizer, build_messages(example, include_target=False))

        full_encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )
        prompt_encoding = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )

        input_ids = full_encoding["input_ids"]
        attention_mask = full_encoding["attention_mask"]
        prompt_len = min(len(prompt_encoding["input_ids"]), len(input_ids))
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


@dataclass
class SupervisedDataCollator:
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        pad_token_id = self.tokenizer.pad_token_id
        max_len = max(len(item["input_ids"]) for item in features)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for item in features:
            seq_len = len(item["input_ids"])
            pad_len = max_len - seq_len
            batch_input_ids.append(item["input_ids"] + [pad_token_id] * pad_len)
            batch_attention_mask.append(item["attention_mask"] + [0] * pad_len)
            batch_labels.append(item["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def train_eval_split(examples: List[Dict], eval_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    if not 0.0 < eval_ratio < 1.0:
        raise ValueError("--eval_ratio must be in the range (0, 1).")
    shuffled = examples[:]
    random.Random(seed).shuffle(shuffled)
    eval_size = max(1, int(len(shuffled) * eval_ratio))
    eval_examples = shuffled[:eval_size]
    train_examples = shuffled[eval_size:]
    if not train_examples:
        raise ValueError("Training split is empty. Reduce --eval_ratio or provide more data.")
    return train_examples, eval_examples


def parse_prediction(text: str) -> Dict[str, str]:
    agent_match = re.search(r"Agent Name:\s*([^\n\r]+)", text, re.IGNORECASE)
    step_match = re.search(r"Step Number:\s*([^\n\r]+)", text, re.IGNORECASE)
    reason_match = re.search(r"Reason for Mistake:\s*(.*)", text, re.IGNORECASE | re.DOTALL)

    predicted_agent = agent_match.group(1).strip() if agent_match else ""
    predicted_step = step_match.group(1).strip() if step_match else ""
    predicted_reason = reason_match.group(1).strip() if reason_match else ""

    predicted_step_digits = re.search(r"\d+", predicted_step)
    if predicted_step_digits:
        predicted_step = predicted_step_digits.group(0)

    return {
        "predicted_agent": predicted_agent,
        "predicted_step": predicted_step,
        "predicted_reason": predicted_reason,
    }


def evaluate_predictions(predictions: List[Dict], total_files: int) -> Dict[str, float]:
    correct_agent = 0
    correct_step = 0

    for item in predictions:
        actual_agent = str(item["actual_agent"])
        actual_step = str(item["actual_step"])
        predicted_agent = item["predicted_agent"]
        predicted_step = item["predicted_step"]

        if actual_agent and actual_agent in predicted_agent:
            correct_agent += 1
        if actual_step and actual_step in predicted_step:
            correct_step += 1

    agent_accuracy = (correct_agent / total_files) * 100 if total_files else 0.0
    step_accuracy = (correct_step / total_files) * 100 if total_files else 0.0
    joint_accuracy = (
        sum(
            1
            for item in predictions
            if str(item["actual_agent"]) in item["predicted_agent"]
            and str(item["actual_step"]) in item["predicted_step"]
        )
        / total_files
        * 100
        if total_files
        else 0.0
    )
    return {
        "agent_accuracy": agent_accuracy,
        "step_accuracy": step_accuracy,
        "joint_accuracy": joint_accuracy,
        "correct_agent": correct_agent,
        "correct_step": correct_step,
        "total_files": total_files,
    }


def save_prediction_log(predictions: List[Dict], log_path: str) -> None:
    with open(log_path, "w", encoding="utf-8") as f:
        for item in predictions:
            f.write(f"Prediction for {item['file_name']}:\n")
            f.write(f"Agent Name: {item['predicted_agent']}\n")
            f.write(f"Step Number: {item['predicted_step']}\n")
            f.write(f"Reason for Mistake: {item['predicted_reason']}\n")
            f.write("\n" + "=" * 50 + "\n\n")


def ensure_tokenizer_padding(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def build_quantization_config(load_in_4bit: bool) -> Optional[BitsAndBytesConfig]:
    if not load_in_4bit:
        return None
    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "--load_in_4bit requires bitsandbytes to be installed."
        ) from exc

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )


def load_model_and_tokenizer(args: argparse.Namespace):
    quantization_config = build_quantization_config(args.load_in_4bit)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    ensure_tokenizer_padding(tokenizer)

    model_kwargs = {
        "trust_remote_code": args.trust_remote_code,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        if args.bf16:
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif args.fp16:
            model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **model_kwargs,
    )
    model.config.use_cache = False

    if args.use_lora:
        if LoraConfig is None or get_peft_model is None:
            raise ImportError(
                "--use_lora requires peft. Please install peft first."
            )
        if args.load_in_4bit:
            if prepare_model_for_kbit_training is None:
                raise ImportError("prepare_model_for_kbit_training is unavailable.")
            model = prepare_model_for_kbit_training(model)

        target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model, tokenizer


def prepare_examples(data_dir: str) -> List[Dict]:
    examples = []
    for file_name in sorted_json_files(data_dir):
        file_path = os.path.join(data_dir, file_name)
        example = load_example(file_path)
        example["file_name"] = file_name
        examples.append(example)
    if not examples:
        raise ValueError(f"No JSON files found in {data_dir}")
    return examples


def generate_prediction(
    model,
    tokenizer,
    example: Dict,
    max_new_tokens: int,
) -> str:
    messages = build_messages(example, include_target=False)
    prompt_text = render_chat(tokenizer, messages)
    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = generated[0][prompt_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def run_generation_eval(
    model,
    tokenizer,
    eval_examples: List[Dict],
    max_new_tokens: int,
    output_dir: str,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()

    predictions = []
    for example in eval_examples:
        prediction_text = generate_prediction(model, tokenizer, example, max_new_tokens)
        parsed = parse_prediction(prediction_text)
        predictions.append(
            {
                "file_name": example["file_name"],
                "actual_agent": example["mistake_agent"],
                "actual_step": example["mistake_step"],
                **parsed,
            }
        )

    prediction_log_path = os.path.join(output_dir, "eval_predictions.txt")
    save_prediction_log(predictions, prediction_log_path)
    metrics = evaluate_predictions(predictions, total_files=len(eval_examples))
    metrics["prediction_log_path"] = prediction_log_path

    with open(os.path.join(output_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    if was_training:
        model.train()
    return metrics


def maybe_save_merged_model(model, tokenizer, output_dir: str) -> Optional[str]:
    if PeftModel is None or not isinstance(model, PeftModel):
        return None
    merged_dir = os.path.join(output_dir, "merged_model")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    return merged_dir


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    examples = prepare_examples(args.data_dir)
    train_examples, eval_examples = train_eval_split(examples, args.eval_ratio, args.seed)

    print(f"Loaded {len(examples)} examples from {args.data_dir}")
    print(f"Train examples: {len(train_examples)}")
    print(f"Eval examples: {len(eval_examples)}")

    model, tokenizer = load_model_and_tokenizer(args)

    train_dataset = SupervisedChatDataset(train_examples, tokenizer, args.max_length)
    eval_dataset = SupervisedChatDataset(eval_examples, tokenizer, args.max_length)
    data_collator = SupervisedDataCollator(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        evaluation_strategy=args.eval_strategy,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to="none",
        remove_unused_columns=False,
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if args.eval_strategy != "no" else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    final_model_dir = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    metrics = run_generation_eval(
        trainer.model,
        tokenizer,
        eval_examples,
        args.max_new_tokens,
        args.output_dir,
    )

    merged_model_dir = None
    if args.save_merged_model and args.use_lora:
        merged_model_dir = maybe_save_merged_model(trainer.model, tokenizer, args.output_dir)

    print("\n=== Final Results ===")
    print(f"Base model: {args.model_name_or_path}")
    print(f"Fine-tuned model saved to: {final_model_dir}")
    if merged_model_dir:
        print(f"Merged full model saved to: {merged_model_dir}")
    print(f"Prediction log saved to: {metrics['prediction_log_path']}")
    print(f"Agent Accuracy: {metrics['agent_accuracy']:.2f}%")
    print(f"Step Accuracy: {metrics['step_accuracy']:.2f}%")
    print(f"Joint Accuracy: {metrics['joint_accuracy']:.2f}%")
    print(
        f"(Accuracy is computed with the same denominator rule as evaluate.py: "
        f"{metrics['total_files']} eval files)"
    )


if __name__ == "__main__":
    main()
