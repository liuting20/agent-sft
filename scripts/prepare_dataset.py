from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import DatasetDict, load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from whoandwhen_sft.data_utils import SYSTEM_PROMPT, FieldMapping, detect_fields, normalize_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the Who_and_When dataset for Qwen SFT.")
    parser.add_argument("--dataset_name", default="Kevin355/Who_and_When")
    parser.add_argument("--dataset_config", default="Algorithm-Generated")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--question_field", default=None)
    parser.add_argument("--answer_field", default=None)
    parser.add_argument("--system_prompt", default=SYSTEM_PROMPT)
    parser.add_argument("--val_size", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset_name, args.dataset_config)

    if "train" in dataset:
        train_dataset = dataset["train"]
    else:
        first_split = next(iter(dataset.keys()))
        train_dataset = dataset[first_split]

    if args.question_field and args.answer_field:
        mapping = FieldMapping(question_field=args.question_field, answer_field=args.answer_field)
    else:
        mapping = detect_fields(train_dataset.column_names)

    normalized = train_dataset.map(
        lambda example: normalize_record(example, mapping=mapping, system_prompt=args.system_prompt),
        remove_columns=train_dataset.column_names,
        desc="Normalizing records",
    )

    split_dataset = normalized.train_test_split(test_size=args.val_size, seed=args.seed)
    prepared = DatasetDict(train=split_dataset["train"], validation=split_dataset["test"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared.save_to_disk(str(output_dir))

    print(f"Saved processed dataset to {output_dir.resolve()}")
    print(f"Detected question field: {mapping.question_field}")
    print(f"Detected answer field: {mapping.answer_field}")
    print(f"Train size: {len(prepared['train'])}")
    print(f"Validation size: {len(prepared['validation'])}")


if __name__ == "__main__":
    main()
