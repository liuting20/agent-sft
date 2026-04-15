from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

QUESTION_CANDIDATES = (
    "question",
    "prompt",
    "instruction",
    "query",
    "input",
    "problem",
)

ANSWER_CANDIDATES = (
    "answer",
    "response",
    "output",
    "target",
    "label",
    "completion",
)

SYSTEM_PROMPT = (
    "You are a careful historian and fact-checking assistant. "
    "Read the question, reason carefully, and provide a concise final answer."
)


@dataclass
class FieldMapping:
    question_field: str
    answer_field: str


def detect_fields(column_names: list[str]) -> FieldMapping:
    question_field = _match_column(column_names, QUESTION_CANDIDATES)
    answer_field = _match_column(column_names, ANSWER_CANDIDATES)

    if question_field is None or answer_field is None:
        joined = ", ".join(column_names)
        raise ValueError(
            "Could not automatically detect question/answer fields. "
            f"Available columns: {joined}. "
            "Please pass --question_field and --answer_field explicitly."
        )

    return FieldMapping(question_field=question_field, answer_field=answer_field)


def build_messages(question: str, answer: str, system_prompt: str = SYSTEM_PROMPT) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question.strip()},
        {"role": "assistant", "content": answer.strip()},
    ]


def stringify_example(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def normalize_record(
    example: dict[str, Any],
    mapping: FieldMapping,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict[str, Any]:
    question = stringify_example(example[mapping.question_field]).strip()
    answer = stringify_example(example[mapping.answer_field]).strip()

    if not question or not answer:
        raise ValueError("Question or answer is empty after normalization.")

    messages = build_messages(question=question, answer=answer, system_prompt=system_prompt)
    return {
        "messages": messages,
        "question": question,
        "answer": answer,
    }


def _match_column(column_names: list[str], candidates: tuple[str, ...]) -> str | None:
    lower_map = {name.lower(): name for name in column_names}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    return None
