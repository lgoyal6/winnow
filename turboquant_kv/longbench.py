"""LongBench-E task configs and official deterministic metrics.

Winnow's existing eval (`experiments/`) scores prompt compression by asking
Claude Haiku to answer and comparing to gold. That is fine for what it measures,
but it never touches the KV cache, and a scorer that is itself a model call is
the wrong instrument for detecting a small quantization regression: it adds its
own variance on top of the effect being measured.

LongBench-E is used here instead because its metrics are pure string functions
(no API, no sampling), it ships length-bucketed splits so a score can be read
per context length, and the retrieval-style tasks are the ones a lossy KV cache
should actually break first.

Prompt templates and `max_gen` are the official ones from the LongBench repo
(`config/dataset2prompt.json`, `config/dataset2maxlen.json`); scoring follows
`metrics.py`. Changing either would make these numbers incomparable to published
LongBench results, so they are copied verbatim rather than paraphrased.
"""
from __future__ import annotations

import json
import os
import re
import string
import zipfile
from collections import Counter

_CACHE = None


def load_task(task: str, split_e: bool = True):
    """Load one LongBench task as a list of dicts.

    `datasets.load_dataset("THUDM/LongBench", ...)` no longer works: the repo
    ships a loading script and current `datasets` refuses to execute those
    ("Dataset scripts are no longer supported"). The repo's only data artifact
    is a single `data.zip`, so it is fetched once and read directly.
    """
    global _CACHE
    from huggingface_hub import hf_hub_download

    if _CACHE is None:
        z = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
        d = os.path.join(os.path.dirname(z), "_extracted")
        if not os.path.isdir(d):
            with zipfile.ZipFile(z) as f:
                f.extractall(d)
        # The zip nests everything under a `data/` directory.
        root = os.path.join(d, "data")
        _CACHE = root if os.path.isdir(root) else d

    name = f"{task}_e" if split_e else task
    path = os.path.join(_CACHE, f"{name}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found; available: "
            f"{sorted(os.listdir(_CACHE))[:12]}")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

# Task -> (prompt template, max generated tokens, metric name).
# Deliberately biased toward short generations: the point is to score many
# samples across context lengths, not to benchmark summarization throughput.
TASKS = {
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\nNow, "
        "answer the following question based on the above text, only give me "
        "the answer and do not output any other words.\n\nQuestion: {input}\n"
        "Answer:", 64, "qa_f1"),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\nThe following are given "
        "passages.\n{context}\n\nAnswer the question based on the given "
        "passages. Only give me the answer and do not output any other words."
        "\n\nQuestion: {input}\nAnswer:", 32, "qa_f1"),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\nThe following are given "
        "passages.\n{context}\n\nAnswer the question based on the given "
        "passages. Only give me the answer and do not output any other words."
        "\n\nQuestion: {input}\nAnswer:", 32, "qa_f1"),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. Please "
        "determine which paragraph the abstract is from.\n\n{context}\n\nThe "
        "following is an abstract.\n\n{input}\n\nPlease enter the number of the "
        "paragraph that the abstract is from. The answer format must be like "
        "\"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
        32, "retrieval"),
}


# ---------------------------------------------------------------------------
# metrics (LongBench metrics.py)
# ---------------------------------------------------------------------------
def normalize_answer(s: str) -> str:
    def remove_articles(t):
        return re.sub(r"\b(a|an|the)\b", " ", t)

    def white_space_fix(t):
        return " ".join(t.split())

    def remove_punc(t):
        return "".join(ch for ch in t if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def qa_f1_score(prediction: str, ground_truth: str) -> float:
    pred = normalize_answer(prediction).split()
    gold = normalize_answer(ground_truth).split()
    common = Counter(pred) & Counter(gold)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred)
    recall = num_same / len(gold)
    return 2 * precision * recall / (precision + recall)


def retrieval_score(prediction: str, ground_truth: str) -> float:
    """Gold looks like 'Paragraph 12'; credit only an exact paragraph match."""
    pattern = r"Paragraph (\d+)"
    matches = re.findall(pattern, ground_truth)
    if not matches:
        return 0.0
    gold_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    return 1.0 if numbers and numbers[0] == gold_id else 0.0


METRICS = {"qa_f1": qa_f1_score, "retrieval": retrieval_score}


def score(metric: str, prediction: str, answers: list[str]) -> float:
    """Best score over the gold aliases, as LongBench does."""
    fn = METRICS[metric]
    return max((fn(prediction, a) for a in answers), default=0.0)


def truncate_middle(tokenizer, prompt: str, max_len: int) -> str:
    """LongBench truncates from the middle, keeping both ends.

    Chopping the tail would remove the question and chopping the head would
    remove the instruction, so either one changes the task rather than the
    context length.
    """
    ids = tokenizer(prompt, truncation=False, return_tensors=None).input_ids
    if len(ids) <= max_len:
        return prompt
    half = max_len // 2
    return (tokenizer.decode(ids[:half], skip_special_tokens=True)
            + tokenizer.decode(ids[-half:], skip_special_tokens=True))
