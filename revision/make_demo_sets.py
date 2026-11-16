"""Build alternative 5-shot demonstration sets (Supplementary Information Section S4).

orig: the set run_llm.py builds by default (deterministic; kept for verification)
A1-3: same labels as orig, one randomly drawn training example per label
B1-3: five labels drawn at random from all labels, one random example each

Writes demo_sets/<task>_<name>.json, which run_llm.py accepts via --demo-set.
The sets used in the article were built with: python make_demo_sets.py reason
"""
import argparse
import json
import math
import random
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

EWS_VERSION: str = "2024.05.2"
MIN_LEN: int = 30  # run_llm.py only uses texts with len(text) > 30
ROOT_DIR: Path = Path(__file__).resolve().parent


def original_indices(train: Dataset, n_labels: int, shots: int) -> list[int]:
    """Replicates the selection in run_llm.py (including train[-0] == train[0])."""
    lst_contexts: list[list[int]] = [[] for _ in range(n_labels)]
    for i in range(len(train)):
        idx: int = (-i) % len(train)
        example: dict = train[idx]
        label: int = example["label"]
        if len(lst_contexts[label]) < math.ceil(
            (shots - label) / n_labels
        ) and len(example["text"]) > MIN_LEN:
            lst_contexts[label].append(idx)
        if sum(map(len, lst_contexts)) == shots:
            break
    lst_idx: list[int] = []
    for i in range(math.ceil(shots / n_labels)):
        for j in range(n_labels):
            if i < len(lst_contexts[j]):
                lst_idx.append(lst_contexts[j][i])
    return lst_idx


def main(task: str, shots: int, n_sets: int) -> None:
    dsd: DatasetDict = load_dataset(
        "retarfi/economy-watchers-survey-evaluation",
        task,
        ews_version=EWS_VERSION,
        trust_remote_code=True,
    )
    train: Dataset = dsd["train"]
    lst_labels: list[str] = train.features["label"].names
    n_labels: int = len(lst_labels)

    lst_orig: list[int] = original_indices(train, n_labels, shots)
    lst_train_label: list[int] = train["label"]
    lst_train_len: list[int] = [len(x) for x in train["text"]]
    lst_cands: list[list[int]] = [[] for _ in range(n_labels)]
    for idx, (label, length) in enumerate(zip(lst_train_label, lst_train_len)):
        if length > MIN_LEN and idx not in lst_orig:
            lst_cands[label].append(idx)

    dct_sets: dict[str, tuple[str, list[int]]] = {
        "orig": ("default selection in run_llm.py", lst_orig)
    }
    lst_orig_labels: list[int] = [lst_train_label[i] for i in lst_orig]
    for k in range(1, n_sets + 1):
        rng = random.Random(f"{task}-A-{k}")
        dct_sets[f"A{k}"] = (
            "same labels as orig, random example per label",
            [rng.choice(lst_cands[label]) for label in lst_orig_labels],
        )
        rng = random.Random(f"{task}-B-{k}")
        lst_b_labels: list[int] = sorted(rng.sample(range(n_labels), shots))
        dct_sets[f"B{k}"] = (
            "random labels, random example per label",
            [rng.choice(lst_cands[label]) for label in lst_b_labels],
        )

    p_dir: Path = ROOT_DIR / "demo_sets"
    p_dir.mkdir(exist_ok=True)
    for name, (description, lst_idx) in dct_sets.items():
        lst_examples: list[dict] = []
        for idx in lst_idx:
            example: dict = train[idx]
            lst_examples.append(
                {
                    "train_index": idx,
                    "id": example.get("id"),
                    "label_id": example["label"],
                    "label": lst_labels[example["label"]],
                    "text": example["text"],
                }
            )
        dct_out: dict = {
            "name": name,
            "task": task,
            "ews_version": EWS_VERSION,
            "description": description,
            "examples": lst_examples,
        }
        with open(p_dir / f"{task}_{name}.json", "w", encoding="utf-8") as f:
            json.dump(dct_out, f, ensure_ascii=False, indent=2)
        print(name, [x["label_id"] for x in lst_examples], [x["train_index"] for x in lst_examples])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["sentiment", "domain", "reason"])
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--n-sets", type=int, default=3)
    args = parser.parse_args()
    main(task=args.task, shots=args.shots, n_sets=args.n_sets)
