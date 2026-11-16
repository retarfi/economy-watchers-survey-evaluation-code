"""Score the demonstration-set runs of the reason task (Supplementary Table S6).

Uses the same label-extraction rule as run_llm.py. Reads the llm-jp outputs in
../results/ written by run_llm.py:
    reason_llm-jp-31-13b-instruct4_0_seed{0,1,2}.jsonl         (0-shot, default)
    reason_llm-jp-31-13b-instruct4_5_seed{0,1,2}.jsonl         (5-shot, default set)
    reason_llm-jp-31-13b-instruct4_5_seed1_demo{A1..B3}.jsonl  (5-shot, alternative sets)
and writes demo_set_results.{csv,md}. Missing files are skipped.
"""
import json
import statistics
from pathlib import Path

import Levenshtein
import pandas as pd
from datasets import DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, f1_score

TASK: str = "reason"
MODEL: str = "llm-jp-31-13b-instruct4"
N_TEST: int = 15493
ROOT_DIR: Path = Path(__file__).resolve().parent
P_RESULTS: Path = ROOT_DIR.parent / "results"


def map_prediction(text: str, lst_labels: list[str]) -> int:
    """Same rule as map_prediction in run_llm.py for the reason task; -1 is invalid."""
    if text in lst_labels:
        return lst_labels.index(text)
    lst_cand: list[str] = [x for x in lst_labels if x in text]
    if len(lst_cand) == 1:
        return lst_labels.index(lst_cand[0])
    lst_dist: list[int] = [Levenshtein.distance(text, x) for x in lst_labels]
    if min(lst_dist) <= 3 and lst_dist.count(min(lst_dist)) == 1:
        return lst_dist.index(min(lst_dist))
    return -1


def score(p_jsonl: Path, lst_labels: list[str], lst_demo_labels: list[int]) -> dict:
    with open(p_jsonl, encoding="utf-8") as f:
        lst_rows: list[dict] = [json.loads(line) for line in f]
    lst_gold: list[int] = [int(x["label"]) for x in lst_rows]
    lst_pred: list[int] = [map_prediction(str(x["output"]), lst_labels) for x in lst_rows]
    lst_label_id: list[int] = list(range(len(lst_labels)))
    lst_f1: list[float] = f1_score(
        lst_gold, lst_pred, average=None, labels=lst_label_id, zero_division=0
    ).tolist()
    lst_valid: list[int] = [x for x in lst_pred if x != -1]
    lst_other_labels: list[int] = [x for x in lst_label_id if x not in lst_demo_labels]
    dct: dict = {
        "n": len(lst_rows),
        "complete": len(lst_rows) == N_TEST,
        "macro_f1": 100 * statistics.mean(lst_f1),
        "accuracy": 100 * accuracy_score(lst_gold, lst_pred),
        "invalid_rate": 100 * (1 - len(lst_valid) / len(lst_pred)),
        # share of valid predictions that fall in the demonstrated labels
        "pred_in_demo_labels": 100
        * sum(x in lst_demo_labels for x in lst_valid)
        / len(lst_valid),
        "gold_in_demo_labels": 100
        * sum(x in lst_demo_labels for x in lst_gold)
        / len(lst_gold),
        "f1_demo_labels": 100 * statistics.mean(lst_f1[x] for x in lst_demo_labels),
        "f1_other_labels": 100 * statistics.mean(lst_f1[x] for x in lst_other_labels),
    }
    for label_id, f1 in zip(lst_label_id, lst_f1):
        dct[f"f1_{label_id}"] = 100 * f1
    return dct


def main() -> None:
    dsd: DatasetDict = load_dataset(
        "retarfi/economy-watchers-survey-evaluation",
        TASK,
        ews_version="2024.05.2",
        trust_remote_code=True,
    )
    lst_labels: list[str] = dsd["train"].features["label"].names

    def demo_labels(name: str) -> list[int]:
        with open(ROOT_DIR / "demo_sets" / f"{TASK}_{name}.json", encoding="utf-8") as f:
            return [x["label_id"] for x in json.load(f)["examples"]]

    # (condition, demo set, generation seed, file); the 0-shot rows use the
    # labels of the original set as the reference for the *_demo_labels columns
    lst_runs: list[tuple[str, str, int, Path]] = []
    for seed in (0, 1, 2):
        lst_runs.append(
            ("0-shot", "orig", seed, P_RESULTS / f"{TASK}_{MODEL}_0_seed{seed}.jsonl")
        )
    for seed in (0, 1, 2):
        lst_runs.append(
            ("5-shot orig", "orig", seed, P_RESULTS / f"{TASK}_{MODEL}_5_seed{seed}.jsonl")
        )
    for p_demo in sorted((ROOT_DIR / "demo_sets").glob(f"{TASK}_[AB]*.json")):
        name: str = p_demo.stem.split("_")[1]
        lst_runs.append(
            (f"5-shot {name}", name, 1, P_RESULTS / f"{TASK}_{MODEL}_5_seed1_demo{name}.jsonl")
        )

    lst_records: list[dict] = []
    for condition, name, seed, p_jsonl in lst_runs:
        if not p_jsonl.exists():
            print(f"missing: {p_jsonl.name}")
            continue
        lst_demo: list[int] = demo_labels(name)
        dct: dict = {
            "condition": condition,
            "seed": seed,
            "demo_labels": " ".join(map(str, lst_demo)),
        }
        dct.update(score(p_jsonl, lst_labels, lst_demo))
        lst_records.append(dct)
    df: pd.DataFrame = pd.DataFrame(lst_records)
    df.to_csv(ROOT_DIR / "demo_set_results.csv", index=False)

    lst_cols: list[str] = [
        "condition", "seed", "demo_labels", "n", "complete", "macro_f1", "accuracy",
        "invalid_rate", "pred_in_demo_labels", "gold_in_demo_labels",
        "f1_demo_labels", "f1_other_labels",
    ]
    lst_md: list[str] = [
        "# Reason task, llm-jp-3.1-13b: demonstration-set variation",
        "",
        "Label ids: " + ", ".join(f"{i}={x}" for i, x in enumerate(lst_labels)),
        "",
        "| " + " | ".join(lst_cols) + " |",
        "|" + "---|" * len(lst_cols),
    ]
    for record in df[lst_cols].to_dict("records"):
        lst_md.append(
            "| "
            + " | ".join(
                f"{x:.2f}" if isinstance(x, float) else str(x) for x in record.values()
            )
            + " |"
        )
    lst_md += [
        "",
        "## Macro F1 summary (complete runs only; SD is the sample SD)",
        "",
        "| group | runs | mean | SD | min | max |",
        "|---|---|---|---|---|---|",
    ]
    df_done: pd.DataFrame = df[df["complete"]]
    dct_groups: dict[str, pd.Series] = {
        "0-shot (3 generation seeds)": df_done["condition"] == "0-shot",
        "5-shot orig (3 generation seeds)": df_done["condition"] == "5-shot orig",
        "5-shot A1-A3": df_done["condition"].str.startswith("5-shot A"),
        "5-shot B1-B3": df_done["condition"].str.startswith("5-shot B"),
        "5-shot A and B": df_done["condition"].str.match(r"5-shot [AB]"),
    }
    for group, mask in dct_groups.items():
        lst_f1: list[float] = df_done.loc[mask, "macro_f1"].tolist()
        if len(lst_f1) < 2:
            continue
        lst_md.append(
            f"| {group} | {len(lst_f1)} | {statistics.mean(lst_f1):.2f} "
            f"| {statistics.stdev(lst_f1):.2f} | {min(lst_f1):.2f} | {max(lst_f1):.2f} |"
        )
    (ROOT_DIR / "demo_set_results.md").write_text("\n".join(lst_md) + "\n", encoding="utf-8")
    print("\n".join(lst_md))


if __name__ == "__main__":
    main()
