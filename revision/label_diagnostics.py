"""Label-wise diagnostics of reason-task outputs (Supplementary Tables S7 and S8).

For each output file, reports macro F1, accuracy, the invalid-output rate, the
share of valid outputs that fall in the demonstrated labels, the mean per-label
F1 over the demonstrated and the other labels, and the per-label F1 scores.
No inference is run; the files are those written by run_llm.py.
    python label_diagnostics.py \
        --pred "ChatGPT 0-shot=../results/reason_gpt-35-turbo-0125_0.jsonl" \
        --pred "ChatGPT 5-shot=../results/reason_gpt-35-turbo-0125_5.jsonl" ...
By default the demonstrated labels are those of the default 5-shot prompt
(label ids 0-4); use --demo-set to take them from a demonstration-set file.
"""
import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import Dataset, load_dataset

from score_demo_sets import score

ROOT_DIR: Path = Path(__file__).resolve().parent


def main(lst_pred: list[str], p_demo_set: Optional[Path], p_out: Path) -> None:
    test: Dataset = load_dataset(
        "retarfi/economy-watchers-survey-evaluation",
        "reason",
        ews_version="2024.05.2",
        split="test",
        trust_remote_code=True,
    )
    lst_labels: list[str] = test.features["label"].names
    lst_demo: list[int] = [0, 1, 2, 3, 4]
    if p_demo_set is not None:
        dct_demo: dict = json.loads(p_demo_set.read_text(encoding="utf-8"))
        lst_demo = [x["label_id"] for x in dct_demo["examples"]]

    lst_records: list[dict] = []
    for spec in lst_pred:
        name, path = spec.split("=", 1)
        lst_records.append({"name": name, **score(Path(path), lst_labels, lst_demo)})
    df: pd.DataFrame = pd.DataFrame(lst_records)
    df.to_csv(p_out.with_suffix(".csv"), index=False)

    lst_md: list[str] = [
        "# Reason task: label-wise diagnostics",
        "",
        f"Demonstrated labels: {', '.join(f'{i}={lst_labels[i]}' for i in lst_demo)}",
        f"Share of the official test labels in the demonstrated labels: "
        f"{df['gold_in_demo_labels'].iloc[0]:.2f}%",
        "",
        "| output | n | complete | macro F1 | accuracy | invalid % | pred in demo labels % "
        "| F1 demo labels | F1 other labels |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in df.to_dict("records"):
        lst_md.append(
            f"| {r['name']} | {r['n']} | {r['complete']} | {r['macro_f1']:.2f} "
            f"| {r['accuracy']:.2f} | {r['invalid_rate']:.2f} | {r['pred_in_demo_labels']:.2f} "
            f"| {r['f1_demo_labels']:.2f} | {r['f1_other_labels']:.2f} |"
        )
    lst_md += [
        "",
        "## Per-label F1",
        "",
        "| label | " + " | ".join(df["name"]) + " |",
        "|---|" + "---|" * len(df),
    ]
    for i, label in enumerate(lst_labels):
        lst_md.append(
            f"| {i} {label} | " + " | ".join(f"{x:.1f}" for x in df[f"f1_{i}"]) + " |"
        )
    p_out.with_suffix(".md").write_text("\n".join(lst_md) + "\n", encoding="utf-8")
    print("\n".join(lst_md))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred", action="append", required=True, help="name=path of a run_llm.py output"
    )
    parser.add_argument("--demo-set", type=Path, help="demonstration-set JSON (default: ids 0-4)")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT_DIR / "label_diagnostics.md",
        help="output path; a .csv with the same stem is also written",
    )
    args = parser.parse_args()
    main(lst_pred=args.pred, p_demo_set=args.demo_set, p_out=args.out)
