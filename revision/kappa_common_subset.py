"""Pairwise Cohen's kappa on common valid subsets (Supplementary Table S5).

For each model pair, kappa is reported for the 0-shot and 5-shot settings on
(a) the instances valid for both models in that setting (as in Fig. S1) and
(b) the instances valid for both models in both settings (common subset),
and the share of instances whose valid outputs contain at least two distinct
labels (the inter-model disagreement proportion reported in the article).
    python kappa_common_subset.py \
        --zero ChatGPT=../results/reason_gpt-35-turbo-0125_0.jsonl ... \
        --five ChatGPT=../results/reason_gpt-35-turbo-0125_5.jsonl ...
Give the same models in the same order for --zero and --five.
"""
import argparse
import itertools
import json
from pathlib import Path

from datasets import Dataset, load_dataset
from sklearn.metrics import cohen_kappa_score

from score_demo_sets import N_TEST, map_prediction

ROOT_DIR: Path = Path(__file__).resolve().parent


def load(
    spec: str, lst_labels: list[str], lst_all_ids: list[str]
) -> tuple[str, dict[str, int]]:
    name, path = spec.split("=", 1)
    # instances without an output (e.g. failed API calls) are treated as invalid
    dct_pred: dict[str, int] = {x: -1 for x in lst_all_ids}
    n_rows: int = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            d: dict = json.loads(line)
            assert d["id"] in dct_pred, f"{path}: unknown id {d['id']}"
            dct_pred[d["id"]] = map_prediction(str(d["output"]), lst_labels)
            n_rows += 1
    if n_rows != N_TEST:
        print(f"WARNING: {path} has {n_rows} of {N_TEST} instances")
    return name, dct_pred


def kappa(dct_a: dict[str, int], dct_b: dict[str, int], lst_ids: list[str]) -> str:
    value: float = cohen_kappa_score([dct_a[x] for x in lst_ids], [dct_b[x] for x in lst_ids])
    return f"{value:.2f} (n={len(lst_ids):,})"


def share_different(dct_preds: dict[str, dict[str, int]], lst_ids: list[str]) -> str:
    k: int = sum(len({dct_pred[x] for dct_pred in dct_preds.values()}) > 1 for x in lst_ids)
    return f"{k:,}/{len(lst_ids):,} = {100 * k / len(lst_ids):.1f}%"


def main(lst_zero: list[str], lst_five: list[str], p_out: Path) -> None:
    test: Dataset = load_dataset(
        "retarfi/economy-watchers-survey-evaluation",
        "reason",
        ews_version="2024.05.2",
        split="test",
        trust_remote_code=True,
    )
    lst_labels: list[str] = test.features["label"].names
    lst_all_ids: list[str] = sorted(test["id"])
    dct_zero: dict[str, dict[str, int]] = dict(
        load(x, lst_labels, lst_all_ids) for x in lst_zero
    )
    dct_five: dict[str, dict[str, int]] = dict(
        load(x, lst_labels, lst_all_ids) for x in lst_five
    )
    assert list(dct_zero) == list(dct_five), "give the same models in the same order"
    lst_models: list[str] = list(dct_zero)

    def valid(dct_preds: dict[str, dict[str, int]], lst_names: list[str]) -> set[str]:
        return {x for x in lst_all_ids if all(dct_preds[m][x] != -1 for m in lst_names)}

    lst_md: list[str] = [
        "# Reason task: pairwise Cohen's kappa, 0-shot vs 5-shot",
        "",
        "## Valid single-label outputs",
        "",
        "| model | 0-shot | 5-shot | both |",
        "|---|---|---|---|",
    ]
    for m in lst_models:
        lst_md.append(
            f"| {m} | {len(valid(dct_zero, [m])):,} | {len(valid(dct_five, [m])):,} "
            f"| {len(valid(dct_zero, [m]) & valid(dct_five, [m])):,} |"
        )

    lst_md += [
        "",
        "## Pairwise kappa",
        "",
        "Setting-specific: instances valid for both models in that setting (Fig. S1). "
        "Common: instances valid for both models in both settings.",
        "",
        "| pair | 0-shot, setting-specific | 5-shot, setting-specific "
        "| 0-shot, common | 5-shot, common |",
        "|---|---|---|---|---|",
    ]
    for a, b in itertools.combinations(lst_models, 2):
        lst_v0: list[str] = sorted(valid(dct_zero, [a, b]))
        lst_v5: list[str] = sorted(valid(dct_five, [a, b]))
        lst_common: list[str] = sorted(set(lst_v0) & set(lst_v5))
        lst_md.append(
            f"| {a}--{b} | {kappa(dct_zero[a], dct_zero[b], lst_v0)} "
            f"| {kappa(dct_five[a], dct_five[b], lst_v5)} "
            f"| {kappa(dct_zero[a], dct_zero[b], lst_common)} "
            f"| {kappa(dct_five[a], dct_five[b], lst_common)} |"
        )

    # subsets shared by a group of models: all models, and each leave-one-out group
    lst_groups: list[list[str]] = [lst_models] + [
        [x for x in lst_models if x != m] for m in lst_models
    ]
    lst_md += [
        "",
        "## Fixed subsets shared by a group of models",
        "",
        "Instances valid for every model of the group in both settings. "
        "Kappa range is over the pairs of the group; 'different' is the share of "
        "instances on which the models of the group do not all give the same label.",
        "",
        "| group | n | 0-shot kappa range | 5-shot kappa range "
        "| 0-shot different | 5-shot different |",
        "|---|---|---|---|---|---|",
    ]
    for lst_group in lst_groups:
        lst_ids: list[str] = sorted(valid(dct_zero, lst_group) & valid(dct_five, lst_group))
        lst_range: list[str] = []
        for dct_preds in (dct_zero, dct_five):
            lst_k: list[float] = [
                cohen_kappa_score(
                    [dct_preds[a][x] for x in lst_ids], [dct_preds[b][x] for x in lst_ids]
                )
                for a, b in itertools.combinations(lst_group, 2)
            ]
            lst_range.append(f"{min(lst_k):.2f}--{max(lst_k):.2f}")
        lst_md.append(
            f"| {', '.join(lst_group)} | {len(lst_ids):,} | {lst_range[0]} | {lst_range[1]} "
            f"| {share_different({m: dct_zero[m] for m in lst_group}, lst_ids)} "
            f"| {share_different({m: dct_five[m] for m in lst_group}, lst_ids)} |"
        )

    # share of instances with at least two distinct valid labels among the
    # models (the definition behind 46.4% and 74.5% in the manuscript)
    def share_distinct(dct_preds: dict[str, dict[str, int]], both: bool) -> str:
        k: int = 0
        for x in lst_all_ids:
            set_labels: set[int] = {
                dct_preds[m][x]
                for m in lst_models
                if dct_preds[m][x] != -1
                and (not both or (dct_zero[m][x] != -1 and dct_five[m][x] != -1))
            }
            k += len(set_labels) > 1
        return f"{k:,}/{len(lst_all_ids):,} = {100 * k / len(lst_all_ids):.1f}%"

    lst_md += [
        "",
        "## Share of instances with at least two distinct valid labels",
        "",
        "Denominator: all test instances. Fixed validity: for each instance, only "
        "the models with a valid output in both settings are compared.",
        "",
        "| | 0-shot | 5-shot |",
        "|---|---|---|",
        f"| setting-specific validity (as in the manuscript) "
        f"| {share_distinct(dct_zero, False)} | {share_distinct(dct_five, False)} |",
        f"| fixed validity | {share_distinct(dct_zero, True)} "
        f"| {share_distinct(dct_five, True)} |",
    ]
    p_out.write_text("\n".join(lst_md) + "\n", encoding="utf-8")
    print("\n".join(lst_md))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero", action="append", required=True, help="name=path (0-shot)")
    parser.add_argument("--five", action="append", required=True, help="name=path (5-shot)")
    parser.add_argument("--out", type=Path, default=ROOT_DIR / "kappa_common_subset.md")
    args = parser.parse_args()
    main(lst_zero=args.zero, lst_five=args.five, p_out=args.out)
