# Evaluation code for "Large-Scale Japanese Economic Datasets from the Economy Watchers Survey for Language Model Evaluation"

This repository contains the scripts used to run the model evaluations reported in the article and to map raw model outputs to task labels.

- `run_llm.py` — zero-/few-shot evaluation of the LLMs reported in the article (ChatGPT `gpt-3.5-turbo-0125`, GPT-4o `gpt-4o-2024-05-13`, Gemini 2.0 Flash, and llm-jp-3.1-13b-instruct4) on the domain, sentiment, and reason tasks, including the deterministic label-extraction (scoring) rule described in the Methods section.
- `run_encoder.py` — supervised fine-tuning and evaluation of the encoder baselines (FinBERT, DeBERTaV2) with the hyperparameter grid described in the article.
- `revision/` — scripts and demonstration sets for the additional analyses of the reason task reported in Supplementary Information Sections S3.1 and S4 (alternative 5-shot demonstration sets, label-wise diagnostics, and Cohen's kappa on common subsets). See [Additional analyses](#additional-analyses-of-the-reason-task).

The benchmark datasets are loaded from Hugging Face Hub: [`retarfi/economy-watchers-survey-evaluation`](https://doi.org/10.57967/hf/4514) (`ews_version="2024.05.2"`). The dataset uses a loading script, so the scripts pass `trust_remote_code=True` and require `datasets<4.0` (script-based datasets are not supported in datasets 4.x).

## Setup

Requires Python 3.9+.

```bash
pip install -r requirements.txt
```

Version bounds in `requirements.txt` keep the scripts compatible with the APIs used at experiment time: `datasets>=2.16,<4.0` (the `trust_remote_code` argument was added in Datasets 2.16, and script-based datasets were removed in 4.x) and `transformers>=4.23,<4.46` (the `evaluation_strategy` and `tokenizer=` Trainer arguments used here were changed in Transformers 4.46).

The dataset loader fetches the source data from Hugging Face Hub in many small requests, so constructing the full dataset without authentication can hit rate limits (HTTP 429). Log in with `huggingface-cli login` or set the `HF_TOKEN` environment variable before the first run.

API keys are read from environment variables (never hard-code keys):

```bash
export OPENAI_API_KEY=...   # for gpt-* models
export GEMINI_API_KEY=...   # for gemini-* models
```

llm-jp models run locally via `transformers` and require a GPU (bfloat16).

## Usage

Zero-shot and 5-shot LLM evaluation (one task × model × shot setting per run):

```bash
python run_llm.py domain --model gpt-4o-2024-05-13 --shots 0
python run_llm.py reason --model gemini-2.0-flash --shots 5
```

Repeated-run analysis (Supplementary Table S4) uses `--seed`:

```bash
python run_llm.py sentiment --model gpt-4o-2024-05-13 --shots 0 --seed 1
```

Note on seeds: the OpenAI API accepts a `seed` parameter and llm-jp runs use `transformers.set_seed`; the Gemini API does not expose a seed parameter, so Gemini reruns are independent runs without an explicit seed.

Raw model outputs are appended to `results/<task>_<model>_<shots>[_seed<seed>][_demo<set>].jsonl`; re-running the same configuration resumes from the existing file. After generation, the script prints macro F1 and accuracy.

Encoder fine-tuning with the hyperparameter grid from the article:

```bash
python run_encoder.py domain -m izumi-lab/bert-base-japanese-fin-additional --grid --bf16   # FinBERT
python run_encoder.py domain -m izumi-lab/deberta-v2-base-japanese --grid --bf16            # DeBERTaV2
```

Results and per-example predictions are written to `results/` (created automatically).

## Scoring rule

Each raw LLM output is mapped to a label deterministically (`map_prediction` in `run_llm.py`):

1. An output exactly equal to one canonical label is assigned that label.
2. Otherwise, if exactly one canonical label occurs as a substring of the output (e.g., a label accompanied by a short explanation), the output is mapped to that label.
3. For the reason task only, a still-unmapped output is assigned the canonical label with the smallest Levenshtein distance when that distance is at most 3 and the minimizing label is unique.
4. All outputs that remain unmapped after these steps (no canonical label, or multiple canonical labels that cannot be uniquely resolved) are invalid, receive no label, and are scored as incorrect in both accuracy and macro F1.

## Few-shot demonstrations

In the 5-shot setting, `run_llm.py` builds the demonstrations with a deterministic rule rather than by random sampling: the five slots are allocated to the labels in label order (`ceil((5 - i) / n_labels)` slots for the label at position `i`), and each slot is filled with the first training example longer than 30 characters found by scanning the training set in the order `train[0], train[-1], train[-2], ...`. The same demonstrations are used for every test instance and every model. For the reason task, this gives one example for each of the first five of the 12 labels.

To replace the default demonstrations with a pre-built set, pass `--demo-set`:

```bash
python run_llm.py reason --model llm-jp-3.1-13b-instruct4 --shots 5 --seed 1 \
    --demo-set revision/demo_sets/reason_A1.json
```

The set named `orig` reproduces the default demonstrations, and `run_llm.py` asserts that the resulting prompt is identical to the default one.

## Additional analyses of the reason task

The scripts in `revision/` reproduce the additional analyses in Supplementary Information Sections S3.1 and S4. Run them from the `revision/` directory; model outputs are read from `results/` as written by `run_llm.py`.

- `make_demo_sets.py` — builds the demonstration sets in `revision/demo_sets/`: `orig` (the default set), `A1`–`A3` (the labels of the default set, one randomly drawn training example per label), and `B1`–`B3` (five randomly drawn labels, one randomly drawn example per label). The random draws are seeded, so `python make_demo_sets.py reason` regenerates the committed files exactly.
- `run_demo_sets.sh` — runs llm-jp-3.1-13b on the reason task (5-shot, generation seed 1) once per demonstration set, e.g. `revision/run_demo_sets.sh 0 A1 A2 A3 B1 B2 B3` from the repository root (the first argument is the CUDA device).
- `score_demo_sets.py` — scores the llm-jp runs with the default set (three seeds, 0- and 5-shot) and with the alternative sets, including per-label F1 and the mean F1 over demonstrated and non-demonstrated labels (Supplementary Table S6). The label-extraction rule is the same as in `run_llm.py`.
- `label_diagnostics.py` — label-wise diagnostics of existing outputs without additional inference (Supplementary Tables S7 and S8):

  ```bash
  python label_diagnostics.py \
      --pred "ChatGPT 0-shot=../results/reason_gpt-35-turbo-0125_0.jsonl" \
      --pred "ChatGPT 5-shot=../results/reason_gpt-35-turbo-0125_5.jsonl"
  ```

- `kappa_common_subset.py` — pairwise Cohen's kappa on the instances valid for both models in a given setting (Supplementary Fig. S1) and on common subsets valid for both models in both settings (Supplementary Table S5), together with the proportion of instances for which the valid outputs of the models contain at least two distinct labels:

  ```bash
  python kappa_common_subset.py \
      --zero ChatGPT=../results/reason_gpt-35-turbo-0125_0.jsonl --zero GPT-4o=../results/reason_gpt-4o-2024-05-13_0.jsonl \
      --zero Gemini=../results/reason_gemini-20-flash_0.jsonl --zero llm-jp=../results/reason_llm-jp-31-13b-instruct4_0.jsonl \
      --five ChatGPT=../results/reason_gpt-35-turbo-0125_5.jsonl --five GPT-4o=../results/reason_gpt-4o-2024-05-13_5.jsonl \
      --five Gemini=../results/reason_gemini-20-flash_5.jsonl --five llm-jp=../results/reason_llm-jp-31-13b-instruct4_5.jsonl
  ```

  Instances missing from an output file are treated as invalid, and a warning is printed.

The versions of the API-based models used in the article were no longer available when these analyses were conducted, so the alternative demonstration sets were evaluated only with llm-jp-3.1-13b.

## Notes

- Generation settings follow the article: temperature 0.0 for GPT and Gemini models, temperature 0.2 (top-p 0.95, repetition penalty 1.05) for llm-jp.
- The main point estimates in the article were obtained from single runs; repeated-run variability is reported in the Supplementary Information.

## Reproducibility note on encoder training

`run_encoder.py` evaluates the validation and test sets at every epoch and reports the metrics logged at the epoch with the best validation macro F1. It does not checkpoint during training (`save_strategy="no"`), so the model saved at the end of training — and hence the per-example prediction file — corresponds to the end-of-training state. In the runs reported in the article these were consistent: for the DeBERTaV2 prediction files used in the article's bootstrap analysis, macro F1 recomputed from those files was 0.815441 for domain, 0.540968 for sentiment, and 0.553979 for reason, reproducing the values reported in the article after rounding. If you need the saved model itself to correspond to the best validation epoch, enable epoch checkpointing with `load_best_model_at_end=True` and `metric_for_best_model` in `TrainingArguments`.
