# Evaluation code for "Large-Scale Japanese Economic Datasets from the Economy Watchers Survey for Language Model Evaluation"

This repository contains the scripts used to run the model evaluations reported in the article and to map raw model outputs to task labels.

- `run_llm.py` — zero-/few-shot evaluation of the LLMs reported in the article (ChatGPT `gpt-3.5-turbo-0125`, GPT-4o `gpt-4o-2024-05-13`, Gemini 2.0 Flash, and llm-jp-3.1-13b-instruct4) on the domain, sentiment, and reason tasks, including the deterministic label-extraction (scoring) rule described in the Methods section.
- `run_encoder.py` — supervised fine-tuning and evaluation of the encoder baselines (e.g., FinBERT, DeBERTaV2) with the hyperparameter grid described in the article.

The benchmark datasets are loaded from Hugging Face Hub: [`retarfi/economy-watchers-survey-evaluation`](https://doi.org/10.57967/hf/4514) (`ews_version="2024.05.2"`).

## Setup

```bash
pip install -r requirements.txt
```

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

Repeated-run analysis (Supplementary Table S3) uses `--seed`:

```bash
python run_llm.py sentiment --model gpt-4o-2024-05-13 --shots 0 --seed 1
```

Raw model outputs are appended to `results/<task>_<model>_<shots>[_seed<seed>].jsonl`; re-running the same configuration resumes from the existing file. After generation, the script prints macro F1 and accuracy.

Encoder fine-tuning:

```bash
python run_encoder.py domain -m <model_name_or_path> --grid --bf16
```

Results and per-example predictions are written to `results/`.

## Scoring rule

Each raw LLM output is mapped to a label deterministically (`map_prediction` in `run_llm.py`):

1. An output exactly equal to one canonical label is assigned that label.
2. Otherwise, if exactly one canonical label occurs as a substring of the output (e.g., a label accompanied by a short explanation), the output is mapped to that label.
3. For the reason task only, a still-unmapped output is assigned the canonical label with the smallest Levenshtein distance when that distance is at most 3 and the minimizing label is unique.
4. All remaining outputs (no canonical label, or two or more canonical labels) are invalid, receive no label, and are scored as incorrect in both accuracy and macro F1.

## Notes

- Generation settings follow the article: temperature 0.0 for GPT and Gemini models, temperature 0.2 (top-p 0.95, repetition penalty 1.05) for llm-jp.
- The main point estimates in the article were obtained from single runs; seed-controlled reruns are reported in the Supplementary Information.
