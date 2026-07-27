import argparse
import itertools
import json
import logging
import os
from typing import Dict, List, NamedTuple, Tuple
import warnings
import unicodedata
from packaging import version

import evaluate
import numpy as np
import torch
import transformers
from datasets import DatasetDict, load_dataset
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    PretrainedConfig,
    PreTrainedTokenizerBase,
)


log_level: str = "error"
assert log_level in transformers.logging.log_levels.keys()
# to ignore UndefinedMetricWarning
warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

BAR_FMT: str = (
    "{n_fmt}/{total_fmt}: {percentage:3.0f}%|{bar:10} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
)


def on_train_begin(self, args, state, control, **kwargs):
    if state.is_world_process_zero:
        self.training_bar = tqdm(total=state.max_steps, bar_format=BAR_FMT)
    self.current_step = 0


transformers.ProgressCallback.on_train_begin = on_train_begin


class HyperParams(NamedTuple):
    max_epochs: int = 2
    real_batch_size: int = 32
    lr: float = 1e-5


class GridHyperParams(NamedTuple):
    max_epochs: Tuple[int, ...] = (5,)
    real_batch_size: Tuple[int, ...] = (32, 64)
    lr: Tuple[float, ...] = (1e-5, 2e-5, 5e-5, 1e-4)


def set_seed(seed: int):
    np.random.seed(seed)
    np.random.RandomState(seed)
    transformers.set_seed(seed)
    transformers.trainer_utils.set_seed(seed=seed)
    torch.manual_seed(seed)


def compute_metrics(p: transformers.EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    metrics = evaluate.load("f1")
    result = metrics.compute(predictions=preds, references=p.label_ids, average="macro")
    acc = evaluate.load("accuracy")
    result.update(acc.compute(predictions=preds, references=p.label_ids))
    # if len(result) > 1:
    #     result["combined_score"] = np.mean(list(result.values())).item()
    return result


def train(
    model_name_or_path: str,
    tokenizer: PreTrainedTokenizerBase,
    task: str,
    dsd: DatasetDict,
    seed: int,
    output_dir: str,
    do_grid: bool = False,
    bf16: bool = False,
) -> Tuple[Dict[str, float], np.ndarray]:
    transformers.logging.set_verbosity(transformers.logging.log_levels[log_level])

    lst_eval_metrics = ["eval_f1", "eval_accuracy"]
    assert version.parse("4.23.0") <= version.parse(transformers.__version__)
    lst_eval_metrics = list(
        map(lambda x: x.replace("eval_", "eval_eval_"), lst_eval_metrics)
    ) + list(map(lambda x: x.replace("eval_", "eval_test_"), lst_eval_metrics))
    # Iterate with max_epochs, real_batch_size, lr
    if do_grid:
        ghp = GridHyperParams()
        product = itertools.product(ghp.max_epochs, ghp.real_batch_size, ghp.lr)
    else:
        hp = HyperParams()
        product = itertools.product([hp.max_epochs], [hp.real_batch_size], [hp.lr])
    config: PretrainedConfig = AutoConfig.from_pretrained(model_name_or_path)
    if task == "sentiment":
        config.num_labels = 5
    elif task == "domain":
        config.num_labels = 3
    elif task == "reason":
        config.num_labels = 12
    else:
        config.num_labels = 0
    # [key score, metrics(dict)]
    lst_grid_results: List[Dict[str, float]] = []
    best_trainer: transformers.Trainer = None
    best_score: float = -1.0
    for max_epochs, real_batch_size, lr in product:
        training_args = transformers.TrainingArguments(
            output_dir="./outputs/",
            do_train=True,
            do_eval=True,
            do_predict=False,  # "test" in raw_datasets,
            evaluation_strategy="epoch",
            per_device_train_batch_size=real_batch_size,
            per_device_eval_batch_size=real_batch_size,
            learning_rate=lr,
            weight_decay=0.01,
            num_train_epochs=max_epochs,
            warmup_ratio=0.1,
            logging_strategy="no",
            save_strategy="no",
            seed=seed,
            report_to="none",
            label_names=None,
            disable_tqdm=False,
            # data_seed=params["seed"],
            bf16=bf16,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )

        trainer = transformers.Trainer(
            model=model,
            args=training_args,
            train_dataset=dsd["train"],
            eval_dataset={"eval": dsd["validation"], "test": dsd["test"]},
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=transformers.default_data_collator,
        )

        # Training
        transformers.logging.set_verbosity(transformers.logging.log_levels[log_level])
        train_result = trainer.train()
        # print(trainer.state.log_history)

        lst_epochs_result: List[Dict[str, float]] = list(
            filter(
                lambda d: len(set(lst_eval_metrics) & set(d.keys())) > 0,
                trainer.state.log_history,
            )
        )
        lst_tmp: List[Dict[str, float]] = []
        for i in range(1, max_epochs + 1):
            d_tmp = {}
            for d in filter(lambda item: item["epoch"] == i, lst_epochs_result):
                d_tmp.update(d)
            lst_tmp.append(d_tmp)
        dct_result = max(enumerate(lst_tmp), key=lambda x: x[1][lst_eval_metrics[0]])[1]
        dct_result = dict(
            filter(lambda item: item[0] in lst_eval_metrics, dct_result.items())
        )
        score = dct_result[lst_eval_metrics[0]]
        if score > best_score:
            best_score = score
            best_trainer = trainer
        lst_grid_results.append(dct_result)
    # get best eval result
    best_result: Dict[str, float] = sorted(
        lst_grid_results, key=lambda x: x[lst_eval_metrics[0]], reverse=True
    )[0]
    best_result = dict(
        [
            (k.replace("eval_eval_", "eval_").replace("eval_test_", "test_"), v)
            for k, v in best_result.items()
        ]
    )
    best_trainer.save_model(output_dir)
    pred_output = best_trainer.predict(dsd["test"])
    pred_labels: np.ndarray = np.argmax(pred_output.predictions, axis=1)
    return best_result, pred_labels


if __name__ == "__main__":
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description="BERTを学習")
    parser.add_argument("task", choices=["sentiment", "domain", "reason"])
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="model name or path of model directory",
    )
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--tokenizer_name_or_path", default="")
    args: argparse.Namespace = parser.parse_args()
    if args.tokenizer_name_or_path == "":
        args.tokenizer_name_or_path = args.model

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tokenizer: PreTrainedTokenizerBase = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path
    )

    dsd: DatasetDict = load_dataset(
        "retarfi/economy-watchers-survey-evaluation",
        args.task,
        ews_version="2024.05.2",
        trust_remote_code=True,
    )

    def normalize(example: dict) -> dict:
        example["text"] = unicodedata.normalize("NFKC", example["text"])
        return example

    dsd = dsd.map(normalize)

    # Search max length
    lst_length: list[int] = [len(tokenizer.encode(x)) for x in dsd["train"]["text"]]
    max_length: int = min(max(lst_length), 512)
    print(f"Max length: {max_length}")

    dsd = dsd.map(
        lambda example: tokenizer(
            example["text"],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
    )

    seed: int = 0
    set_seed(seed)
    model_basename: str = os.path.basename(args.model)
    model_output_dir: str = f"outputs/{args.task}-{model_basename}"
    result, pred_labels = train(
        model_name_or_path=args.model,
        tokenizer=tokenizer,
        task=args.task,
        dsd=dsd,
        seed=seed,
        output_dir=model_output_dir,
        do_grid=args.grid,
        bf16=args.bf16,
    )
    print(result)
    os.makedirs("results", exist_ok=True)
    with open(f"results/{args.task}-{model_basename}.json", "w") as f:
        json.dump(result, f, indent=2)
    test_dataset = dsd["test"]
    id_column = test_dataset["id"] if "id" in test_dataset.column_names else list(range(len(test_dataset)))
    with open(f"results/{args.task}-{model_basename}_predictions.jsonl", "w") as f:
        for id_, gold, pred in zip(id_column, test_dataset["label"], pred_labels.tolist()):
            f.write(json.dumps({"id": id_, "gold_label": gold, "output_label": pred}) + "\n")
