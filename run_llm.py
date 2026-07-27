import argparse
import json
import math
import os
from pathlib import Path
from typing import Optional, Union

import Levenshtein
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from google import genai
from google.genai import types
from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score
from tenacity import retry, stop_after_attempt, wait_fixed, wait_random_exponential
from tqdm import tqdm

BAR_FMT: str = (
    "{n_fmt}/{total_fmt}: {percentage:3.0f}%|{bar:10} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
)
ROOT_DIR: Path = Path(os.path.dirname(os.path.abspath(__file__)))


def get_openai_client() -> OpenAI:
    # Requires the OPENAI_API_KEY environment variable.
    return OpenAI()


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(20))
def openai_with_backoff(client: OpenAI, **kwargs):
    return client.chat.completions.create(**kwargs)


def get_gemini_client() -> genai.Client:
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


@retry(wait=wait_fixed(3), stop=stop_after_attempt(2000))
def gemini_with_backoff(client: genai.Client, **kwargs):
    return client.models.generate_content(**kwargs).text


def main(task: str, model: str, shots: int, seed: Optional[int], is_debug: bool) -> None:
    dsd: DatasetDict = load_dataset(
        "retarfi/economy-watchers-survey-evaluation",
        task,
        ews_version="2024.05.2",
        trust_remote_code=True,
    )
    test_ds: Dataset = dsd["test"]
    if is_debug:
        test_ds = test_ds.select(range(25))
    lst_labels: list[str] = dsd["train"].features["label"].names
    lst_label_id: list[int] = list(range(len(lst_labels)))

    # prompt and format
    def format_choices(choices: list[str]) -> str:
        return "答えは{}のいずれかでなければなりません。".format(
            "、".join(map(lambda x: f"「{x}」", choices))
        )

    prompt: str
    if task == "sentiment":
        lst_label_id = lst_label_id[::-1]
        choices: str = format_choices(lst_labels[::-1])
        prompt = "与えられたテキストのセンチメントをラベル付けしてください。"\
                 f"{choices}"\
                 "なお、◎は良い、○はやや良い、□はどちらとも言えない、▲はやや悪い、×は悪い、を表します。\n\n"
    elif task == "domain":
        choices: str = format_choices(lst_labels)
        prompt = "与えられたテキストがどの項目に関連しているかを分類してください。"\
                 f"{choices}"\
                 "\n\n"
    elif task == "reason":
        choices: str = format_choices(lst_labels)
        prompt = "与えられたテキストがどの点に着目しているかを分類してください。"\
                 f"{choices}"\
                 "\n\n"

    # in context learning
    fmt_context: str = "テキスト: {text}\n答え: {label}"
    lst_contexts: list[list[str]] = [[] for _ in range(len(lst_label_id))]
    for i in range(len(dsd["train"])):
        example: dict = dsd["train"][-i]
        priority_idx: int = lst_label_id.index(example["label"])
        if len(lst_contexts[priority_idx]) < math.ceil(
            (shots - priority_idx) / len(lst_label_id)
        ) and len(example["text"]) > 30:
            lst_contexts[priority_idx].append(
                fmt_context.format(
                    text=example["text"],
                    label=dsd["train"].features["label"].int2str(example["label"]),
                )
            )
        if sum(map(len, lst_contexts)) == shots:
            break
    for i in range(math.ceil(shots / len(lst_label_id))):
        for j in range(len(lst_label_id)):
            if i < len(lst_contexts[j]):
                prompt += lst_contexts[j][i] + "\n"

    debug: str = "debug_" if is_debug else ""
    str_seed: str = f"_seed{seed}" if seed is not None else ""
    p_output_jsonl: Path = (
        ROOT_DIR
        / "results" / f"{debug}{task}_{model.replace('.', '')}_{shots}{str_seed}.jsonl"
    )
    os.makedirs(p_output_jsonl.parent, exist_ok=True)
    df_loaded: pd.DataFrame
    if p_output_jsonl.exists():
        df_loaded = pd.read_json(p_output_jsonl, orient="records", lines=True)
        test_ds = test_ds.filter(
            lambda x: len(
                df_loaded[
                    (df_loaded["label"] == x["label"])
                    & (df_loaded["text"] == x["text"])
                ]
            )
            == 0
        )
    # load llm-jp model
    if model.startswith("llm-jp-"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        if seed:
            set_seed(seed)
        model_name: str = f"llm-jp/{model}"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tfmodel = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.bfloat16
        )

    msg_system: str = "あなたは優秀なAIアシスタントです。簡潔に回答してください。"
    for i in tqdm(range(len(test_ds)), bar_format=BAR_FMT):
        d: dict[str, Union[str, int]] = test_ds[i]
        msg_content: str = prompt + fmt_context.format(text=d["text"], label="")
        if model.startswith("gemini-"):
            client_gemini: genai.Client = get_gemini_client()
            response: str = gemini_with_backoff(
                client=client_gemini,
                model=model,
                contents=[
                    types.ModelContent(parts=[types.Part.from_text(text=msg_system)]),
                    types.UserContent(parts=[types.Part.from_text(text=msg_content)]),
                ],
                config=types.GenerateContentConfig(temperature=0.0),
            )
            d["output"] = response
        elif model.startswith("gpt-"):
            client_openai: OpenAI = get_openai_client()
            completion = openai_with_backoff(
                client=client_openai,
                model=model,
                messages=[
                    {"role": "system", "content": msg_system},
                    {"role": "user", "content": msg_content},
                ],
                temperature=0,
                seed=seed,
            )
            d["output"] = completion.choices[0].message.content
        elif model.startswith("llm-jp-"):
            messages: list[dict[str, str]] = [
                {"role": "system", "content": msg_system},
                {"role": "user", "content": msg_content},
            ]
            tokenized_input = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
            ).to(tfmodel.device)
            with torch.no_grad():
                output = tfmodel.generate(
                    tokenized_input,
                    max_new_tokens=256,
                    temperature=0.2,
                    top_p=0.95,
                    repetition_penalty=1.05,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )[0]
            d["output"] = tokenizer.decode(
                output[tokenized_input.shape[1]:], skip_special_tokens=True
            )
        else:
            raise ValueError(f"Unknown model: {model}")
        with open(p_output_jsonl, "a", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)
            f.write("\n")

    # Scoring: map each raw output to a task label (or -1 for invalid).
    # 1. An output exactly equal to one canonical label is assigned that label.
    # 2. Otherwise, if exactly one canonical label occurs as a substring of the
    #    output, the output is mapped to that label.
    # 3. For the reason task, a still-unmapped output is assigned the canonical
    #    label with the smallest Levenshtein distance when that distance is at
    #    most 3 and the minimizing label is unique.
    # 4. All remaining outputs are invalid (-1) and scored as incorrect.
    def map_prediction(text: str) -> int:
        label_output: str
        if text in lst_labels:
            label_output = text
        else:
            lst_cand: list[str] = [x for x in lst_labels if x in text]
            if len(lst_cand) == 1:
                label_output = lst_cand[0]
            elif task == "reason":
                lst_dist: list[int] = [
                    Levenshtein.distance(text, x) for x in lst_labels
                ]
                if min(lst_dist) <= 3 and lst_dist.count(min(lst_dist)) == 1:
                    label_output = lst_labels[lst_dist.index(min(lst_dist))]
                else:
                    return -1
            elif text in ("特に気になることはない。", "よく分からない。", "分類できません。"):
                return -1
            else:
                return -1
        return dsd["train"].features["label"].str2int(label_output)

    df_loaded = pd.read_json(p_output_jsonl, orient="records", lines=True)
    df_loaded["label_output"] = df_loaded["output"].map(map_prediction)
    f1: float = f1_score(
        df_loaded["label"].values,
        df_loaded["label_output"].values,
        average="macro",
        labels=list(range(len(lst_labels))),
    )
    acc: float = accuracy_score(
        df_loaded["label"].values, df_loaded["label_output"].values
    )
    print(f"F1: {f1:.5f}, Accuracy: {acc:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["sentiment", "domain", "reason"])
    parser.add_argument(
        "--model", choices=[
            "gpt-3.5-turbo-0125",
            "gpt-4o-2024-05-13",
            "gemini-2.0-flash",
            "llm-jp-3.1-13b-instruct4",
        ], required=True
    )
    parser.add_argument(
        "--shots", help="Number of shots", required=True, type=int
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    main(
        task=args.task,
        model=args.model,
        shots=args.shots,
        seed=args.seed,
        is_debug=args.debug,
    )
