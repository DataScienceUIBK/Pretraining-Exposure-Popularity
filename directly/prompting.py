import json
import argparse
import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from tenacity import retry


def to_prompts(entities):
    prompt_template = """
You are a popularity estimator using your data and knowledge. Estimate the popularity of the entity below. Just a SINGLE integer number between 0 to 1000 WITHOUT ANY explanation.
There is no more information. If you do have enough information for the entity, show it using a low score without any explanation.

Entity: {0}

Score (0 to 1000): 
"""

    prompts = []
    for qid, entity in entities:
        message = [
            {
                "role": "user",
                "content": prompt_template.format(entity),
            },
        ]
        prompts.append((qid, message))

    return prompts


def valid(x: str) -> bool:
    return x.isdigit() and 0 <= int(x) <= 1000


@retry
def prompt(prompt_batch, tokenizer, model, max_new_tokens):
    output = dict()

    chat_texts = [
        tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        for _, message in prompt_batch
    ]

    enc = tokenizer(
        chat_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    gen = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.5,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.batch_decode(
        gen[:, enc["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    for tpl, out_text in zip(prompt_batch, decoded):
        text = out_text.strip()
        if valid(text):
            output[tpl[0]] = int(text)
        else:
            raise Exception()

    return output


def prompting(prompts, model_name, batch_size, max_new_tokens, time):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    ).eval()

    os.makedirs(f'./{time}', exist_ok=True)
    file_name = f'./{time}/prompt_directly_{model_name.split("/")[1]}.json'

    results = dict()
    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            results = json.load(f)

    with torch.inference_mode():
        for start in tqdm(range(0, len(prompts), batch_size), desc=f"Directly method using {model_name}"):
            end = start + batch_size
            if end < len(results):
                continue
            prompt_batch = prompts[start:end]
            output = prompt(prompt_batch, tokenizer, model, max_new_tokens)
            for qid in output:
                results[qid] = output[qid]
            if start % (batch_size * 10) == 0:
                with open(file_name, 'w') as file:
                    json.dump(results, file, indent=4)
        with open(file_name, 'w') as file:
            json.dump(results, file, indent=4)


def main(time):
    models = [('allenai/Olmo-3-7B-Instruct', 256), ('allenai/Olmo-3.1-32B-Instruct', 64)]
    with open('../dataset/dataset.json', 'r') as f:
        dataset = json.load(f)

    entities = []
    for _type in dataset:
        for category in ['low', 'high']:
            for item in dataset[_type][category].values():
                entities.append((item['qid'], item['enwiki_title']))

    prompts = to_prompts(entities)
    for _model in models:
        model_name, batch_size = _model
        prompting(prompts, model_name, batch_size, 1, time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', required=True, type=str)
    args = parser.parse_args()

    time = args.time

    main(time)
