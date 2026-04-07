import json
import argparse
import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from tenacity import retry
from itertools import combinations


def to_prompts(entities):
    prompt_template = """You are a classifier that outputs a single digit. If you do have enough information for the entity, consider it as low popular.

Valid outputs:
- 1 → Entity 1 is more popular
- 2 → Entity 2 is more popular

Constraints:
- Output exactly one digit.
- No text, no punctuation, no spaces.
- If uncertain, still choose one.

Entities:
1) {0}
2) {1}

Digit (1 or 2):
"""

    person = entities[:400]
    loc_gpe = entities[400:800]
    org_fac = entities[800:1200]
    work_of_art = entities[1200:1600]
    products = entities[1600:2000]

    prompts = []
    for _type in [person, loc_gpe, org_fac, work_of_art, products]:
        for (q_1, e_1), (q_2, e_2) in list(combinations(_type, 2)):
            message = [
                {
                    "role": "user",
                    "content": prompt_template.format(e_1, e_2),
                },
            ]
            prompts.append((q_1, q_2, message))

    return prompts


def valid(x: str) -> bool:
    return x.isdigit() and 1 <= int(x) <= 2


@retry
def prompt(prompt_batch, tokenizer, model, max_new_tokens):
    output = dict()

    chat_texts = [
        tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        for _, _, message in prompt_batch
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
            output[f'{tpl[0]}-{tpl[1]}'] = int(text)
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
    file_name = f'./{time}/prompt_comparison_{model_name.split("/")[1]}.json'

    results = dict()
    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            results = json.load(f)

    with torch.inference_mode():
        for start in tqdm(range(0, len(prompts), batch_size), desc=f"Comparison method using {model_name}"):
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
