import json


def main():
    with open('../dataset/dataset.json', 'r') as f:
        dataset = json.load(f)
    dataset_dict = dict()

    for k in dataset:
        for category in dataset[k]:
            for item in dataset[k][category].values():
                qid = item['qid']
                dataset_dict[qid] = item

    for model in ['Olmo-3-7B-Instruct', 'Olmo-3.1-32B-Instruct']:
        for time in ['1', '2', '3']:
            with open(f'./{time}/prompt_directly_{model}.json', 'r') as f:
                prompt_directly = json.load(f)
            for qid in prompt_directly:
                popularity = prompt_directly[qid]
                if f'popularity_{model}' not in dataset_dict[qid]:
                    dataset_dict[qid][f'popularity_{model}'] = 0
                dataset_dict[qid][f'popularity_{model}'] += popularity

    for k in dataset:
        for category in dataset[k]:
            for item in dataset[k][category].values():
                for model in ['Olmo-3-7B-Instruct', 'Olmo-3.1-32B-Instruct']:
                    item[f'popularity_{model}'] = round(item[f'popularity_{model}'] / 3.0, 3)

    with open('./dataset_directly.json', 'w') as f:
        json.dump(dataset, f, indent=4)


if __name__ == "__main__":
    main()
