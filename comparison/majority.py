import json


def main():
    for model in ['Olmo-3-7B-Instruct', 'Olmo-3.1-32B-Instruct']:
        dataset_json = dict()
        for time in ['1', '2', '3']:
            with open(f'./{time}/prompt_comparison_{model}.json', 'r') as f:
                comparison_json = json.load(f)
            for qid, label in comparison_json.items():
                if qid not in dataset_json:
                    dataset_json[qid] = []
                dataset_json[qid].append(label)

        for qid in dataset_json:
            dataset_json[qid] = 1 if dataset_json[qid].count(1) >= 2 else 2

        with open(f'./majority_{model}.json', 'w') as f:
            json.dump(dataset_json, f, indent=4)


if __name__ == '__main__':
    main()
