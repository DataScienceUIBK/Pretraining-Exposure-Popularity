import json
from collections import defaultdict
from itertools import product


def dict_to_wins(pairs_dict):
    """
    pairs_dict: {"a-b": 1 or 2}
      1 => a better than b
      2 => b better than a

    returns:
      wins[i][j] = count of i beating j
      items = sorted list of all ids
    """
    wins = defaultdict(lambda: defaultdict(int))
    items = set()

    for k, v in pairs_dict.items():
        a, b = k.split("-")
        items.update([a, b])

        if v == 1:
            wins[a][b] += 1
        elif v == 2:
            wins[b][a] += 1
        else:
            raise ValueError(f"Value must be 1 or 2, got {v} for {k}")

    return wins, sorted(items)


def fit_bradley_terry_mm(wins, items, max_iter=10000, tol=1e-12, eps=1e-12):
    """
    MM / iterative scaling for Bradley–Terry.
    Returns positive abilities p[i] (sum to 1).
    """
    p = {i: 1.0 for i in items}

    for _ in range(max_iter):
        p_old = p.copy()

        for i in items:
            w_i = sum(wins[i].values())

            denom = 0.0
            for j in items:
                if j == i:
                    continue
                n_ij = wins[i][j] + wins[j][i]  # total comps between i and j
                if n_ij > 0:
                    denom += n_ij / (p[i] + p[j] + eps)

            if denom > 0:
                p[i] = max(eps, w_i / denom)
            # else: i never compared -> keep as is

        # normalize (scale is arbitrary)
        s = sum(p.values())
        for i in items:
            p[i] /= (s + eps)

        if max(abs(p[i] - p_old[i]) for i in items) < tol:
            break

    return p


def extract_related_items(_model, _type, _categories):
    qid_list = []
    with open('../dataset/dataset.json', 'r') as f:
        dataset = json.load(f)
    for _category in _categories:
        for item in dataset[_type][_category].values():
            qid = item['qid']
            qid_list.append(qid)

    result_items = dict()
    with open(f'./majority_{_model}.json', 'r') as f:
        majority_json = json.load(f)
    for q_1, q_2 in list(product(qid_list, qid_list)):
        _id = f'{q_1}-{q_2}'
        result_items[_id] = majority_json[_id]
    return result_items


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
        for _type in ['PERSON', 'LOC_GPE', 'ORG_FAC', 'WORK_OF_ART', 'PRODUCT']:
            for _categories in [['low'], ['high'], ['low', 'high']]:
                print(model, _type, _categories)
                pairs = extract_related_items(model, _type, _categories)
                wins, items = dict_to_wins(pairs)
                abilities = fit_bradley_terry_mm(wins, items)
                for q_id in abilities.keys():
                    _cat_title = '_'.join(_categories)
                    dataset_dict[q_id][f'popularity_{model}_{_type}_{_cat_title}'] = abilities[q_id]
                with open('./dataset_comparison.json', 'w') as f:
                    json.dump(dataset, f, indent=4)


if __name__ == "__main__":
    main()
