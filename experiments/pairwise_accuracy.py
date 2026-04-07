import json
import matplotlib.pyplot as plt
from itertools import product


def extract_related_items(_model, _type, _categories):
    qid_list = []
    with open('../dataset/dataset.json', 'r') as f:
        dataset = json.load(f)
    for _category in _categories:
        for qid in dataset[_type][_category]:
            item = dataset[_type][_category][qid]
            corpus_popularity = item['popularity']['corpus']
            qid_list.append((qid, corpus_popularity))
    qid_list = sorted(qid_list, key=lambda x: x[1])
    qid_list = [qid for qid, _ in qid_list]

    result_items = dict()
    with open(f'../comparison/majority_{_model}.json', 'r') as f:
        majority_json = json.load(f)
    for q_1, q_2 in list(product(qid_list, qid_list)):
        _id = f'{q_1}-{q_2}'
        result_items[_id] = majority_json[_id]
    return qid_list, result_items


def accuracy(sorted_qids, pairs):
    total = 0
    for k, v in pairs.items():
        q1, q2 = k.split('-')
        if sorted_qids.index(q1) < sorted_qids.index(q2) and pairs[k] == 2:
            total += 1
        elif sorted_qids.index(q1) > sorted_qids.index(q2) and pairs[k] == 1:
            total += 1
    pairs_total = len(pairs) - len(sorted_qids)
    return round(total / pairs_total, 2)


def draw_chart(data):
    models = ["Olmo-3-7B-Instruct", "Olmo-3.1-32B-Instruct"]
    models_dict = {"Olmo-3-7B-Instruct": "OLMo 7B", "Olmo-3.1-32B-Instruct": "OLMo 32B"}
    types = ["Person", "Loc_gpe", "Org_fac", "Work_of_art", "Product"]
    splits = ["low", "high", "full"]

    split_labels = {
        "low": "Sparse Entities",
        "high": "Popular Entities",
        "full": "All Entities",
    }

    type_labels = {
        "Person": "Person",
        "Loc_gpe": "Location",
        "Org_fac": "Organization",
        "Work_of_art": "Art",
        "Product": "Product",
    }

    x = range(len(types))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharey=True)

    handles = []
    labels = []

    for ax, model in zip(axes, models):
        for i, split in enumerate(splits):
            values = [data[f"{model}-{t}-{split}"] for t in types]
            bars = ax.bar(
                [p + i * width for p in x],
                values,
                width,
                label=split_labels[split],
            )

            # collect legend handles only once
            if model == models[0]:
                handles.append(bars[0])
                labels.append(split_labels[split])

        ax.set_xticks([p + width for p in x])
        ax.set_xticklabels([type_labels[t] for t in types], rotation=30)
        ax.set_title(models_dict[model])
        ax.set_ylim(0, 1)

    axes[0].set_ylabel("Accuracy")

    # single legend on top, horizontal
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.55, 1.02),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("./pairwise_accuracy.pdf", dpi=300)
    plt.show()


def main():
    results = dict()
    for model in ['Olmo-3-7B-Instruct', 'Olmo-3.1-32B-Instruct']:
        for _type in ['PERSON', 'LOC_GPE', 'ORG_FAC', 'WORK_OF_ART', 'PRODUCT']:
            for _categories in [['low'], ['high'], ['low', 'high']]:
                sorted_qids, pairs = extract_related_items(model, _type, _categories)
                acc = accuracy(sorted_qids, pairs)
                key = f'{model}-{_type.capitalize()}'
                if len(_categories) == 1 and _categories[0] == 'low':
                    key = f'{key}-low'
                elif len(_categories) == 1 and _categories[0] == 'high':
                    key = f'{key}-high'
                elif len(_categories) == 2:
                    key = f'{key}-full'
                results[key] = acc
    print(results)
    draw_chart(results)


if __name__ == '__main__':
    main()
