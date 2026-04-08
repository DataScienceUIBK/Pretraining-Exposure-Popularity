import json
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def extract_wikipedia(_type, _categories):
    qid_list = []
    with open('../dataset/dataset.json', 'r') as f:
        dataset = json.load(f)
    for _category in _categories:
        for qid in dataset[_type][_category]:
            item = dataset[_type][_category][qid]
            corpus_popularity = item['popularity']['corpus']
            wikipedia_popularity = item['popularity']['wikipedia']
            qid_list.append((qid, corpus_popularity, wikipedia_popularity))
    corpus_sorted = sorted(qid_list, key=lambda x: x[1])
    corpus_pops = [pop for _, pop, _ in corpus_sorted]
    wikipedia_pops = [pop for _, _, pop in corpus_sorted]
    return corpus_pops, wikipedia_pops


def extract_directly(_model, _type, _categories):
    qid_list = []
    with open('../dataset/dataset.json', 'r') as f:
        dataset = json.load(f)
    for _category in _categories:
        for qid in dataset[_type][_category]:
            item = dataset[_type][_category][qid]
            corpus_popularity = item['popularity']['corpus']
            directly_popularity = item['popularity']['directly'][_model]
            qid_list.append((qid, corpus_popularity, directly_popularity))
    corpus_sorted = sorted(qid_list, key=lambda x: x[1])
    corpus_pops = [pop for _, pop, _ in corpus_sorted]
    directly_pops = [pop for _, _, pop in corpus_sorted]
    return corpus_pops, directly_pops


def extract_comparison(_model, _type, _categories):
    qid_list = []
    with open('../dataset/dataset.json', 'r') as f:
        dataset = json.load(f)
    for _category in _categories:
        for qid in dataset[_type][_category]:
            item = dataset[_type][_category][qid]
            corpus_popularity = item['popularity']['corpus']
            if len(_categories) == 1:
                comparison_popularity = item['popularity']['comparison'][_model]['category']
            else:
                comparison_popularity = item['popularity']['comparison'][_model]['full']
            qid_list.append((qid, corpus_popularity, comparison_popularity))
    corpus_sorted = sorted(qid_list, key=lambda x: x[1])
    corpus_pops = [pop for _, pop, _ in corpus_sorted]
    comparison_pops = [pop for _, _, pop in corpus_sorted]
    return corpus_pops, comparison_pops


def build_olmo_table(data):
    """
    Convert nested Olmo results dict into a multi-index pandas table.

    Parameters
    ----------
    data : dict
        Nested dictionary with structure:
        data[model][entity][difficulty][method] -> score

    Returns
    -------
    pd.DataFrame
        Formatted table matching the paper layout
    """

    # fixed display order (matches your figure)
    models = [('7b', 'Olmo 7B'), ('32b', 'Olmo 32B')]
    entities = [
        ('PERSON', 'Person'),
        ('LOC_GPE', 'Location'),
        ('ORG_FAC', 'Organization'),
        ('WORK_OF_ART', 'Art'),
        ('PRODUCT', 'Product')
    ]
    difficulties = ['Full', 'Low', 'High']
    methods = ['Wikipedia', 'Directly', 'Comparison']

    # Multi-level columns
    columns = pd.MultiIndex.from_product(
        [[m[1] for m in models],
         [e[1] for e in entities]]
    )

    rows = []
    index = []

    for diff in difficulties:
        for method in methods:
            index.append((diff, method))
            row = []
            for model_key, _ in models:
                for entity_key, _ in entities:
                    row.append(data[model_key][entity_key][diff][method])
            rows.append(row)

    df = pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(index, names=['Split', 'Method']),
        columns=columns
    )

    df.to_excel("correlation_results.xlsx")


def main():
    final_results = dict()
    for method in ['Wikipedia', 'Directly', 'Comparison']:
        for model in ['7b', '32b']:
            if model not in final_results:
                final_results[model] = dict()
            for _type in ['PERSON', 'LOC_GPE', 'ORG_FAC', 'WORK_OF_ART', 'PRODUCT']:
                if _type not in final_results[model]:
                    final_results[model][_type] = dict()
                for _categories in [['low'], ['high'], ['low', 'high']]:
                    if len(_categories) == 1:
                        __cat = _categories[0].capitalize()
                    else:
                        __cat = 'Full'
                    if __cat not in final_results[model][_type]:
                        final_results[model][_type][__cat] = dict()

                    if method == 'Wikipedia':
                        corpus_pops, wikipedia_pops = extract_wikipedia(_type, _categories)
                        pearson_corr, _ = pearsonr(corpus_pops, wikipedia_pops)
                        spearman_corr, _ = spearmanr(corpus_pops, wikipedia_pops)
                    elif method == 'Directly':
                        corpus_pops, directly_pops = extract_directly(model, _type, _categories)
                        pearson_corr, _ = pearsonr(corpus_pops, directly_pops)
                        spearman_corr, _ = spearmanr(corpus_pops, directly_pops)
                    elif method == 'Comparison':
                        corpus_pops, comparison_pops = extract_comparison(model, _type, _categories)
                        pearson_corr, _ = pearsonr(corpus_pops, comparison_pops)
                        spearman_corr, _ = spearmanr(corpus_pops, comparison_pops)
                    final_results[model][_type][__cat][method] = round(spearman_corr, 3)
    build_olmo_table(final_results)



if __name__ == '__main__':
    main()
