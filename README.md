<a href=""><img src="https://img.shields.io/static/v1?label=Paper&message=ACM%20SIGIR&color=green&logo=arXiv"></a> <a href="https://opensource.org/license/mit"><img src="https://img.shields.io/static/v1?label=License&message=MIT&color=red"></a>

# Pretraining Exposure Explains Popularity Judgments in Large Language Models

> *Why do LLMs think something is popular? Mostly because of how often they saw it during training.*

## 📌 Overview

Large Language Models (LLMs) exhibit systematic preferences for well-known entities, often referred to as *popularity bias*. In this work, we show that these preferences are primarily driven by **pretraining exposure**—the frequency with which entities appear in the training corpus—rather than external signals such as real-world popularity.

Using fully observable pretraining data, we conduct the first large-scale analysis that directly links:

* **Pretraining exposure**
* **Wikipedia popularity**
* **LLM-generated popularity judgments**

## 🗂 Repository Structure

```text
.
├── comparison/        # Pairwise comparison prompting & aggregation
│   ├── majority.py    # Majority voting over repeated pairwise judgments
│   ├── merge.py       # Merge pairwise outputs
│   ├── prompting.py   # Pairwise prompting pipeline
│   └── run.sh         # Run pairwise experiments
│
├── dataset/           # Full dataset with all signals
│   └── dataset.json   # Entities, metadata, exposure, pageviews, model outputs
│
├── directly/          # Direct popularity estimation pipeline
│   ├── merge.py       # Merge repeated outputs
│   ├── prompting.py   # Direct estimation prompting
│   └── run.sh         # Run direct estimation
│
└── experiments/       # Evaluation and analysis
    ├── correlation.py
    └── pairwise_accuracy.py
```

## 📊 Dataset

We release a **fully integrated, analysis-ready dataset** that consolidates all signals used in the paper into a single resource.

The dataset contains **2,000 entities** spanning five semantic types:
**Person, Location, Organization, Art, and Product**.

Each entity is enriched with multiple complementary signals:

* structured knowledge (Wikidata metadata and entity type)
* linguistic coverage (validated aliases)
* real-world popularity (Wikipedia pageviews)
* training-derived statistics (pretraining exposure counts)
* model-based estimates:

  * direct scalar popularity predictions
  * pairwise comparison outcomes

Unlike typical benchmarks, this dataset is **self-contained**:
it includes both *raw inputs and computed signals*, allowing researchers to:

* reproduce results without rerunning expensive pipelines
* directly study relationships between signals
* extend the analysis to new models or ranking methods

## 🚀 Getting Started

### Setup

```bash
git clone https://github.com/DataScienceUIBK/Pretraining-Exposure-Popularity.git
cd Pretraining-Exposure-Popularity
pip install -r requirements.txt
```

## 🤖 Models

We evaluate two fully open large language models released by the Allen Institute for AI:

* **OLMo-3-7B-Instruct**
  https://huggingface.co/allenai/Olmo-3-7B-Instruct

* **OLMo-3.1-32B-Instruct**
  https://huggingface.co/allenai/Olmo-3.1-32B-Instruct

These models are fully open and provide access to their training data, enabling direct measurement of **pretraining exposure**.

## 🧪 Running Experiments

### 🔹 Direct Estimation

```bash
cd directly
bash run.sh
```

### 🔹 Pairwise Comparison

```bash
cd comparison
bash run.sh
```

### 🔹 Analysis

```bash
cd experiments
python correlation.py
python pairwise_accuracy.py
```

## 🔁 Reproducibility

This repository is designed for **full reproducibility** of the reported results.

### What is included

* Complete dataset with all signals (exposure, Wikipedia, LLM outputs)
* Prompting pipelines for direct and pairwise estimation
* Aggregation scripts (majority voting, merging)
* Evaluation scripts (correlation and accuracy)

### Requirements

* Python **3.11+**
* Dependencies listed in `requirements.txt`
* Access to **GPT models (required for parts of the pipeline)**
* GPU recommended for efficient inference

### Notes

* Results may vary slightly due to LLM stochasticity
* Multiple runs and aggregation are used for stability
* All evaluated models are publicly available

## 📊 Key Findings

* Pretraining exposure strongly correlates with real-world popularity
* LLM judgments align more with **exposure than Wikipedia signals**
* Pairwise comparison yields the most reliable estimates
* Effects persist in long-tail entities

## 📄 Citation

```bibtex
```

## 📜 License

MIT License — see `LICENSE`.
