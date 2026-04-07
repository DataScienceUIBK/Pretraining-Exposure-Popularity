# Pretraining Exposure Explains Popularity Judgments in Large Language Models

> *Why do LLMs think something is popular? Mostly because of how often they saw it during training.*

[![Conference](https://img.shields.io/badge/SIGIR-2026-blue)]()
[![Paper](https://img.shields.io/badge/Paper-SIGIR%202026-green)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official repository for the **SIGIR 2026** paper:
**“Pretraining Exposure Explains Popularity Judgments in Large Language Models”**

🔗 https://github.com/DataScienceUIBK/Pretraining-exposure-popularity

---

## 📌 Overview

Large Language Models (LLMs) exhibit systematic preferences for well-known entities, often referred to as *popularity bias*. In this work, we show that these preferences are primarily driven by **pretraining exposure**—the frequency with which entities appear in the training corpus—rather than external signals such as real-world popularity.

Using fully observable pretraining data, we conduct the first large-scale analysis that directly links:

* **Pretraining exposure**
* **Wikipedia popularity**
* **LLM-generated popularity judgments**

---

## 🗂 Repository Structure

```id="jv5p97"
.
├── comparison/        # Pairwise comparison prompting & aggregation
│   ├── majority.py
│   ├── merge.py
│   ├── prompting.py
│   └── run.sh
│
├── dataset/           # Full dataset with all signals
│   └── dataset.json
│
├── directly/          # Direct popularity estimation pipeline
│   ├── merge.py
│   ├── prompting.py
│   └── run.sh
│
└── experiments/       # Analysis and evaluation scripts
    ├── correlation.py
    └── pairwise_accuracy.py
```

---

## 📊 Dataset

We provide a **fully processed dataset** containing **2,000 entities** across five types:

* Person
* Location
* Organization
* Art
* Product

Each entity includes:

* Wikidata metadata
* Entity type
* Validated aliases
* Wikipedia pageviews
* Pretraining exposure scores
* LLM popularity signals:

  * Direct estimation
  * Pairwise comparison

👉 The dataset is **self-contained** and enables full reproducibility of the paper results.

---

## 🚀 Getting Started

### Setup

```bash id="0199ar"
git clone https://github.com/DataScienceUIBK/Pretraining-exposure-popularity.git
cd Pretraining-exposure-popularity
pip install -r requirements.txt
```

---

## 🤖 Models

We use **fully open models** available on Hugging Face (e.g., OLMo variants).
No API keys or proprietary access are required.

---

## 🧪 Running Experiments

### 🔹 Direct Estimation

```bash id="1hel7v"
cd directly
bash run.sh
```

---

### 🔹 Pairwise Comparison

```bash id="ipre3b"
cd comparison
bash run.sh
```

---

### 🔹 Analysis

```bash id="gzody0"
cd experiments
python correlation.py
python pairwise_accuracy.py
```

---

## 🔁 Reproducibility

This repository is designed for **full reproducibility** of the reported results.

### What is included

* ✅ Complete dataset with all signals (exposure, Wikipedia, LLM outputs)
* ✅ Prompting pipelines
* ✅ Aggregation scripts
* ✅ Evaluation scripts

### Requirements

* Python 3.9+
* Dependencies listed in `requirements.txt`
* GPU recommended (optional)

### Notes

* Results may vary slightly due to LLM stochasticity
* Multiple runs + aggregation used for stability
* Models available via Hugging Face

---

## 📊 Key Findings

* Pretraining exposure strongly correlates with real-world popularity
* LLM judgments align more with **exposure than Wikipedia signals**
* Pairwise comparison yields the most reliable estimates
* Effects persist in long-tail entities

---

## 📄 Citation

```bibtex id="i46g8u"
@inproceedings{pretraining_exposure_popularity_2026,
  title={Pretraining Exposure Explains Popularity Judgments in Large Language Models},
  author={Anonymous},
  booktitle={SIGIR 2026},
  year={2026}
}
```

📌 **Camera-ready citation (with author names) coming soon.**

---

## 📜 License

MIT License — see `LICENSE`.

---

## 📬 Contact

Open an issue for questions or feedback.

---
