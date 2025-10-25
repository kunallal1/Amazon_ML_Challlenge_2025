# 🧠 Hybrid DistilBERT + Tabular Price Prediction
_Amazon ML Challenge 2025 – Top 1% (Rank 107 / 7000+)_

![Project Banner](https://via.placeholder.com/1200x400.png?text=Hybrid+DistilBERT+%2B+Tabular+Price+Prediction)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Transformers-DistilBERT-yellow.svg?logo=huggingface)](https://huggingface.co/)
[![Optuna](https://img.shields.io/badge/Optuna-HPO-success.svg)](https://optuna.org/)
[![Rank](https://img.shields.io/badge/Top%201%25%20-%20Rank%20107%2F7000+-brightgreen)](https://www.amazonmlchallenge.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 📖 Table of Contents
- [Overview](#-overview)
- [Model Architecture](#-model-architecture)
- [Custom Loss](#-custom-loss)
- [Training Strategy](#-training-strategy)
- [Results](#-results)
- [Tech Stack](#-tech-stack)
- [Insights](#-insights)
- [Installation](#-installation)
- [Usage](#-usage)
- [Contributors](#-contributors)
- [License](#-license)

---

## 🏁 Overview
This repository contains the solution for the **Amazon ML Challenge 2025**, where the goal was to **predict the optimal price** of e-commerce products using **text, image, and tabular features**.  

The evaluation metric was **SMAPE (Symmetric Mean Absolute Percentage Error)**.

> 🏆 Achieved **Top 1% (Rank 107 / 7000+)** among participants.

---

## ⚙️ Model Architecture

### 🧠 Text Encoder — DistilBERT
- Pretrained **DistilBERT** model for embedding `catalog_content`
- Mean, max, and attention pooling for richer text representation
- Locally loaded model for faster inference (`huggingface_hub`)

### 🔢 Tabular Features
Feature-engineered numeric inputs such as:  
- `Value_log`, `word_count`, `has_image`, `value_unit_ratio`, `text_len_per_word`
- Unit-level categorical embeddings (`nn.Embedding`)
- Scaled via `StandardScaler`

### 🔗 Fusion Layer
Combines:  
- [CLS], mean, max, and attention-pooled embeddings  
- Engineered numeric features  
- Unit embeddings  

Then passes through a **deep feed-forward network**:  
`Linear → LayerNorm → GELU → Dropout → Linear → Output`

---

## 🧮 Custom Loss
```python
HybridLoss = 0.7 * SMAPE + 0.3 * LogCosh
# SMAPE handles scale-invariant error
# LogCosh ensures smoother convergence and stability
```

This hybrid loss improved robustness to outliers and training stability.

---

## 🔧 Training Strategy

| Component              | Configuration                                           |
|------------------------|--------------------------------------------------------|
| Optimizer              | AdamW (separate LRs for BERT and head)                |
| Scheduler              | Cosine schedule with warmup                             |
| Precision              | Mixed FP16 (torch.amp)                                 |
| Early Stopping         | Based on validation SMAPE                               |
| Hyperparameter Tuning  | Optuna (dropout, hidden dims, LR ratios)              |

---

## 📈 Results

| Metric     | Value             |
|------------|-----------------|
| Best SMAPE | 44               |
| Rank       | 107 / 7000+      |
| Percentile | Top 1%           |

---

## 🧰 Tech Stack
- Python
- PyTorch
- Transformers (DistilBERT)
- Optuna
- Pandas
- Scikit-learn
- Kaggle
- Hugging Face Hub

---

## 💡 Insights
- 🧩 Multimodal fusion outperformed single-modality models.  
- 🧮 Hybrid loss improved robustness to outliers.  
- ⚖️ LayerNorm + GELU stabilized deeper stacks.  
- 🧠 Attention pooling captured fine-grained textual pricing cues.

---

## ⚙️ Installation
```bash
git clone https://github.com/your-username/amazon-price-predictor-hybrid.git
cd amazon-price-predictor-hybrid
pip install -r requirements.txt
```

---

## 👥 Contributors

| Name          | Role                       |
|---------------|----------------------------|
| Kunal Lal     | Model Design & Training    |
| Sainky Gupta  | Feature Engineering        |
| Aayush Prasad | Data Processing            |
| Manish Shaw   | Evaluation & Optimization  |

---

## 🏷️ Tags
`#AmazonMLChallenge2025 #MachineLearning #PyTorch #DistilBERT #Optuna #Hackathon #MultimodalLearning`

---

## 📜 License
This project is licensed under the **MIT License**.  
See the LICENSE file for details.

⭐ If you found this project helpful, give it a star on GitHub!






