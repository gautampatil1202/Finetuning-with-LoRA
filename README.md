# LoRA-RoBERTa AG News: Efficient Text Classification

This repository contains the implementation of a **parameter-efficient LoRA fine-tuning** approach applied to the **RoBERTa-base** model for multi-class text classification on the **AG News** dataset.

---

## Overview

- **Dataset:** AG News (120,000 articles, 4 categories)
- **Architecture:** RoBERTa-base Transformer with Low-Rank Adaptation (LoRA)
- **Enhancements:** Label smoothing, Cosine LR scheduling, validation monitoring, adapter-based fine-tuning
- **Goal:** Achieve high accuracy while staying under 1M trainable parameters using LoRA

---

## Methodology

This project fine-tunes a frozen `roberta-base` encoder using **LoRA (Low-Rank Adaptation)**, updating only injected low-rank matrices in attention layers.

- **LoRA Parameters:**
  - Rank: `r = 8`
  - Alpha: `16`
  - Dropout: `0.1`
  - Trainable Parameters: `~888,580` (vs ~125M full fine-tuning)
- **Target Modules:** Query and Value projection layers

---

## Model Architecture

- **Backbone:** RoBERTa-base
  - 12 Transformer layers
  - Hidden size: 768
  - Max token length: 128
- **Classifier Head:**
  - Dropout + Linear + ReLU + Output Layer (4 classes)

---

## Training Setup

- **Optimizer:** AdamW
- **Learning Rate Scheduler:** Cosine decay with 10% warm-up
- **Loss Function:** Cross-Entropy with Label Smoothing
- **Batch Size:** 16
- **Epochs:** 3
- **Evaluation:** Accuracy, F1-score, Confusion Matrix
- **Device:** CUDA / CPU

---

## Results

- **Final Validation Accuracy:** 92.57%
- **Final Validation Loss:** 0.4985
- **Macro F1-Score:** 0.92
- **Weighted F1-Score:** 0.93
- **Best Class:** Sports (F1 = 0.98)


