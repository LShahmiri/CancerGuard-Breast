# Breast Cancer Histopathology Classification using Xception and MIGT

This repository presents a deep learning framework for breast cancer
histopathology image classification using a transfer learning approach
based on the Xception architecture.  
The study focuses on understanding how **dataset selection strategies**
influence model generalization and overfitting under identical
experimental conditions.

---

## Motivation

In medical image classification, high accuracy alone does not guarantee
robust or reliable performance. Models trained on randomly selected
datasets may achieve impressive accuracy while suffering from severe
overfitting, limiting their applicability to unseen data.

This project investigates whether **guiding dataset selection using
Mutual Information (MI)** can mitigate overfitting and improve
generalization, particularly when working with high-dimensional RGB
histopathology images.

---

## Key Findings

Under identical experimental conditions and using the **same RGB
images**, the results demonstrate that the **data selection strategy
plays a critical role in controlling overfitting**:

- Random RGB sampling can achieve high classification accuracy but often
  leads to pronounced overfitting, indicated by a large gap between
  training and validation performance.
- Applying **Mutual Information–guided selection to colorful (RGB)
  images significantly reduces overfitting**, resulting in a more stable
  learning process and improved generalization.
- Notably, **MI-guided RGB selection outperforms MI-guided grayscale
  selection** in terms of overfitting reduction, highlighting the
  importance of preserving informative color cues during MI-based sample
  selection.

These findings show that **dataset selection can act as an implicit
regularization mechanism**, improving model robustness without altering
the network architecture or training hyperparameters.

---

## Contributions

- Systematic analysis of overfitting under identical training conditions
- Demonstration that dataset selection strategy impacts generalization
  more strongly than raw accuracy
- Integration of Mutual Information–guided dataset partitioning with
  deep CNN training
- Comparative evaluation of Random, MI-guided RGB, and MI-guided
  Grayscale datasets
- Reproducible, config-driven experimental pipeline

---

## Dataset

- **BreaKHis**: Breast Cancer Histopathological Images
- Binary classification:
  - Benign
  - Malignant
- Images organized into `train`, `validation`, and `test` splits
- MIGT datasets are pre-generated using mutual information–based sample
  selection

> Due to dataset licensing, images are not included in this repository.

---

## Experimental Settings

Three experimental configurations are evaluated:

1. **Random RGB**
   - Standard random sampling
   - RGB images

2. **MIGT RGB**
   - Mutual Information–guided selection
   - RGB images

3. **MIGT Grayscale**
   - Mutual Information–guided selection
   - Grayscale images

All experiments use the same model architecture and training protocol to
ensure a fair comparison.

---

## Model Architecture

- Backbone: **Xception** (pretrained on ImageNet)
- Input size: `224 × 224`
- Global Average Pooling
- Dropout (0.3)
- Sigmoid output layer for binary classification

---

## Training Strategy

1. **Stage 1 – Head Training**
   - Base model frozen
   - Train classifier head

2. **Stage 2 – Fine-Tuning**
   - Unfreeze last 30 layers of Xception
   - Batch Normalization layers kept frozen
   - Reduced learning rate for stable convergence

---

## Evaluation Metrics

- Accuracy
- Area Under the ROC Curve (AUC)
- Confusion Matrix
- Precision, Recall, and F1-score
- ROC Curve visualization

---






