## 🧠 Breast Cancer Histopathology Classification using Xception and MIGT

This repository implements a complete deep learning pipeline for **breast cancer histopathology image classification** using **Mutual Information Guided Training (MIGT)** and a fine-tuned **Xception convolutional neural network**.

The primary objective is to investigate how **guided dataset partitioning** influences model generalisation in comparison with **random dataset splitting**, using the **BreaKHis** dataset as a benchmark.

**Dataset download:**  
https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/

---

## 🧪 Experiments: MIGT vs Random Dataset Splitting

This study compares two dataset partitioning strategies under an identical deep learning architecture and training protocol:

- **MIGT-guided splitting**
- **Random splitting (baseline)**

Both experiments use the same model, hyperparameters, and evaluation metrics to ensure a fair comparison.

---
### ⚙️ MIGT Dataset Splitting

```python
from migt import MIGTSplitter

splitter = MIGTSplitter(
    dataset_root="/DATASETS/cancer/BreaKHis_final/",
    bins=4,
    min_bin=10,
    train=0.5,
    test=0.4,
    val=0.1,
    strict_shape=False,
    resize_to=(224, 224),
    mode="color",
    seed=42
)

splitter.run(output_root="/DATASETS/cancer/migt-color/")

```

### 📁 Dataset
- **Dataset:** BreaKHis
- **Classes:** Benign, Malignant
- **Split strategy:** Mutual Information Guided Training (MIGT)
  - Train: 50%
  - Validation: 10%
  - Test: 40%
- **Test samples:** 803 images

MIGT distributes samples based on their **mutual information content**, ensuring that each subset represents the underlying data distribution more consistently and reducing feature bias.

---

### 🧠 Model & Training Setup (Same for All Experiments)
- Backbone: **Xception** (ImageNet pretrained)
- Input size: 224 × 224 × 3
- Global Average Pooling + Dropout (0.3)
- Binary classifier (Sigmoid)
- Batch size: 32
- Optimiser:
  - Head training: Adam (1e-3)
  - Fine-tuning: Adam (1e-5)
- Training strategy:
  - Stage 1: Train head (backbone frozen)
  - Stage 2: Fine-tune last 30 layers (BatchNorm frozen)
- Best model selected via **validation loss**

---

### 📊 Test Results (MIGT)

| Metric | Value |
|------|------|
| **Accuracy** | **93.5%** |
| **AUC** | **0.977** |

**Classification Report**
- Benign: Precision 0.89 | Recall 0.91 | F1-score 0.90  
- Malignant: Precision 0.96 | Recall 0.95 | F1-score 0.95  

MIGT achieves **high recall for malignant cases** while maintaining stable training and validation behaviour.

<p align="center">
<img width="846" height="393" alt="11" src="https://github.com/user-attachments/assets/31cc7aed-1c0d-4994-a25c-760313aab220" />

<img width="855" height="393" alt="22" src="https://github.com/user-attachments/assets/ed547124-e743-416e-bc62-01a87d2d546c" />

<img width="498" height="416" alt="23" src="https://github.com/user-attachments/assets/66efc9a1-65d2-4e0d-9d54-c97eb8a3aa82" /> 

</p>


---

## 🔹 Experiment 2: Random Dataset Splitting (Baseline)

### 📁 Dataset
- **Dataset:** BreaKHis
- **Classes:** Benign, Malignant
- **Split strategy:** Random (seed = 42)
  - Train: 50%
  - Validation: 10%
  - Test: 40%
- **Test samples:** 805 images

Random splitting serves as a baseline and does not explicitly control the distribution of informative samples across subsets.

---

### 📊 Test Results (Random)

| Metric | Value |
|------|------|
| **Accuracy** | **92.8%** |
| **AUC** | **0.977** |

**Classification Report**
- Benign: Precision 0.91 | Recall 0.85 | F1-score 0.88  
- Malignant: Precision 0.94 | Recall 0.96 | F1-score 0.95  

While overall accuracy is comparable, training showed **higher variance** and less controlled data distribution compared to MIGT.

<p align="center"><img width="846" height="393" alt="31" src="https://github.com/user-attachments/assets/19917506-2a9a-4780-92a6-e20501d986dd" />
<img width="855" height="393" alt="32" src="https://github.com/user-attachments/assets/8e41af64-7409-4f23-8c25-553fab54783c" />
<img width="498" height="416" alt="33" src="https://github.com/user-attachments/assets/ae110f30-7e64-47b5-bc52-866c997e253e" /> </p>


---

## 🔍 Comparative Analysis

| Aspect | MIGT | Random |
|-----|-----|------|
| Accuracy | **93.5%** | 92.8% |
| AUC | 0.977 | 0.977 |
| Data Distribution Control | ✅ Yes | ❌ No |
| Training Stability | ✅ High | ⚠️ Moderate |

---

## 🧠 Key Takeaway

Although both methods achieve strong performance, **MIGT provides better control over data distribution and more stable generalisation**, which is especially important in **medical imaging applications** where dataset bias can significantly impact clinical reliability.

---

