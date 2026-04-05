# 🫁 Lab Assignment 2 – Transfer Learning (VGG16 for Pneumonia Detection)
Transfer Learning using VGG16 for Pneumonia Detection from Chest X-Ray images.  Two-phase training: head-only training (Phase 1) + fine-tuning last 4 layers (Phase 2).  Dataset: Kaggle Chest X-Ray (5,863 images).  Results: Accuracy 84.46%, ROC-AUC 0.925. Handles class imbalance using class weighting.


**Student Name:** Vedant Ingle
**PRN:** 202402040031
**Roll No:** 297
**Batch:** DL3
**Subject:** Deep Learning LAB
**Assignment No:** 02

---

## 📄 Aim

To fine-tune a pre-trained VGG16 convolutional neural network for binary classification of chest X-ray images (Normal vs. Pneumonia), demonstrating the practical application of transfer learning by modifying top layers and optimizing hyperparameters.

---

## 📁 Dataset

| Property | Details |
|---|---|
| **Name** | Chest X-Ray Images (Pneumonia) |
| **Source** | Kaggle — Paul Mooney |
| **Link** | https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia |
| **Total Images** | 5,863 JPEG images |
| **Classes** | NORMAL and PNEUMONIA |
| **Input Size** | 224 × 224 pixels (RGB) |
| **Train** | Normal: 1,341 — Pneumonia: 3,875 — Total: 5,216 |
| **Validation** | Normal: 8 — Pneumonia: 8 — Total: 16 |
| **Test** | Normal: 234 — Pneumonia: 390 — Total: 624 |
| **Class Imbalance** | ~2.9× more Pneumonia than Normal |

---

## 🏗️ Model Architecture

```
────────────────────────────────────────────────────
Pre-trained VGG16 (ImageNet weights)
  16 weight layers (13 Conv + 3 FC)
  Phase 1: ALL layers FROZEN
  Phase 2: Last 4 layers UNFROZEN (fine-tuned)
────────────────────────────────────────────────────
        ↓
Global Average Pooling 2D
        ↓
Dense (256 neurons, ReLU)
        ↓
Dropout (0.5)
        ↓
Dense (2 neurons, Softmax)  ← NORMAL / PNEUMONIA
────────────────────────────────────────────────────
```

---

## ⚙️ Hyperparameters

| Hyperparameter | Value | Justification |
|---|---|---|
| Image Size | 224 × 224 px | VGG16 designed for 224×224 |
| Batch Size | 32 | Balance memory & gradient stability |
| Phase 1 Epochs | 15 (max) | Head training + early stopping |
| Phase 2 Epochs | 8 (max) | Fine-tuning, shorter to avoid overfitting |
| Phase 1 LR | 1e-4 | Standard for head-only training |
| Phase 2 LR | 1e-5 | 10× smaller to preserve ImageNet weights |
| Dropout Rate | 0.5 | Prevent overfitting on small dataset |
| Dense Units | 256 | Sufficient for 2-class problem |
| Optimizer | Adam | Adaptive LR, widely used for CNNs |
| Loss Function | Categorical CE | Multi-class one-hot classification |
| Unfrozen Layers | Last 4 of VGG16 | Lower layers keep generic features |

---

## 🔬 Two-Phase Training Strategy

### Phase 1 — Head Training
- All VGG16 base layers are **FROZEN**
- Only the custom Dense/Dropout/Softmax head trains
- Learning rate: **1e-4** (higher for fast head learning)
- Allows new head to learn task-specific features without disturbing pre-trained weights

### Phase 2 — Fine-Tuning
- Last **4 layers** of VGG16 base are **UNFROZEN**
- Learning rate: **1e-5** (10× smaller to avoid catastrophic forgetting)
- Gently adapts upper layers to chest X-ray texture patterns

---

## 🔧 Data Preprocessing & Augmentation

### Training Images (Augmented):
| Technique | Value |
|---|---|
| Rescale | 1/255 (normalize to 0-1) |
| Rotation Range | ±15° |
| Width Shift | 10% |
| Height Shift | 10% |
| Zoom Range | 10% |
| Horizontal Flip | Yes |

### Validation & Test Images:
- Only normalized (rescale 1/255) — no augmentation

---

## 📊 Results

### Final Model Performance on Test Set

| Metric | Score |
|---|---|
| **Accuracy** | **84.46%** |
| Precision | 85.13% |
| Recall | 84.46% |
| F1-Score | 84.61% |
| **ROC-AUC** | **0.925** |

### Confusion Matrix

| | Predicted NORMAL | Predicted PNEUMONIA |
|---|---|---|
| **True NORMAL** | 200 (TN) | 34 (FP) |
| **True PNEUMONIA** | 63 (FN) | 327 (TP) |

---

## 📈 Comparison with Published Results

| Model / Study | Dataset | Acc (%) | Prec (%) | Rec (%) | F1 (%) |
|---|---|---|---|---|---|
| **Our Model (VGG16 + Fine-tune)** | Kaggle (5,863) | 84.46 | 85.13 | 84.46 | 84.61 |
| Sharma & Gulerial 2022 (VGG16+NN) | Kaggle (5,856) | 92.00 | 94.28 | 93.08 | 93.70 |
| Ayan & Unver 2019 (VGG16) | Kaggle (5,856) | 87.00 | N/A | N/A | N/A |
| Customized VGG16 (IEEE 2024) | Kaggle | 93.50 | N/A | N/A | N/A |
| VGG16-based DL (Eur. J. 2025) | Kaggle | 92.79 | 94.12 | 94.36 | 94.24 |

---

## 🖼️ Visualizations Generated

- ✅ Chest X-Ray sample images (Normal vs Pneumonia)
- ✅ Dataset class distribution bar chart
- ✅ Training curves — Phase 1 and Phase 2
- ✅ Confusion matrix (test set)
- ✅ ROC Curve (AUC = 0.925)
- ✅ Sample predictions with confidence scores

---

## 🔍 Key Conclusions

1. **Transfer learning** significantly reduces training time and data requirements compared to training from scratch
2. **Two-phase fine-tuning** (frozen base then gradual unfreeze) effectively adapts pre-trained CNN features to new domain
3. **Class imbalance** must be addressed explicitly using class weights — ignoring it leads to biased models
4. **Callbacks** (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint) are essential for stable training
5. Model performance is comparable to published VGG16-based benchmarks on the same dataset

---
