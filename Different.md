# Difference Matrix: OLD (Baseline) vs NEW (PE1)

Comparison between the **OLD Baseline** (`/Users/hanish/Downloads/BrainTumor-Reproduction`) and the **NEW PE1 Model** (`/Users/hanish/Downloads/BrainTumor-PE1`).

---

## 1. System & Architecture Comparison

| Feature / Property | OLD Baseline (`BrainTumor-Reproduction`) | NEW PE1 (`BrainTumor-PE1`) | Difference (NEW - OLD) |
| :--- | :--- | :--- | :--- |
| **Model Backbone** | EfficientNet-B1 | EfficientNet-B3 | Upgrade B1 → B3 |
| **Feature Vector Dimension** | 1,280 | 1,536 | +256 features |
| **Classification Head** | `Dropout(0.3) → Linear(1280 → 4)` | `BatchNorm1d(1536) → Dropout(0.3) → Linear(1536 → 512) → GELU → Dropout(0.2) → Linear(512 → 4)` | Added BatchNorm, GELU, 512 hidden layer |
| **Total Parameters** | **6,518,308** | **11,488,300** | **+4,969,992 (+76.25%)** |
| **Trainable Parameters** | 6,518,308 | 11,488,300 | +4,969,992 |
| **Input Image Resolution** | 224 × 224 | 224 × 224 | Identical |
| **Optimizer & LR** | Adam ($10^{-4}$), Weight Decay $10^{-5}$ | Adam ($10^{-4}$), Weight Decay $10^{-5}$ | Identical |
| **LR Scheduler** | `ReduceLROnPlateau(factor=0.1, patience=5)` | `ReduceLROnPlateau(factor=0.1, patience=5)` | Identical |
| **Loss Function** | CrossEntropyLoss | CrossEntropyLoss | Identical |
| **Dataset Split Protocol** | 4,000 Train / 1,000 Val / 1,000 Test | 4,000 Train / 1,000 Val / 1,000 Test | Identical (`seed=42`) |

---

## 2. Test Set Performance Difference Matrix (1,000-Image Official Test Set)

| Metric | OLD Baseline (`EfficientNet-B1`) | NEW PE1 (`EfficientNet-B3`) | Absolute Difference (NEW - OLD) | Relative Change |
| :--- | :---: | :---: | :---: | :---: |
| **Test Loss** | **0.0236** | 0.0562 | +0.0326 | +138.1% |
| **Overall Accuracy** | **0.9920 (99.20%)** | 0.9870 (98.70%) | **-0.0050 (-0.50%)** | -0.50% |
| **Macro Precision** | **0.9930** | 0.9853 | -0.0077 | -0.78% |
| **Macro Recall** | **0.9932** | 0.9890 | -0.0042 | -0.42% |
| **Macro F1-Score** | **0.9931** | 0.9871 | **-0.0060** | -0.60% |
| **Weighted Precision** | **0.9921** | 0.9871 | -0.0050 | -0.50% |
| **Weighted Recall** | **0.9920** | 0.9870 | -0.0050 | -0.50% |
| **Weighted F1-Score** | **0.9920** | 0.9870 | **-0.0050** | -0.50% |

---

## 3. Per-Class F1 Score Breakdown

| Tumor Class | OLD Baseline F1 | NEW PE1 F1 | F1 Difference (NEW - OLD) | Test Support |
| :--- | :---: | :---: | :---: | :---: |
| **glioma** | **0.9922** | 0.9921 | -0.0001 | 254 images |
| **meningioma** | **0.9870** | 0.9787 | -0.0083 | 306 images |
| **no_tumor** | **0.9933** | 0.9859 | -0.0074 | 140 / 300 images |
| **pituitary** | **1.0000** | 0.9916 | -0.0084 | 300 / 140 images |

---

## 4. Scientific Conclusion

1. **Accuracy & F1**: The **OLD Baseline (EfficientNet-B1)** achieves higher accuracy (**0.9920** vs 0.9870) and higher Macro F1 (**0.9931** vs 0.9871) than NEW PE1.
2. **Model Efficiency**: The OLD baseline requires **4,969,992 fewer parameters** (6.52M vs 11.49M).
3. **Verdict**: Upgrading the backbone to EfficientNet-B3 and adding a 512-dim GELU classification head did not improve performance on this task under the controlled experimental protocol.
