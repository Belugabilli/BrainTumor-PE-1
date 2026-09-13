# Difference Matrix: OLD Baseline vs NEW Project Exhibition 1 (PE1)

Comparative evaluation between the **OLD Baseline Repository** ([BrainTumor-Reproduction](https://github.com/Belugabilli/BrainTumor-Reproduction)) and the **NEW Project Exhibition 1 (PE1) Model** ([BrainTumor-PE-1](https://github.com/Belugabilli/BrainTumor-PE-1.git)).

---

## 1. System & Architecture Comparison

| Feature / Property | OLD Baseline (`BrainTumor-Reproduction`) | NEW Project Exhibition 1 (`BrainTumor-PE1`) | Difference (NEW - OLD) |
| :--- | :--- | :--- | :--- |
| **Model Backbone** | EfficientNet-B1 | EfficientNet-B3 | Upgrade B1 → B3 |
| **Feature Vector Dimension** | 1,280 | 1,536 | +256 features |
| **Classification Head** | `Dropout(0.3) → Linear(1280 → 4)` | `BatchNorm1d(1536) → Dropout(0.3) → Linear(1536 → 512) → GELU → Dropout(0.2) → Linear(512 → 4)` | Added BatchNorm, GELU, 512 hidden layer |
| **Total Parameters** | **6,518,308** | **11,488,300** | **+4,969,992 (+76.25%)** |
| **Trainable Parameters** | 6,518,308 | 11,488,300 | +4,969,992 |
| **Input Image Resolution** | 224 × 224 | 224 × 224 | Identical |
| **Optimizer & Learning Rate** | Adam ($10^{-4}$), Weight Decay $10^{-5}$ | Adam ($10^{-4}$), Weight Decay $10^{-5}$ | Identical |
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
| **no_tumor** | **0.9933** | 0.9859 | -0.0074 | 300 images |
| **pituitary** | **1.0000** | 0.9916 | -0.0084 | 140 images |

---

## 4. Key Takeaways & Scientific Summary

> [!NOTE]
> **Key Finding**: The **OLD Baseline (EfficientNet-B1)** outperforms the larger NEW PE1 architecture across accuracy (+0.50%) and F1-score (+0.60%) while requiring **43% fewer parameters** (6.52M vs 11.49M).

1. **Accuracy & F1-Score**: The OLD Baseline achieves **99.20% Accuracy** and **0.9931 Macro F1**, whereas the NEW PE1 model achieves **98.70% Accuracy** (-0.0050 difference).
2. **Parameter Efficiency**: EfficientNet-B1 has **6.52M parameters**, whereas EfficientNet-B3 with custom classification head expands parameter count by +76.25% to **11.49M parameters** without performance gain on this dataset size.
3. **Verdict**: Upgrading backbone size and head complexity without expanding dataset diversity or using learning rate decay tuning led to slight overfitting, making the baseline model superior in both accuracy and parameter efficiency.
