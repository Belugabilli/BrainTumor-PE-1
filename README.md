# Project Exhibition 1 (PE1): EfficientNet-B3 Brain Tumor Classification Experiment

Controlled architecture modification experiment comparing **EfficientNet-B3 + Improved Classification Head (Project Exhibition 1 - PE1)** against the **EfficientNet-B1 baseline** on the BRISC2025 brain tumor MRI dataset.

---

## 1. Project Purpose

This repository implements **Project Exhibition 1 (PE1)**, a controlled modification of the baseline classification pipeline.

The goal is to answer the research question:
> *"Which model performs better on the same BRISC2025 classification task under the same training/evaluation protocol?"*

---

## 2. PE1 Architecture

```
Input (3 x 224 x 224)
  ↓
EfficientNet-B3 Backbone (ImageNet Pretrained)
  ↓
Global Average Pooling (1536-dim)
  ↓
BatchNorm1d(1536)
  ↓
Dropout(0.30)
  ↓
Linear(1536 → 512)
  ↓
GELU Activation
  ↓
Dropout(0.20)
  ↓
Linear(512 → 4)
  ↓
Class Logits [glioma, meningioma, no_tumor, pituitary]
```

- **Backbone**: `efficientnet_b3` (timm)
- **Total Parameters**: 11,488,300
- **Trainable Parameters**: 11,488,300
- **Baseline Parameters**: 6,518,308 (+4,969,992 parameters for PE1)

---

## 3. Dataset Structure

Dataset: **BRISC2025 Brain Tumor MRI**

```
datasets/classification_task/
├── train/ (5,000 images total)
│   ├── glioma (1,147)
│   ├── meningioma (1,329)
│   ├── no_tumor (1,067)
│   └── pituitary (1,457)
└── test/ (1,000 official untouched test images)
    ├── glioma (254)
    ├── meningioma (306)
    ├── no_tumor (140)
    └── pituitary (300)
```

### Class Mapping
```python
CLASS_TO_INDEX = {
    "glioma": 0,
    "meningioma": 1,
    "no_tumor": 2,
    "pituitary": 3
}
```

### Data Split Protocol
- **Training Set (80%)**: 4,000 images (`seed=42`)
- **Validation Set (20%)**: 1,000 images (`seed=42`)
- **Official Test Set**: 1,000 images (never used during training/validation/selection)

---

## 4. Environment & Installation

- **Python Version**: 3.11.9
- **Conda Environment**: `brain-tumor-pe1`
- **Device**: Apple Silicon `mps`
- **Key Dependencies**:
  - `torch==2.13.0`
  - `torchvision==0.28.0`
  - `timm==1.0.28`
  - `albumentations`
  - `scikit-learn`

---

## 5. Configuration

Hyperparameters are configured in [`configs/classification.yaml`](configs/classification.yaml):

- **Image Size**: `224 × 224`
- **Batch Size**: `32`
- **Max Epochs**: `50`
- **Optimizer**: `Adam` (`lr=1e-4`, `weight_decay=1e-5`)
- **Loss**: `CrossEntropyLoss`
- **Scheduler**: `ReduceLROnPlateau` (`factor=0.1`, `patience=5`, `mode='min'`)
- **Random Seed**: `42`

---

## 6. Execution Commands

### Unit Tests
```bash
/opt/anaconda3/envs/brain-tumor-pe1/bin/python -m pytest tests/test_classifier.py tests/test_classification_dataset.py tests/test_dataloader.py tests/test_pe1_config.py -v
```

### 1-Epoch Smoke Test
```bash
/opt/anaconda3/envs/brain-tumor-pe1/bin/python train_classifier.py --epochs 1 --checkpoint-dir weights/classification_smoke --log-dir logs/classification_smoke --num-workers 0
```

### Official 50-Epoch Training
```bash
/opt/anaconda3/envs/brain-tumor-pe1/bin/python train_classifier.py --config configs/classification.yaml --num-workers 4
```

### Official Test Set Evaluation
```bash
/opt/anaconda3/envs/brain-tumor-pe1/bin/python -m evaluation.evaluate_classifier --config configs/classification.yaml --checkpoint weights/classification/best_model.pth --results-dir results
```

---

## 7. Measured Experimental Results & Baseline Comparison

All metrics were computed on the official **1,000-image untouched test set** using the best validation checkpoint (`epoch 13`).

### Aggregate Performance Comparison

| Metric | Baseline (EfficientNet-B1) | PE1 (EfficientNet-B3 + Head) | Difference (PE1 - Baseline) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **0.9920** | 0.9870 | -0.0050 |
| **Macro Precision** | **0.9930** | 0.9853 | -0.0077 |
| **Macro Recall** | **0.9932** | 0.9890 | -0.0042 |
| **Macro F1** | **0.9931** | 0.9871 | -0.0060 |
| **Weighted F1** | **0.9920** | 0.9870 | -0.0050 |
| **Parameters** | **6,518,308** | 11,488,300 | +4,969,992 |
| **Training Duration** | N/A | 4,445.04 sec (50 epochs) | N/A |
| **Inference Time / Image** | N/A | 1.65 ms (`mps`) | N/A |

### PE1 Per-Class Breakdown (Test Set)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **glioma** | 0.9921 | 0.9921 | 0.9921 | 254 |
| **meningioma** | 0.9803 | 0.9771 | 0.9787 | 306 |
| **no_tumor** | 0.9722 | 1.0000 | 0.9859 | 140 |
| **pituitary** | 0.9966 | 0.9867 | 0.9916 | 300 |
| **Overall / Total** | **0.9871** | **0.9870** | **0.9870** | **1,000** |

---

## 8. Research Conclusion

Under identical training/evaluation protocols and data splits:
- **Baseline EfficientNet-B1** achieves an accuracy of **0.9920** and Macro F1 of **0.9931** with **6.52M parameters**.
- **PE1 (EfficientNet-B3)** achieves an accuracy of **0.9870** and Macro F1 of **0.9871** with **11.49M parameters**.

**Verdict**: Increasing backbone size to EfficientNet-B3 and using the modified classification head did **not** improve classification performance on this dataset split. The baseline EfficientNet-B1 maintains slightly higher accuracy and F1 score with ~43% fewer parameters.

---

## 9. Reproducibility & Output Artifacts

All experiment metadata, checkpoints, and evaluation results are saved under:
- `weights/classification/best_model.pth` & `pe1_best.pth`
- `results/pe1_test_metrics.json`
- `results/pe1_classification_report.txt`
- `results/pe1_confusion_matrix.png`
- `results/pe1_predictions.csv`
- `results/baseline_vs_pe1_comparison.csv`
- `results/baseline_vs_pe1_comparison.md`
- `results/pe1_analysis.md`
- `results/pe1_experiment_metadata.json`
