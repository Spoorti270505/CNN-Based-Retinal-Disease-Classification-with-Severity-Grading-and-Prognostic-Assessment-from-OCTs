# Eye Retinal Disease Classification – Multi-Task Learning

A deep learning project for retinal Optical Coherence Tomography (OCT) analysis using a Multi-Task Learning (MTL) framework. The system is designed to perform:

1. **Retinal disease classification**
2. **Severity grading**
3. **Prognostic risk prediction**

The project focuses on four OCT categories:

- Choroidal Neovascularization (CNV)
- Diabetic Macular Oedema (DME)
- Drusen
- Normal

The accompanying research paper describes a shared CNN feature encoder with task-specific heads for disease classification, severity grading, and prognosis assessment.

## Project Overview

The proposed framework uses a shared feature representation and three task-specific outputs:

```text
                         OCT Image
                             │
                             ▼
                    DenseNet121 Encoder
                             │
                             ▼
                     Shared Bottleneck
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        Disease Head   Severity Head   Prognosis Head
              │              │              │
              ▼              ▼              ▼
        4-class output   3-level grade    Risk score
```

The research methodology uses patient-aware dataset splitting to reduce the risk of having scans from the same patient distributed across training and evaluation subsets.

## Main Components

### 1. Disease Classification

The disease classification task predicts one of:

- CNV
- DME
- DRUSEN
- NORMAL

The project evaluates CNN architectures including:

- DenseNet121
- ResNet50
- EfficientNet-B0
- EfficientNetV2-B0 / TF-EfficientNet variants

The reported architecture-comparison results in the paper give DenseNet121 a classification accuracy of **97.53%** and an F1-score of **92.76%**.

### 2. Severity Grading

Severity is modeled as a secondary classification task with:

- None
- Mild
- Severe

Because clinician-annotated severity labels were unavailable, the paper uses a morphological biomarker heuristic based on the pathological occupancy ratio (POR).

The heuristic analyzes OCT morphology using image-processing operations such as intensity thresholding and region-of-interest processing.

### 3. Prognosis Assessment

The prognosis task predicts a continuous risk score in the range **0.0–1.0**.

The implementation derives prognosis targets from morphological characteristics of the OCT images rather than from longitudinal clinical outcome labels.

The regression code evaluates metrics including MAE, MSE, RMSE, R², Pearson correlation, Spearman correlation, and bootstrap confidence intervals.

## Dataset

The research paper reports a total of **83,484 OCT images**:

| Split | Images |
|---|---:|
| Training | 58,681 |
| Validation | 11,459 |
| Testing | 13,344 |
| **Total** | **83,484** |

Class distribution reported in the paper:

| Class | Images |
|---|---:|
| CNV | 37,205 |
| DME | 11,348 |
| Drusen | 8,616 |
| Normal | 25,915 |

The project uses the Kermany OCT dataset.

### Patient-Aware Splitting

The methodology extracts patient IDs from image filenames and keeps patient groups separated between train, validation, and test sets:

```text
P_train ∩ P_val ∩ P_test = ∅
```

This is intended to reduce patient-level data leakage during evaluation.

> **Important:** The dataset itself is not included in this repository. Download and prepare the dataset separately, then place it under the path expected by the training scripts.

## Preprocessing

The methodology described in the paper includes:

- Resizing OCT images to **224 × 224**
- Converting grayscale OCT images to three channels when required
- Random horizontal flipping during training
- Random rotation up to approximately ±10°
- Color jitter augmentation
- ImageNet normalization

The source code uses the ImageNet normalization values:

```text
Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]
```

## Model Architecture

The primary backbone used by the MTL implementation is **DenseNet121**.

The architecture consists of:

1. DenseNet121 shared encoder
2. Shared bottleneck
3. Disease classification head
4. Severity classification head
5. Prognosis regression head

The paper defines the combined training objective as:

```text
L_total = L_disease + λ1 L_severity + λ2 L_prognosis
```

with categorical cross-entropy used for the classification tasks and mean squared error for prognosis regression.

## Reliability and Interpretability

The project includes additional evaluation and interpretability components.

### Temperature Scaling

Temperature scaling is used to calibrate prediction confidence and evaluate Expected Calibration Error (ECE).

### Adaptive Test-Time Augmentation

Inference can use multiple transformed views of an OCT image, including:

- Original image
- Horizontal flip
- Rotation

The predictions are combined using weighted averaging.

### Grad-CAM

Grad-CAM visualizations are generated to inspect the image regions contributing to disease predictions.

### Additional Evaluation

The repository contains results for:

- Confusion matrices
- ROC curves
- Precision-recall curves
- Calibration analysis
- t-SNE feature visualization
- Misclassified examples
- Grad-CAM
- Training history
- Inference-time comparison
- Model comparison
- Regression error analysis
- Correlation analysis

## Repository Structure

```text
Eye-Retinal-Disease-Classfication-ML-Project/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── newml.py
│   ├── td.py
│   ├── tp.py
│   ├── ts.py
│   ├── train_disease.py
│   ├── train_severity.py
│   └── train_prognosis.py
│
├── results/
│   ├── disease_classification/
│   ├── overall_evaluation/
│   ├── prognosis/
│   └── severity_grading/
│
└── paper/
    └── 486.pdf
```

## Source Files

| File | Purpose |
|---|---|
| `newml.py` | Integrated / multi-task model and evaluation workflow |
| `train_disease.py` | Disease classification training and evaluation |
| `train_severity.py` | Severity grading training and evaluation |
| `train_prognosis.py` | Prognosis regression training and evaluation |
| `td.py` | Disease classification / evaluation workflow |
| `ts.py` | Severity-related evaluation workflow |
| `tp.py` | Prognosis / regression evaluation workflow |

## Results

The `results/` directory contains generated experimental outputs organized by task.

### Disease Classification

Includes:

- DenseNet121 checkpoint
- Classification reports
- Detailed metrics
- Confusion matrix
- ROC curves
- Precision-recall curves
- Calibration analysis
- t-SNE visualization
- Training history

### Severity Grading

Includes:

- DenseNet121 checkpoint
- Severity classification report
- Severity confusion matrix
- ROC curves
- Precision-recall curves
- Calibration analysis
- Training history
- Metrics CSV files

### Prognosis

Includes:

- DenseNet121 checkpoint
- TF-EfficientNetV2-B0 checkpoint
- Prognosis predictions
- Error analysis
- Error-by-range analysis
- Correlation analysis
- Training histories
- Regression result visualizations

### Overall Evaluation

Includes:

- Model comparison
- MTL vs STL comparison
- Grad-CAM samples
- Misclassified examples
- Class distribution
- Calibration analysis
- ROC and precision-recall curves
- t-SNE visualization
- Inference-time comparison
- Training history

## Installation

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
```

### Windows

```cmd
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Path

The training scripts expect the OCT dataset under:

```text
data/kermany2018/OCT2017
```

A typical directory layout is:

```text
data/
└── kermany2018/
    └── OCT2017/
        ├── train/
        │   ├── CNV/
        │   ├── DME/
        │   ├── DRUSEN/
        │   └── NORMAL/
        └── val/
            ├── CNV/
            ├── DME/
            ├── DRUSEN/
            └── NORMAL/
```

The complete dataset is intentionally excluded from Git.

## Running the Code

Example:

```bash
python src/train_disease.py
```

```bash
python src/train_severity.py
```

```bash
python src/train_prognosis.py
```

The exact command and configuration may vary depending on the selected script and experimental setup.

## Research Paper

The repository includes the associated paper:

**CNN-Based Retinal Disease Classification with Severity Grading and Prognostic Assessment from OCT Images**

Authors:

- Spoorti B
- Vrunda Patil
- Shreyas P Wali
- Satish Chikkamath
- Kaushik M B
- Suneeta V Budihal

The paper describes the MTL architecture, patient-aware splitting strategy, morphological biomarker heuristic, calibration, adaptive TTA, and Grad-CAM analysis.

## Important Research Note

The paper explicitly states that clinician-annotated ground-truth labels for the secondary tasks were unavailable. Severity and prognosis targets were therefore generated using a morphological biomarker heuristic and pathological occupancy ratio (POR).

Consequently, the severity and prognosis outputs should be interpreted as **research-model targets derived from image morphology**, not as validated clinical ground-truth outcomes.

Similarly, reported performance values are experimental results from the described dataset and evaluation methodology and should not be interpreted as clinical validation.

## Citation

If you use this repository or the associated work, please cite the accompanying paper.

```text
Spoorti B, Vrunda Patil, Shreyas P Wali, Satish Chikkamath,
Kaushik M B, Suneeta V Budihal,
"CNN-Based Retinal Disease Classification with Severity Grading
and Prognostic Assessment from OCT Images."
```

## Repository

Published Paper Link:

https://ieeexplore.ieee.org/document/11651259
## License

No license has been specified for this repository yet.
