# Enhancing Cardiovascular Disease Diagnosis Through Knowledge Distillation and Hybrid Transformer-CNN-SENet Architecture

Official implementation of a lightweight ECG classification framework based on knowledge distillation from a high-capacity **Transformer-CNN-SENet (T-CST)** teacher model to a lightweight **CNN-SENet (S-CS)** student model.

## Overview

Cardiovascular diseases (CVDs) are a major cause of mortality worldwide. Accurate and computationally efficient ECG analysis is therefore important for automated diagnosis, particularly in portable and wearable healthcare systems.

This work proposes a knowledge distillation framework in which the high-capacity **T-CST** teacher transfers its diagnostic knowledge to the lightweight **S-CS** student model.

The **T-CST** architecture combines convolutional neural networks (CNNs), **Squeeze-and-Excitation (SENet)** blocks, and **Transformer encoders** to capture both local and long-range temporal dependencies in ECG signals.

The lightweight **S-CS** model uses a CNN-SENet architecture and is trained using response-based knowledge distillation from the teacher model.

A forward lead-selection strategy is also employed to identify informative ECG leads and reduce the amount of input information required by the lightweight model.

Experiments on the **PTB-XL** dataset demonstrate that the student model can achieve performance close to the teacher while substantially reducing computational requirements. In particular, the proposed approach achieves approximately **63× reduction in computational cost** while using a reduced number of ECG leads.

<p align="center">
  <img src="figures/figure4.png"
       alt="T-CST and S-CS networks with knowledge distillation"
       width="900">
</p>

**Figure 4.** The complete structure of the T-CST and S-CS networks, where the T-CST network transfers its dark knowledge to the S-CS network through response-based knowledge distillation.

---

## Dataset

The experiments were conducted using the publicly available **PTB-XL ECG dataset**.

Each PTB-XL sample contains a **10-second, 12-lead ECG recording**, available at sampling frequencies of **500 Hz and 100 Hz**.

The dataset contains detailed annotations provided by cardiology experts. These annotations include diagnostic, form, and rhythm descriptors.

In this study, the **five superdiagnostic classes** were used:

| Class | Description            |
| ----- | ---------------------- |
| NORM  | Normal ECG             |
| MI    | Myocardial Infarction  |
| STTC  | ST/T Change            |
| CD    | Conduction Disturbance |
| HYP   | Hypertrophy            |

To reduce computational complexity and improve processing efficiency, ECG recordings sampled at **100 Hz** were used.

---

## Preprocessing

To address the imbalance between diagnostic classes and generate suitable input segments, a class-dependent sliding-window strategy was applied to the ECG recordings.

Each original ECG recording contains **10 seconds of data**, which was divided into segments of **2.5 seconds**.

### Data Split

The official record-wise PTB-XL split was applied **before windowing** to prevent data leakage.

* **Folds 1–8:** Training
* **Fold 9:** Validation
* **Fold 10:** Testing

Because the split was performed at the record level before window extraction, all windows generated from the same ECG recording remained in the same subset.

This ensures that no ECG recording is shared between the training, validation, and test sets.

### Windowing Strategy

A window length of **2.5 seconds** was used with class-dependent overlap rates.

For the training set:

* **CD:** 50% overlap
* **MI:** 50% overlap
* **STTC:** 50% overlap
* **HYP:** 80% overlap
* **NORM:** Randomly sampled 2.5-second segments

The higher overlap for the HYP class was used because of its limited number of available samples.

For the test set:

* **STTC:** 80% overlap
* **MI:** 80% overlap
* **CD:** 80% overlap
* **HYP:** 90% overlap
* **NORM:** 50% overlap

For the validation set, a **50% overlap** was applied to all classes.

The resulting sample distributions are reported in **Figure 5**, while examples of the different windowing and overlap strategies are illustrated in **Figure 6**.

---

## Models

### T-CST Teacher

The **T-CST (Transformer-CNN-SENet)** is the high-capacity teacher network.

It combines:

* Convolutional layers for local temporal feature extraction
* Squeeze-and-Excitation (SE) blocks for channel-wise feature recalibration
* Transformer encoder layers for modeling long-range temporal dependencies
* Fully connected layers for final classification

The teacher model is trained using the full ECG input and provides the soft predictions used for knowledge distillation.

### S-CS Student

The **S-CS (CNN-SENet)** is designed as a lightweight student architecture.

It combines:

* Convolutional layers
* Squeeze-and-Excitation blocks
* Fully connected classification layers

The student is trained to reproduce the useful information contained in the teacher's output distribution while maintaining a substantially lower computational cost.

---

## Knowledge Distillation

Response-based knowledge distillation is used to transfer the teacher's **dark knowledge** to the lightweight student network.

The distillation objective combines:

1. The Kullback–Leibler divergence between the softened teacher and student predictions.
2. The standard cross-entropy classification loss using the ground-truth labels.

The temperature parameter is set to:

**T = 2.0**

---

## Experimental Setup

The experiments were implemented using **Python and PyTorch**.

| Component             | Configuration   |
| --------------------- | --------------- |
| Python                | 3.11.8          |
| PyTorch               | 2.2.1           |
| CUDA                  | 11.8            |
| CPU                   | Intel Core i9   |
| RAM                   | 128 GB          |
| GPU                   | NVIDIA RTX 3090 |
| Batch Size            | 128             |
| Teacher Epochs        | 190             |
| Student Epochs        | 45              |
| Initial Learning Rate | 4 × 10⁻⁴        |
| Weight Decay          | 1 × 10⁻⁵        |
| Dropout               | 0.4             |
| Activation            | ReLU            |
| KD Temperature        | 2.0             |
| Student Optimizer     | AdamW           |

---

## Evaluation Metrics

The proposed models were evaluated using six standard classification metrics:

* Accuracy
* Precision
* Recall (Sensitivity)
* F1-Score
* Specificity
* Area Under the Receiver Operating Characteristic Curve (AUROC)

The metrics are calculated based on true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN).

### Accuracy

$$
Accuracy =
\frac{TP + TN}
{TP + TN + FP + FN}
$$

### Precision

$$
Precision =
\frac{TP}
{TP + FP}
$$

### Recall

$$
Recall =
\frac{TP}
{TP + FN}
$$

### F1-Score

$$
F1 =
2 \times
\frac{Precision \times Recall}
{Precision + Recall}
$$

### Specificity

$$
Specificity =
\frac{TN}
{TN + FP}
$$

### AUROC

AUROC represents the area under the Receiver Operating Characteristic (ROC) curve and measures the ability of the model to distinguish between different diagnostic classes across classification thresholds.

---

## Citation

If you use this implementation in your research, please cite:

**Nasim Beigzadeh and Abdolhossein Fathi.**

*Enhancing cardiovascular disease diagnosis through knowledge distillation and hybrid Transformer-CNN-SENet architecture.*

**Journal of Medical Engineering & Technology.**

