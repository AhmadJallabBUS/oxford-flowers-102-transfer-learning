# Option 2 Project Proposal
## Automated Classification of Parathyroid Scintigraphy Using Transfer Learning

**Course:** Computer Vision  
**Submission Date:** 2026-05-16  
**Proposal Deadline:** 2026-05-21  
**Group Members:** Ahmad Jallab (8240339) — Majdi Jaser (240320)

---

### Problem Statement

Primary hyperparathyroidism (PHPT) is one of the most common endocrine disorders, most often caused by a single parathyroid adenoma. Pre-operative localisation relies on dual-phase ⁹⁹ᵐTc-Sestamibi nuclear scintigraphy (MIBI scanning). A radiologist must interpret two or three complementary grayscale images per patient study and identify subtle tracer-retention patterns that distinguish adenoma from normal tissue. Inter-reader agreement is moderate (κ ≈ 0.55–0.70) and the task is time-intensive. We propose to build a computer vision system that classifies each patient study as **adenoma present** or **no adenoma** automatically, using only the two MIBI phases (early and delayed), without requiring the additional thyroid-tracer scan. This two-image constraint is a concrete clinical requirement from our medical collaborators at the University of Jordan Hospital Radiology Department, as eliminating the third scan reduces patient radiation dose, cost, and procedure time.

### Dataset

All data are provided directly by the Radiology Department, University of Jordan Hospital, under a research agreement. The current labeled set contains **101 patient studies**. Each study provides three grayscale nuclear medicine images: (1) early-phase MIBI, (2) delayed-phase MIBI, and (3) a thyroid-specific tracer image. Labels are binary: adenoma (positive) or normal/no finding (negative), confirmed against clinical and surgical outcomes. We will train and evaluate on the two-image input (images 1 and 2 only), treating image 3 as an optional ablation condition to measure the information cost of excluding it. A stratified patient-level split (≈70/15/15) will be used to prevent data leakage.

### Proposed Approach

We will use transfer learning with pretrained CNN backbones fine-tuned on the two-channel (or two-concatenated-image) input. The two-phase images for each study will be stacked channel-wise to form a single composite input tensor. We will compare two pretrained architectures — **MobileNetV2** and **ResNet50V2** — under a controlled setup (same input size, same augmentation, same classifier head) to identify which backbone generalises better on this small medical dataset. A frozen-backbone baseline followed by selective fine-tuning of upper layers will be the primary training strategy. Grad-CAM heatmaps will be generated for every prediction to provide spatial explainability aligned with radiologist review needs.

### Technical Feasibility

The dataset (101 studies × 3 images) fits entirely in memory and trains within minutes on Google Colab's free GPU tier. Image sizes will be standardised to 224×224. Augmentation (flips, slight rotation, brightness jitter) will compensate for the small dataset. All experiments are self-contained in a single Jupyter notebook and require no external labeled data beyond what the hospital provides.

### Evaluation Plan

| Metric | Rationale |
|--------|-----------|
| Macro F1-score | Primary metric; accounts for class imbalance |
| AUC-ROC | Threshold-independent discrimination |
| Sensitivity / Specificity | Clinically required for adenoma detection |
| Macro Precision / Recall | Full classification profile |

Results will be reported separately for (a) two-image input and (b) three-image input to quantify the diagnostic cost of removing the thyroid-tracer scan. A confusion matrix and per-class error analysis will be included.

### Why This Qualifies as Option 2

The project is in computer vision, uses deep learning (transfer learning with CNNs), is fully feasible on Colab, includes rigorous quantitative evaluation, and is comparable in difficulty to the default project — it uses the same two backbone architectures (MobileNetV2, ResNet50V2) but applied to a real medical imaging problem with a small dataset, multi-channel input fusion, and an explainability requirement that adds meaningful additional scope.
