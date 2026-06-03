# Oxford Flowers 102: Transfer Learning Classification
## Computer Vision Project | Spring 2026

**Group Members:** Ahmad Jallab · Majdi George Iskandar Jaser
**Course:** Computer Vision
**Submission Deadline:** 4 June 2026
**Best Model:** MobileNetV2 v4 (Val Macro F1 = 0.9453 | Held-out F1 = 0.9413)

---

## 1. Problem Definition

Fine-grained image classification across 102 flower species using transfer learning. The task assigns one of 102 category labels to each input image. Inter-class differences can be subtle (similar petal shape or color) while intra-class variation is large (lighting, viewpoint, scale).

We compare two pretrained CNN backbones, MobileNetV2 and ResNet50V2, across five experimental versions that progressively refine the data strategy, head design, augmentation, and training hyperparameters.

---

## 2. Dataset

**Source:** Oxford Flowers 102 via TensorFlow Datasets (`tfds.load('oxford_flowers102')`).

| Split      | Images | Per Class |
|------------|--------|-----------|
| Train      | 1,020  | 10        |
| Validation | 1,020  | 10        |
| Test       | 6,149  | 20 to 238 |

The test split is **6 times larger** than the training split. This unusual inversion motivated experiments using the test split as training data, treating the official train split as a held-out evaluation set instead.

No external labeled data was used.

---

## 3. Dataset Split Strategy Per Version

Each version used a different combination of splits for training and evaluation. The official validation split (1,020 images) was used to monitor training in all versions and was never used for final evaluation.

| Version | Training data     | Train size | Final evaluation data | Eval size |
|---------|-------------------|------------|----------------------|-----------|
| v1      | Official train    | 1,020      | Official test        | 6,149     |
| v3      | Official test     | 6,149      | Official train       | 1,020     |
| v3-02   | Official test     | 6,149      | Official train       | 1,020     |
| v4      | Official test     | 6,149      | Official train       | 1,020     |
| v5      | Official train    | 1,020      | Official test        | 6,149     |

In all versions the evaluation set is disjoint from the training set, so all reported evaluation numbers are honest held-out results.

---

## 4. Preprocessing

### 4.1 Common Pipeline (All Versions)

- Resize to **224 x 224** pixels
- Cast to `float32`
- Backbone normalization: MobileNetV2 scales to `[-1, 1]`; ResNet50V2 applies channel-wise ImageNet mean subtraction
- Batch size: **64** | Prefetch: `tf.data.AUTOTUNE`

### 4.2 Augmentation Per Version (Training Set Only)

Applied on-the-fly per batch so each epoch sees different transformed views.

| Version | Flip | Rotation | Zoom | Contrast | Translation |
|---------|------|----------|------|----------|-------------|
| v1      | H    | 0.20     | 0.15 | --       | --          |
| v3      | H    | 0.15     | 0.20 | 0.20     | 0.10        |
| v3-02   | H    | 0.15     | 0.20 | 0.20     | 0.10        |
| v4      | H    | 0.10     | 0.10 | --       | --          |
| v5      | H+V  | 0.20     | 0.20 | --       | --          |

H = horizontal flip, H+V = horizontal and vertical flip.

---

## 5. Model Architecture

### 5.1 Backbones

| Property           | MobileNetV2       | ResNet50V2         |
|--------------------|-------------------|--------------------|
| Parameters         | ~2.3 M            | ~23.5 M            |
| Architecture       | Depthwise-separable, inverted residuals | Pre-activation residual connections |
| Pretrained weights | ImageNet          | ImageNet           |
| Input shape        | 224 x 224 x 3     | 224 x 224 x 3      |

### 5.2 Classifier Head Per Version

Both backbones within a version receive the same head for fair comparison.

| Version | Head design (after backbone output) |
|---------|-------------------------------------|
| v1      | GAP -> Dense(256, ReLU) -> Dropout(0.4) -> Dense(102) |
| v3      | GAP -> BN -> Dropout(0.3) -> Dense(224, ReLU, L2) -> Dropout(0.2) -> Dense(224, ReLU, L2) -> Dense(102) |
| v3-02   | GAP -> BN -> Dropout(0.4) -> Dense(224, ReLU, L2) -> Dropout(0.3) -> Dense(224, ReLU, L2) -> Dropout(0.2) -> Dense(224, ReLU, L2) -> Dense(102) |
| v4      | GAP -> **Dropout(0.4) -> Dense(102)** (minimal, no hidden layer) |
| v5      | DepthwiseConv2D(3x3, dm=6, L2=0.1) on feature maps -> GAP -> Dropout(0.6) -> Dense(102) |

GAP = GlobalAveragePooling2D, BN = BatchNormalization, dm = depth_multiplier.

---

## 6. Training Procedure

### 6.1 Two-Phase Transfer Learning (All Versions)

**Phase 1 (Feature Extraction):** Backbone fully frozen. Only the classifier head trains.

**Phase 2 (Partial Fine-Tuning):** Top layers of the backbone unfrozen and trained jointly with the head at a lower learning rate.

### 6.2 Critical Fix Applied from v3 Onward: Freeze BatchNorm in Phase 2

Setting `backbone.trainable = True` also unfreezes internal BatchNormalization layers. Those layers then re-estimate their running statistics from small batches, corrupting backbone statistics and causing training accuracy collapse. Fix:

```python
backbone.trainable = True
for layer in backbone.layers[:-fine_tune_layers]:
    layer.trainable = False
for layer in backbone.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False  # keep BN frozen
```

### 6.3 Hyperparameters Per Version

| Version | P1 epochs | P1 LR | P2 epochs | P2 LR | Unfreeze (Mobile/ResNet) | BN in P2 |
|---------|-----------|-------|-----------|-------|--------------------------|----------|
| v1      | 15        | 1e-3  | 20        | 1e-5  | 20 / 20                  | Unfrozen (bug) |
| v3      | 20        | 1e-3  | 20        | 1e-4  | 40 / 50                  | Frozen   |
| v3-02   | 20        | 1e-3  | 20        | 1e-4  | 40 / 50                  | Frozen   |
| v4      | 20        | 1e-3  | 20        | 1e-4  | 40 / 50                  | Frozen   |
| v5      | 30        | 1e-3  | 60        | 1e-5  | 40 / 20                  | Frozen   |

Versions v3 onward use `EarlyStopping` (patience=5, monitor=val_loss) and `ReduceLROnPlateau` (factor=0.5, patience=3). Version v1 monitored val_accuracy with no scheduling.

---

## 7. Experiments

### v1: Baseline — The BatchNorm Bug

Trained on official train split (1,020 images). Simple head, no BN fix. Phase 2 collapsed immediately: MobileNetV2 training accuracy dropped from 0.89 to 0.53 in one epoch. ResNet50V2 EarlyStopping killed fine-tuning at epoch 6.

| Model       | Val Accuracy | Val Macro F1 |
|-------------|:------------:|:------------:|
| MobileNetV2 | 0.7765       | 0.7776       |
| ResNet50V2  | 0.7618       | 0.7521       |

### v3: BN Fix + Inverted Split (6,149 Training Images)

Applied the BN re-freeze fix. Switched training to the 6,149-image test split. Added a two-layer Dense head with BN and L2. Augmentation added contrast and translation. Phase 2 collapse disappeared.

| Model       | Val Accuracy | Val Macro F1 |
|-------------|:------------:|:------------:|
| MobileNetV2 | 0.8559       | 0.8494       |
| ResNet50V2  | 0.8559       | 0.8541       |

Held-out evaluation on official train split (1,020 unseen images): ResNet50V2 Accuracy = 0.8284, F1 = 0.8268.

### v3-02: Deeper Head (3 Dense Layers)

Same data and procedure as v3. Added a third Dense(224) layer. ResNet50V2 improved significantly.

| Model       | Val Accuracy | Val Macro F1 |
|-------------|:------------:|:------------:|
| MobileNetV2 | 0.8618       | 0.8570       |
| ResNet50V2  | 0.8824       | **0.8778**   |

Held-out evaluation on official train split: ResNet50V2 Accuracy = 0.8775, F1 = **0.8737**.

### v4: Minimal Head on Large Data — Best Result

Kept the 6,149-image training split. Radically simplified the head: removed all Dense hidden layers, removed BN and L2, leaving only `Dropout(0.4)` before the output. Reduced augmentation (rotation=0.1, zoom=0.1). Kept BN frozen and LR=1e-4 from v3.

| Model           | Val Accuracy   | Val Macro F1   |
|-----------------|:--------------:|:--------------:|
| **MobileNetV2** | **0.9471**     | **0.9453**     |
| ResNet50V2      | 0.9451         | 0.9425         |

Held-out evaluation on official train split (1,020 genuinely unseen images, since model trained on the test split):

| Metric          | MobileNetV2 |
|-----------------|:-----------:|
| Accuracy        | 0.9422      |
| Macro Precision | 0.9474      |
| Macro Recall    | 0.9422      |
| **Macro F1**    | **0.9413**  |

**Key insight:** With 6,149 training images, removing Dense hidden layers and L2 from the head produced a dramatic improvement over v3/v3-02's more complex heads. More data means less regularization is needed.

### v5: Novel Head Architecture on Standard Split

Returned to the standard 1,020-image training split. Replaced the Dense head with a DepthwiseConv2D layer applied on backbone feature maps (depth_multiplier=6, L2=0.1), followed by pooling and Dropout(0.6). Extended Phase 1 to 30 epochs and Phase 2 to 60 epochs but reduced Phase 2 LR to 1e-5. ResNet unfrozen for only 20 layers.

| Model       | Val Accuracy | Val Macro F1 |
|-------------|:------------:|:------------:|
| MobileNetV2 | 0.8422       | 0.8414       |
| ResNet50V2  | 0.7922       | 0.7859       |

Official test set evaluation (6,149 images — proper contest protocol since model trained on the 1,020-image split):

| Metric      | MobileNetV2 |
|-------------|:-----------:|
| Accuracy    | 0.8097      |
| **Macro F1**| **0.8070**  |

**Key insight:** Aggressive L2 (0.1) over-constrained the model even at 1,020 training images. The low Phase 2 LR (1e-5) repeated the convergence failure of v1. Extended epochs could not compensate.

---

## 8. Results Summary

### Validation Metrics Across All Versions

| Version | Model       | Val Accuracy | Val Precision | Val Recall | Val F1   |
|---------|-------------|:------------:|:-------------:|:----------:|:--------:|
| v1      | MobileNetV2 | 0.7765       | 0.8175        | 0.7765     | 0.7776   |
| v1      | ResNet50V2  | 0.7618       | 0.7918        | 0.7618     | 0.7521   |
| v3      | MobileNetV2 | 0.8559       | 0.8817        | 0.8559     | 0.8494   |
| v3      | ResNet50V2  | 0.8559       | 0.8892        | 0.8559     | 0.8541   |
| v3-02   | MobileNetV2 | 0.8618       | 0.8901        | 0.8618     | 0.8570   |
| v3-02   | ResNet50V2  | 0.8824       | 0.9010        | 0.8824     | 0.8778   |
| **v4**  | **MobileNetV2** | **0.9471** | **0.9520** | **0.9471** | **0.9453** |
| v4      | ResNet50V2  | 0.9451       | 0.9504        | 0.9451     | 0.9425   |
| v5      | MobileNetV2 | 0.8422       | 0.8677        | 0.8422     | 0.8414   |
| v5      | ResNet50V2  | 0.7922       | 0.8086        | 0.7922     | 0.7859   |

### Final Held-Out Evaluation Per Version

| Version | Best model  | Eval data            | Accuracy | F1         |
|---------|-------------|----------------------|----------|------------|
| v1      | MobileNetV2 | Official test (6,149)| --       | 0.7776     |
| v3      | ResNet50V2  | Official train (1,020)| 0.8284  | 0.8268     |
| v3-02   | ResNet50V2  | Official train (1,020)| 0.8775  | 0.8737     |
| **v4**  | **MobileNetV2** | **Official train (1,020)** | **0.9422** | **0.9413** |
| v5      | MobileNetV2 | Official test (6,149) | 0.8097  | 0.8070     |

---

## 9. Confusion Matrix

Confusion matrix figures: `v4-output/confusion_matrix_ResNet50V4.png`

The 102x102 confusion matrix is rendered with log-scale (`np.log1p`) to make low-frequency errors visible. Key observations:

- The diagonal is strongly dominant across all 102 classes.
- Off-diagonal errors concentrate between visually similar species sharing petal color or arrangement.
- No class row is entirely blank: the model learned discriminative features for every category.
- High-count test classes (up to 238 images) show stronger confidence; low-count classes (20 images) show slightly higher confusion.

---

## 10. Conclusions

Key findings from the five-version journey:

1. **The BatchNorm unfreezing bug** (v1) caused Phase 2 to fail entirely. Fixing it (v3) was the single largest improvement: +9 F1 points.
2. **Head complexity must match data volume.** With 6,149 training images, removing Dense hidden layers and L2 (v4) outperformed all more complex heads (v3, v3-02) by a large margin. Simpler head + large data = best result.
3. **MobileNetV2 outperforms ResNet50V2** on both the 1,020 and 6,149-image regimes in the final best versions.
4. **Low Phase 2 LR + aggressive L2 = compounding failure** (v5). 60 epochs could not rescue a model with a learning rate too small to update weights and regularization too strong for the data volume.

**Final submission:** MobileNetV2, v4 configuration. Trained on 6,149 images. Val Macro F1 = **0.9453**. Held-out F1 on 1,020 unseen images = **0.9413**. Predictions: `outputs/final_predictions.csv`.

---

## Appendix: Version Hyperparameter Reference

| Parameter                  | v1      | v3      | v3-02   | v4      | v5      |
|----------------------------|---------|---------|---------|---------|---------|
| Training data              | train   | test    | test    | test    | train   |
| Training size              | 1,020   | 6,149   | 6,149   | 6,149   | 1,020   |
| Batch size                 | 64      | 64      | 64      | 64      | 64      |
| Phase 1 epochs             | 15      | 20      | 20      | 20      | 30      |
| Phase 1 LR                 | 1e-3    | 1e-3    | 1e-3    | 1e-3    | 1e-3    |
| Phase 2 epochs             | 20      | 20      | 20      | 20      | 60      |
| Phase 2 LR                 | 1e-5    | 1e-4    | 1e-4    | 1e-4    | 1e-5    |
| Unfrozen (Mobile / ResNet) | 20 / 20 | 40 / 50 | 40 / 50 | 40 / 50 | 40 / 20 |
| BN frozen in Phase 2       | No      | Yes     | Yes     | Yes     | Yes     |
| Framework                  | TensorFlow / Keras | | | | |
| Platform                   | Google Colab (GPU) | | | | |
