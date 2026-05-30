# System Architecture — Oxford Flowers 102 Transfer Learning Classifier

**Course:** Computer Vision | **Deadline:** 4 June 2026  
**Dataset:** Oxford Flowers 102 (102 classes) | **Approach:** Transfer Learning

---

## 1. High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT PIPELINE                              │
│   TF Datasets  →  Resize  →  Cast to float32  →  Preprocess  →     │
│   Augment (Train only)  →  Batch & Prefetch (No caching in code)    │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
               ┌────────────────┴─────────────────┐
               ▼                                  ▼
   ┌───────────────────────┐          ┌───────────────────────┐
   │   MODEL A             │          │   MODEL B             │
   │   MobileNetV2         │          │   ResNet50V2          │
   │   (ImageNet weights)  │          │   (ImageNet weights)  │
   │   ─────────────────   │          │   ─────────────────   │
   │   Frozen backbone     │          │   Frozen backbone     │
   │   + Custom head       │          │   + Custom head       │
   └───────────┬───────────┘          └───────────┬───────────┘
               │                                  │
               ▼                                  ▼
   ┌───────────────────────┐          ┌───────────────────────┐
   │  Phase 1: Train head  │          │  Phase 1: Train head  │
   │  (backbone frozen)    │          │  (backbone frozen)    │
   └───────────┬───────────┘          └───────────┬───────────┘
               │                                  │
               ▼                                  ▼
   ┌───────────────────────┐          ┌───────────────────────┐
   │  Phase 2: Fine-tune   │          │  Phase 2: Fine-tune   │
   │  (last 20 layers unfrz)│          │  (last 20 layers unfrz)│
   └───────────┬───────────┘          └───────────┬───────────┘
               │                                  │
               └────────────────┬─────────────────┘
                                ▼
               ┌────────────────────────────────┐
               │       EVALUATION MODULE        │
               │  Accuracy / Precision /        │
               │  Recall / F1 / Confusion Mat.  │
               └────────────────┬───────────────┘
                                ▼
               ┌────────────────────────────────┐
               │    SELECT BEST MODEL           │
               │    → Save Test Predictions     │
               └────────────────────────────────┘
```

---

## 2. Project File Structure

The real implementation is structured as a consolidated notebook with outputs written to an `outputs` directory:

```
project/
├── Project Description Spring 2026 1.pdf   # Course project specification guidelines
├── SYSTEM_ARCHITECTURE.md                 # Project architecture and implementation specs
├── cell_check_output-v1.md                # Markdown log containing execution outputs of training
├── flowers102_classifier.ipynb            # Consolidated notebook with the entire training/evaluation pipeline
├── mobilenetv2-trainingAccuracy-v1.png    # Pre-generated training accuracy curve (v1)
├── resnet50-trainingAccuracy-v1.png       # Pre-generated training accuracy curve (v1)
├── outputs/                               # Output directory automatically created by code
│   ├── best_MobileNetV2.keras             # Best saved checkpoint for MobileNetV2 (monitors val_accuracy)
│   ├── best_ResNet50V2.keras              # Best saved checkpoint for ResNet50V2 (monitors val_accuracy)
│   ├── curves_MobileNetV2.png             # Epoch vs Accuracy training curves for MobileNetV2
│   ├── curves_ResNet50V2.png              # Epoch vs Accuracy training curves for ResNet50V2
│   ├── confusion_matrix_MobileNetV2.png   # Log-scale validation confusion matrix heatmap (for the best model)
│   └── final_predictions.csv              # Test set model predictions containing 'predicted' and 'true' columns
└── other project option2/                 # Alternative proposal materials
    ├── Proposal_Parathyroid_CV.docx
    ├── Proposal_Parathyroid_CV.pdf
    └── proposal.md
```

---

## 3. Data Pipeline

### 3.1 Dataset Loading

Dataset is loaded using TensorFlow Datasets:

```python
(ds_train, ds_val, ds_test), info = tfds.load(
    'oxford_flowers102',
    split=['train', 'validation', 'test'],
    as_supervised=True,
    with_info=True
)
# Splits: train=1020, val=1020, test=6149 (102 classes)
```

### 3.2 Preprocessing

| Step | Train Pipeline | Val / Test Pipeline |
|------|----------------|---------------------|
| Resize | `224×224` | `224×224` |
| Cast | `tf.float32` | `tf.float32` |
| Normalization | Backbone preprocess function | Backbone preprocess function |
| Shuffling | `shuffle(1024)` | None |
| Augmentation | Enabled | Disabled |
| Batch Size | `32` | `32` |
| Prefetch | `tf.data.AUTOTUNE` | `tf.data.AUTOTUNE` |

> Backbone-specific preprocessing handles appropriate scaling: MobileNetV2 normalizes inputs to `[-1, 1]` while ResNet50V2 performs channel-wise zero-centering using ImageNet stats.

### 3.3 Data Augmentation (train only)

```python
augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.15),
])
```

### 3.4 Pipeline Assembly

```python
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32
IMG_SIZE = 224

def make_pipeline(ds, preprocess_fn, training=False):
    def prep(img, label):
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32)
        img = preprocess_fn(img)
        return img, label

    ds = ds.map(prep, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.shuffle(1024)
        ds = ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
```

---

## 4. Model Architecture

### 4.1 Shared Classifier Head

To ensure a fair comparison, both models append the exact same classifier head design:

```
Backbone (MobileNetV2 or ResNet50V2)
    └── GlobalAveragePooling2D
        └── Dense(256, activation='relu')
            └── Dropout(0.4)
                └── Dense(102, activation='softmax')
```

### 4.2 Model A — MobileNetV2

| Property | Value |
|----------|-------|
| Input shape | `(224, 224, 3)` |
| Backbone params | ~2.3 M |
| Head Trainable params | ~355 K (dense layers) |
| Fine-tune layers | Last 20 layers of backbone unfrozen in Phase 2 |
| Backbone preprocess | `tf.keras.applications.mobilenet_v2.preprocess_input` |

### 4.3 Model B — ResNet50V2

| Property | Value |
|----------|-------|
| Input shape | `(224, 224, 3)` |
| Backbone params | ~23.5 M |
| Head Trainable params | ~550 K (dense layers) |
| Fine-tune layers | Last 20 layers of backbone unfrozen in Phase 2 |
| Backbone preprocess | `tf.keras.applications.resnet_v2.preprocess_input` |

---

## 5. Training Strategy

### 5.1 Two-Phase Transfer Learning

```
Phase 1 — Feature Extraction
─────────────────────────────
  Backbone:  FROZEN (trainable=False, weights='imagenet')
  Train:     classifier head only
  Optimizer: Adam(learning_rate=1e-3)
  Loss:      sparse_categorical_crossentropy
  Epochs:    15 epochs
  Goal:      Train classification head weights without modifying backbone weights

Phase 2 — Fine-Tuning
──────────────────────
  Backbone:  Last 20 layers of backbone set to trainable=True, rest remain FROZEN
  Train:     Unfrozen backbone layers + classifier head
  Optimizer: Adam(learning_rate=1e-5) (low learning rate to prevent catastrophic forgetting)
  Loss:      sparse_categorical_crossentropy
  Epochs:    20 epochs
  Goal:      Refine high-level feature extraction filters for Oxford Flowers 102 patterns
```

### 5.2 Callbacks

Both phases use Early Stopping and Checkpointing to ensure training stability:

```python
callbacks=[
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=f'outputs/best_{name}.keras',
        save_best_only=True,
        monitor='val_accuracy'
    )
]
```

### 5.3 Loss & Metrics

The network is compiled to monitor `accuracy` during training. Macro-averaged metrics are calculated post-training:

```python
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)
```

---

## 6. Evaluation Module

### 6.1 Metrics Computed on Validation / Test Set

Validation and test predictions are loaded from the best checkpoint and metrics are computed macro-averaged:

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

metrics = {
    'accuracy':  accuracy_score(y_true, y_pred),
    'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
    'recall':    recall_score(y_true, y_pred, average='macro', zero_division=0),
    'f1':        f1_score(y_true, y_pred, average='macro', zero_division=0),
}
```

### 6.2 Outputs Generated

- **Model Comparison Table**: Compares validation metrics of both model families.
- **Confusion Matrix**: Logs a 102x102 heatmap using `np.log1p(cm)` to visualize misclassification distribution.
- **Training Curves**: Generates accuracy progression plots (`outputs/curves_{name}.png`).
- **Predictions CSV**: Saves a dataframe with `predicted` and `true` columns for test evaluation.

---

## 7. Experiment Tracking

### Model Performance (Validation Set Results)

According to execution logs (`cell_check_output-v1.md`), MobileNetV2 slightly outperformed ResNet50V2 on the validation set (primarily due to macro F1-score performance):

| Model | Phase 1 LR | Phase 2 LR | Fine-Tune Layers | Val Accuracy | Val Precision | Val Recall | Val F1-score |
|---|---|---|---|---|---|---|---|
| **MobileNetV2** (Best) | `1e-3` | `1e-5` | Last 20 | **0.7765** | **0.8175** | **0.7765** | **0.7776** |
| **ResNet50V2** | `1e-3` | `1e-5` | Last 20 | 0.7618 | 0.7918 | 0.7618 | 0.7521 |

---

## 8. Final Prediction Generation

Predictions are executed on the test dataset using the best model determined by F1-score:

```python
test_ds = make_pipeline(ds_test, best_preprocess, training=False)

y_true_test, y_pred_test = [], []
for imgs, labels in test_ds:
    preds = best_model.predict(imgs, verbose=0)
    y_pred_test.extend(np.argmax(preds, axis=1))
    y_true_test.extend(labels.numpy())
y_true_test = np.array(y_true_test)
y_pred_test = np.array(y_pred_test)

# Save predictions for test evaluation
pd.DataFrame({'predicted': y_pred_test, 'true': y_true_test}).to_csv(
    'outputs/final_predictions.csv', index=False
)
```

---

## 9. Deliverables Checklist

- [x] `flowers102_classifier.ipynb` — Consolidated notebook implementing data prep, training, and evaluation pipeline
- [x] `SYSTEM_ARCHITECTURE.md` — Updated architecture document representing the actual implementation state
- [x] `cell_check_output-v1.md` — Notebook execution log for MobileNetV2 and ResNet50V2
- [x] `mobilenetv2-trainingAccuracy-v1.png` & `resnet50-trainingAccuracy-v1.png` — Pre-generated training curves
- [x] `outputs/final_predictions.csv` — Generated final test set predictions with `predicted` and `true` labels
- [ ] Setup and run instructions README
- [ ] Computer Vision Course Report (4-6 pages)
- [ ] Presentation Slides (10 mins)
