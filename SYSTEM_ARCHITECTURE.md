# System Architecture — Oxford Flowers 102 Transfer Learning Classifier

**Course:** Computer Vision | **Deadline:** 4 June 2026  
**Dataset:** Oxford Flowers 102 (102 classes) | **Approach:** Transfer Learning

---

## 1. High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT PIPELINE                              │
│  TF Datasets  →  Resize  →  Normalize  →  Augment  →  Batch/Cache  │
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
   │  (top N layers unfrz) │          │  (top N layers unfrz) │
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
               │    → Final Contest Predictions │
               └────────────────────────────────┘
```

---

## 2. Project File Structure

```
project/
├── data/                        # auto-downloaded by tfds
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_mobilenetv2.ipynb
│   ├── 03_resnet50v2.ipynb
│   └── 04_evaluation_final.ipynb
├── src/
│   ├── data_pipeline.py         # dataset loading, augmentation, splits
│   ├── model_builder.py         # build_model(backbone_name, ...)
│   ├── trainer.py               # two-phase training loop
│   └── evaluate.py              # metrics, confusion matrix, error analysis
├── outputs/
│   ├── mobilenetv2_history.json
│   ├── resnet50v2_history.json
│   ├── mobilenetv2_metrics.json
│   ├── resnet50v2_metrics.json
│   ├── confusion_matrix_*.png
│   └── final_predictions.csv
├── README.md
└── requirements.txt
```

---

## 3. Data Pipeline

### 3.1 Dataset Loading

```python
# via TensorFlow Datasets (recommended)
import tensorflow_datasets as tfds

(ds_train, ds_val, ds_test), info = tfds.load(
    'oxford_flowers102',
    split=['train', 'validation', 'test'],
    as_supervised=True,
    with_info=True
)
# Splits: train=1020, val=1020, test=6149  (102 classes)
```

### 3.2 Preprocessing

| Step | Train | Val / Test |
|------|-------|------------|
| Resize | 224×224 | 224×224 |
| Normalize | `/255.0` → `[0,1]` | `/255.0` → `[0,1]` |
| Backbone preprocess | `preprocess_input()` | `preprocess_input()` |

> Each backbone has its own `preprocess_input` (MobileNetV2 scales to `[-1,1]`; ResNet50V2 does channel-wise mean subtraction). Apply **after** resize, **before** augmentation on val/test.

### 3.3 Data Augmentation (train only)

```python
augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomBrightness(0.1),
    tf.keras.layers.RandomContrast(0.1),
])
```

### 3.4 Pipeline Assembly

```python
BATCH_SIZE = 32
AUTOTUNE   = tf.data.AUTOTUNE

def build_pipeline(ds, training=False):
    if training:
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    return (ds
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
        .cache())        # cache after first epoch on Colab
```

---

## 4. Model Architecture

### 4.1 Shared Classifier Head

Both backbones use the same head design for a **fair comparison**:

```
Backbone (frozen or partially unfrozen)
    └── GlobalAveragePooling2D
        └── BatchNormalization
            └── Dense(512, activation='relu')
                └── Dropout(0.4)
                    └── Dense(256, activation='relu')
                        └── Dropout(0.3)
                            └── Dense(102, activation='softmax')
```

### 4.2 Model A — MobileNetV2

| Property | Value |
|----------|-------|
| Input shape | (224, 224, 3) |
| Backbone params | ~2.3 M |
| Total trainable (head only) | ~400 K |
| Fine-tune layers | last 30 layers unfrozen in phase 2 |
| Backbone preprocess | `tf.keras.applications.mobilenet_v2.preprocess_input` |

### 4.3 Model B — ResNet50V2

| Property | Value |
|----------|-------|
| Input shape | (224, 224, 3) |
| Backbone params | ~23.5 M |
| Total trainable (head only) | ~400 K |
| Fine-tune layers | last 50 layers unfrozen in phase 2 |
| Backbone preprocess | `tf.keras.applications.resnet_v2.preprocess_input` |

---

## 5. Training Strategy

### 5.1 Two-Phase Transfer Learning

```
Phase 1 — Feature Extraction
─────────────────────────────
  Backbone:  FROZEN (include_top=False, weights='imagenet')
  Train:     classifier head only
  Optimizer: Adam(lr=1e-3)
  Epochs:    15–20
  Goal:      warm up the new head without destroying pretrained weights

Phase 2 — Fine-Tuning
──────────────────────
  Backbone:  top N layers UNFROZEN (see per-model above)
  Train:     unfrozen backbone layers + head
  Optimizer: Adam(lr=1e-5)   ← much lower LR to avoid catastrophic forgetting
  Epochs:    20–30
  Goal:      adapt high-level features to flower-specific patterns
```

### 5.2 Callbacks

```python
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='best_{backbone}.keras',
        monitor='val_macro_f1',
        save_best_only=True,
        mode='max'
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7
    ),
    tf.keras.callbacks.CSVLogger('training_log_{backbone}.csv')
]
```

### 5.3 Loss & Metrics

```python
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)
# Macro F1 tracked separately via sklearn after each epoch (or custom metric)
```

---

## 6. Evaluation Module

### 6.1 Metrics Computed on Test Set

```python
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    classification_report, confusion_matrix
)

metrics = {
    'accuracy':          accuracy_score(y_true, y_pred),
    'macro_precision':   precision_score(y_true, y_pred, average='macro'),
    'macro_recall':      recall_score(y_true, y_pred, average='macro'),
    'macro_f1':          f1_score(y_true, y_pred, average='macro'),   # PRIMARY contest metric
}
```

### 6.2 Outputs

- **Comparison table** — side-by-side metrics for MobileNetV2 vs ResNet50V2
- **Confusion matrix** — 102×102 heatmap (log-scale for readability)
- **Per-class report** — highlight top-5 worst-performing classes
- **Error analysis** — show misclassified samples with true vs predicted label
- **Training curves** — loss and accuracy per epoch for both phases

---

## 7. Experiment Tracking

All experiments are logged to CSV and JSON. Compare across runs:

| Experiment | Backbone | LR Phase1 | LR Phase2 | Augment | Val F1 | Test F1 |
|------------|----------|-----------|-----------|---------|--------|---------|
| baseline   | MobileNetV2 | 1e-3 | — | none | — | — |
| exp-01     | MobileNetV2 | 1e-3 | 1e-5 | standard | — | — |
| exp-02     | ResNet50V2  | 1e-3 | 1e-5 | standard | — | — |
| exp-03     | best model  | 1e-3 | 1e-5 | extended | — | — |

---

## 8. Final Prediction Generation

```python
# Load best saved model
best_model = tf.keras.models.load_model('best_resnet50v2.keras')  # or mobilenetv2

# Run on test set
y_pred = np.argmax(best_model.predict(ds_test), axis=1)

# Save contest submission
pd.DataFrame({'image_id': test_ids, 'label': y_pred}).to_csv(
    'outputs/final_predictions.csv', index=False
)
```

---

## 9. Implementation Notes

### Colab / Kaggle Compatibility
- Use `tfds.load()` — handles download, split, and caching automatically.
- Enable GPU: `Runtime → Change runtime type → T4 GPU`.
- Cache the dataset after the first epoch to avoid re-reading from disk.
- Save checkpoints to Google Drive to survive session resets.

### Fairness Constraints
- Test set is used **only** for final evaluation — never for tuning.
- Both models trained with identical augmentation, batch size, and head design.
- All hyperparameter decisions made using validation set only.

### Class Imbalance
- Oxford Flowers 102 default splits are small (1020 train images, 10 per class).
- Augmentation is especially important given the tiny training set.
- Consider `class_weight` in `model.fit()` if val F1 shows highly skewed per-class performance.

---

## 10. Deliverables Checklist

- [ ] `01_data_exploration.ipynb` — dataset stats, sample images, class distribution
- [ ] `02_mobilenetv2.ipynb` — full training + evaluation of MobileNetV2
- [ ] `03_resnet50v2.ipynb` — full training + evaluation of ResNet50V2
- [ ] `04_evaluation_final.ipynb` — comparison table, confusion matrices, error analysis, final predictions
- [ ] `outputs/final_predictions.csv` — contest submission file
- [ ] `README.md` — setup and run instructions
- [ ] Report (4–6 pages) — problem, data, models, experiments, results, conclusions
- [ ] Presentation (10 min)

---

## 11. Technology Stack

| Component | Library / Tool |
|-----------|---------------|
| Language | Python 3.10+ |
| Deep Learning | TensorFlow 2.x / Keras |
| Dataset | TensorFlow Datasets (`tensorflow-datasets`) |
| Backbones | `tf.keras.applications.MobileNetV2`, `ResNet50V2` |
| Metrics | scikit-learn |
| Visualization | matplotlib, seaborn |
| Experiment logging | CSV + JSON |
| Compute | Google Colab (T4 GPU) or Kaggle (P100) |
