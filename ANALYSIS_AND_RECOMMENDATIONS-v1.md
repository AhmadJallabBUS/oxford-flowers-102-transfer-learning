# Analysis & Recommendations — Oxford Flowers 102

This document explains the **v1 training results**, why they happened, and the
**v2 changes** made to the notebook to improve performance.

---

## 1. v1 Results (what we got)

| Model | Val Accuracy | Macro Precision | Macro Recall | Macro F1 |
|-------|:---:|:---:|:---:|:---:|
| **MobileNetV2** | 0.7765 | 0.8175 | 0.7765 | **0.7776** |
| ResNet50V2 | 0.7618 | 0.7918 | 0.7618 | 0.7521 |

**Best model: MobileNetV2** (higher F1 *and* accuracy, and ~3× faster to train).

> Note: a larger network (ResNet50V2, 23.5M params) lost to a smaller one
> (MobileNetV2, 2.3M params). That is expected here — see Problem 3.

---

## 2. Reading the two training curves

**MobileNetV2 curve** — Phase 1 climbs smoothly to val ≈ 0.77. At the fine-tune
line (epoch 15) **train accuracy collapses from 0.89 to 0.53**, then slowly
re-climbs while **val accuracy stays flat at ~0.77**. Fine-tuning added almost
nothing (+1.2%).

**ResNet50V2 curve** — Same Phase-1 shape (val ≈ 0.76). At fine-tune, train drops
to ~0.65 and **validation loss immediately rises** (0.92 → 0.97), so
EarlyStopping killed it at phase-2 epoch 6. Fine-tuning **hurt** this model.

The shared symptom — a sharp drop exactly when Phase 2 begins — is the key clue.

---

## 3. Root-cause diagnosis

### Problem 1 — BatchNorm was unfrozen during fine-tuning (the main bug)
When Phase 2 ran `backbone.trainable = True`, it also un-froze the **BatchNorm**
layers inside the backbone. BatchNorm then started updating its running
mean/variance from our **tiny 32-image batches**. Those corrupted statistics no
longer matched what the classifier head was trained on → the accuracy crash you
see at the fine-tune line. This is the textbook transfer-learning mistake.

**Fix (v2):** keep every `BatchNormalization` layer frozen during Phase 2:
```python
backbone.trainable = True
for layer in backbone.layers[:-fine_tune_layers]:
    layer.trainable = False
for layer in backbone.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False   # <- prevents the crash
```

### Problem 2 — Fine-tuning was too timid and unstable
Phase-2 used LR = 1e-5 and only the **last 20** layers. Combined with the BN
corruption, the model could not make useful progress. Once BN is frozen
(Problem 1), it is safe to learn a bit faster and adapt more layers.

**Fix (v2):** LR = 1e-4 (10× higher) and unfreeze **40 (MobileNet) / 50 (ResNet)**
top layers.

### Problem 3 — Overfitting caused by data scarcity (the real ceiling)
Train accuracy reached **0.90** while validation sat at **0.77** — a ~13% gap =
overfitting. The cause is structural: the official Flowers102 **train split has
only 1020 images = 10 per class** for 102 classes. With so few examples, a big
model (ResNet50V2) memorizes faster and generalizes worse, which is exactly why
MobileNetV2 won.

**Fixes (v2):**
- **Stronger augmentation** (added `RandomContrast`, `RandomTranslation`, larger
  rotation/zoom) — multiplies the effective number of training views.
- **More regularization in the head** (`BatchNormalization` + two `Dropout`
  layers + L2 weight decay) — shrinks the train/val gap.
- **Optional: train the final model on train+val merged** (~20 imgs/class). This
  is allowed (test set stays untouched) and is usually the single biggest F1
  gain. Added as optional Section 11 in the notebook.

### Problem 4 — Noisy checkpoint metric
v1 saved the "best" model on `val_accuracy`, which is jumpy and was plateaued.

**Fix (v2):** monitor `val_loss` for both EarlyStopping and best-weight
restoration, and add `ReduceLROnPlateau` so the LR drops when val_loss stalls
(the Phase-1 plateau from epoch 8–15 shows this was needed).

---

## 4. Summary of v1 → v2 changes

| Area | v1 | v2 |
|------|----|----|
| BatchNorm in fine-tune | Unfrozen (bug) | **Frozen** |
| Fine-tune LR | 1e-5 | **1e-4** |
| Unfrozen layers | last 20 | **40 / 50** |
| Augmentation | flip, rotate, zoom | **+ contrast, translation, stronger** |
| Head | Dense+Dropout(0.4) | **BN + Dropout(0.3) + Dense(L2) + Dropout(0.4)** |
| Callbacks | EarlyStop, Checkpoint(val_acc) | **EarlyStop + ReduceLROnPlateau (val_loss)** |
| Phase-1 epochs | 15 | 20 |
| Extra | — | **Optional train+val retrain (Section 11)** |

---

## 5. What to expect after v2

- The fine-tune **crash should disappear** — the curve should rise or stay flat
  at the fine-tune line instead of dropping.
- Fine-tuning should now **add a few real points** of val F1 (especially for
  ResNet50V2, which previously regressed).
- The train/val gap should **shrink** thanks to stronger regularization.
- Realistic target on the **test set**: validation here (~0.78) is on the small
  1020-image val split; the 6149-image test split typically lands in a similar
  0.75–0.85 F1 band. Merging train+val (Section 11) is the best lever to push
  toward the top of that range.

---

## 6. Recommendation for the report / contest

1. **Submit MobileNetV2 as the best model** — it beat ResNet50V2 on F1 and
   accuracy and trains far faster. Good talking point: *on a 10-image/class
   dataset, the smaller backbone generalized better.*
2. **Show both training curves** (v1 vs v2) to demonstrate you diagnosed and
   fixed the BatchNorm crash — this is exactly the "error analysis" the rubric
   rewards (3 pts for evaluation & analysis).
3. **For the final contest run, set `RUN_COMBINED = True`** to retrain on
   train+val, then report the test-set F1 from that model.
4. In the error-analysis section, mention that most confusion is between
   **visually similar flower classes** (similar color/shape) — inspect the
   off-diagonal cells of the confusion matrix to name a few specific pairs.
