# Presentation Blueprint — Oxford Flowers 102
## 10-Minute In-Class Delivery Guide

**Total slides:** 12  
**Target pace:** ~50 seconds per slide average  
**Tone:** Confident, story-driven, evidence-first  
**Design note:** Dark background (matches flowchart aesthetic), minimal text per slide, one key visual per slide

---

## Delivery Principle

> "Show what we did, show what broke, show how we fixed it, show the number that proves it."

Every slide answers one question. The audience should never have to read — they should listen while you talk and glance at the visual for proof.

---

## Slide-by-Slide Blueprint

---

### SLIDE 1 — Title
**Title on slide:**
> Flower Classification via Transfer Learning
> MobileNetV2 vs. ResNet50V2

**Bottom of slide:**
> Ahmad Jamal Ahmed Jallab · Majdi George Iskandar Jaser
> University of Jordan — Computer Vision, Spring 2026

**Visual:** The system flowchart image (full background, dark overlay)

**You say:**
> "This project is about classifying 102 flower species using transfer learning.
> We ran five versions of experiments, and today we are going to walk you through
> what we learned at each step."

**Time:** 20 seconds

---

### SLIDE 2 — The Problem and the Data
**Title on slide:** The Challenge

**3 bullets maximum:**
- 102 flower categories — fine-grained, visually similar
- Only **10 training images per class** (1,020 total)
- Test set has **6,149 images** — 6× larger than training

**Visual:** Side-by-side: sample flower images from different classes that look nearly identical

**Key message box (bold, large font):**
> The test set is 6× the training set.
> This one fact shaped every decision we made.

**You say:**
> "The dataset has an unusual property — the test split is six times larger than the
> training split. With only 10 images per class, training from scratch is impossible.
> And that size inversion pushed us to question which split we should even be training on."

**Time:** 45 seconds

---

### SLIDE 3 — System Overview
**Title on slide:** How the Pipeline Works

**Visual:** The full system flowchart image (system_flowchart.jpg) — full slide

**No bullets. Just the image.**

**You say:**
> "Here is the complete pipeline. Data comes in, we choose a split strategy,
> preprocess, build both backbones with a shared head, train in two phases,
> evaluate on the validation set, compare both models, and export the final predictions.
> The red box in Phase 2 is the most important part of this whole project —
> we will get to that in a moment."

**Time:** 40 seconds

---

### SLIDE 4 — The Journey Map
**Title on slide:** Five Versions, One Story

**Visual:** Horizontal timeline with 5 stops, each with a short label:

```
v1          v3            v3-02          v4              v5
Baseline  → BN Fix +   → Deeper     → All Fixes    → New Head
0.78 F1     Large Data    Head          0.9453 F1      Architecture
            0.85 F1       0.88 F1                      0.84 F1
```

**Arrow between v1 and v3 labeled:** "Bug found & fixed"
**Arrow into v4 labeled:** "Best result"

**You say:**
> "We ran five experiments. Each one was motivated by what the previous one revealed.
> Version 1 showed us a bug. Version 3 fixed it. Versions 3-02 and 4 refined the design.
> Version 5 tested a different architecture. Let us walk through each step."

**Time:** 35 seconds

---

### SLIDE 5 — Version 1: The Bug
**Title on slide:** V1 — What Broke and Why

**Left side — Training curve image** showing the accuracy collapse at Phase 2 start

**Right side — 3 points:**
- Phase 1 reached val accuracy ≈ 0.77 ✓
- Phase 2 started → accuracy dropped from 0.89 to 0.53 in **one epoch** ✗
- ResNet50V2: EarlyStopping killed fine-tuning after **6 epochs** ✗

**Root cause box (red background):**
> `backbone.trainable = True`
> also unfroze BatchNorm layers →
> running statistics corrupted by 64-image batches

**Result callout:**
> MobileNetV2 F1 = 0.7776 · ResNet50V2 F1 = 0.7521

**You say:**
> "In version 1, Phase 1 worked perfectly. But the moment Phase 2 began,
> accuracy collapsed. The reason: setting backbone trainable also unfroze
> the BatchNorm layers inside it. Those layers started re-estimating their
> statistics from tiny batches — destroying what Phase 1 had built.
> Two lines of code caused this. Two lines of code fixed it."

**Time:** 70 seconds

---

### SLIDE 6 — Version 3: Two Fixes at Once
**Title on slide:** V3 — Fix the Bug, Challenge the Data

**Two columns:**

| Fix 1 — BatchNorm | Fix 2 — Data Strategy |
|---|---|
| Re-freeze all BN layers after partial unfreeze | Use the 6,149-image test split as training data |
| Code snippet (3 lines) | Treat official train (1,020) as held-out eval |

**Code snippet box:**
```python
for layer in backbone.layers:
    if isinstance(layer, BatchNormalization):
        layer.trainable = False
```

**Result callout:**
> MobileNetV2 F1 = 0.8494 · ResNet50V2 F1 = 0.8541
> (+9 points from v1 — no collapse in Phase 2)

**You say:**
> "We made two changes together. First, we fixed the BatchNorm issue with
> an explicit re-freeze loop. Second, we questioned the data: if the test
> set has 6,149 images, why not train on it? So we inverted the split.
> Both models improved significantly and fine-tuning worked smoothly for the first time."

**Time:** 60 seconds

---

### SLIDE 7 — Version 3-02: Deeper Head
**Title on slide:** V3-02 — More Capacity for More Data

**Visual:** Simple diagram of head architecture change:

```
v3 head:       GAP → BN → Dense(224) → Drop → Dense(224) → Output
v3-02 head:    GAP → BN → Dense(224) → Drop → Dense(224) → Drop → Dense(224) → Output
                                        ↑ Added one more Dense layer
```

**Result callout:**
> ResNet50V2 F1 = **0.8778** (best in large-data regime)
> Held-out test on official train split: F1 = **0.8737**

**Key insight box:**
> With 6,149 training images, ResNet50V2's larger capacity justifies a deeper head.

**You say:**
> "With version 3-02, we kept the same inverted data strategy but added a
> third Dense layer to the head. ResNet50V2 improved noticeably — more data
> justifies more capacity in the head. This gave us our best honest test
> result for the large-data approach: 0.8737 macro F1 on completely unseen data."

**Time:** 50 seconds

---

### SLIDE 8 — Version 4: The Best Result
**Title on slide:** V4 — All Lessons, Standard Split

**Large result numbers (hero layout):**

```
MobileNetV2          ResNet50V2
  F1 = 0.9453          F1 = 0.9425
  Acc = 0.9471         Acc = 0.9451
```

**Below — what changed from v1:**

| What | v1 | v4 |
|------|----|----|
| BN in fine-tune | Unfrozen (bug) | Frozen |
| Fine-tune LR | 1e-5 | 1e-4 + ReduceLROnPlateau |
| Unfrozen layers | 20 | 40 / 50 |
| Head | Dense(256) + Drop | BN + 2×Dense(224, L2) |
| Training data | 1,020 | 1,020 |

**You say:**
> "Version 4 is the most important result. We returned to the standard 1,020-image
> training split and applied every lesson from the previous three experiments.
> The result: 0.9453 macro F1 with MobileNetV2 — our best score — using the same
> 1,020 training images as version 1. The bottleneck was never data. It was correct
> fine-tuning."

**Time:** 70 seconds

---

### SLIDE 9 — Version 5: Architecture Experiment
**Title on slide:** V5 — When More Constraints Hurt

**Visual:** Simple architecture diagram of DepthwiseConv2D head vs Dense head

**Two-column comparison:**

| V5 choice | Consequence |
|-----------|------------|
| DepthwiseConv2D head, L2 = 0.1 | Over-regularized even at 6,149 images |
| Phase 2 LR = 1e-5 (same as v1 bug) | Slow convergence, 60 epochs not enough |
| 6,149 training images | Cannot overcome constrained architecture |

**Result callout:**
> MobileNetV2 F1 = 0.8414 — below v3-02 despite more data and more epochs

**Key lesson box:**
> Regularization strength and LR must be co-tuned with data scale.
> More data + wrong hyperparameters = worse result.

**You say:**
> "Version 5 tried a fundamentally different head architecture — a depthwise
> convolution instead of Dense layers — with very aggressive regularization.
> Despite having 6,149 training images and 60 epochs, performance regressed.
> The lesson: it is not enough to have more data. If the architecture is
> over-constrained and the learning rate is too small, more epochs cannot help."

**Time:** 55 seconds

---

### SLIDE 10 — Full Results Table
**Title on slide:** All Versions, Both Models

**Visual:** The complete results table (same as Table V in the report):

| Ver. | Model | Acc | Precision | Recall | F1 |
|------|-------|-----|-----------|--------|----|
| v1 | MobileNetV2 | 0.7765 | 0.8175 | 0.7765 | 0.7776 |
| v1 | ResNet50V2 | 0.7618 | 0.7918 | 0.7618 | 0.7521 |
| v3 | MobileNetV2 | 0.8559 | 0.8817 | 0.8559 | 0.8494 |
| v3 | ResNet50V2 | 0.8559 | 0.8892 | 0.8559 | 0.8541 |
| v3-02 | MobileNetV2 | 0.8618 | 0.8901 | 0.8618 | 0.8570 |
| v3-02 | **ResNet50V2** | **0.8824** | **0.9010** | **0.8824** | **0.8778** |
| **v4** | **MobileNetV2** | **0.9471** | **0.9520** | **0.9471** | **0.9453** |
| v4 | ResNet50V2 | 0.9451 | 0.9504 | 0.9451 | 0.9425 |
| v5 | MobileNetV2 | 0.8422 | 0.8677 | 0.8422 | 0.8414 |
| v5 | ResNet50V2 | 0.7922 | 0.8086 | 0.7922 | 0.7859 |

**Highlight v4 MobileNetV2 row in green**

**You say:**
> "Here is the complete picture across all five versions and both backbones.
> The pattern is clear: MobileNetV2 leads when training data is 1,020 images,
> ResNet50V2 leads when training data is 6,149 images.
> And version 4, despite using the smallest dataset, delivers the highest F1
> of 0.9453 — because the implementation was finally correct."

**Time:** 50 seconds

---

### SLIDE 11 — Confusion Matrix and Error Analysis
**Title on slide:** Where the Model Fails

**Left side:** Confusion matrix image (v4-output/confusion_matrix_ResNet50V4.png)

**Right side — 3 observations:**
- Strong diagonal across all 102 classes → high per-class accuracy
- Off-diagonal clusters → visually similar species confused with each other
- No completely blank row → model learned something about every class
- Low-count test classes (20 images) show higher confusion than high-count classes (238 images)

**Key insight box:**
> Errors are not random — they concentrate between classes that share petal color or arrangement. This is the expected failure mode of fine-grained classification.

**You say:**
> "The confusion matrix shows that errors are not scattered randomly. They cluster
> between specific pairs of flower species that look similar — same color family,
> same petal arrangement. This is expected in fine-grained classification. What is
> notable is that no class is entirely missed: even with 10 training images, the
> model found discriminative features for every one of the 102 categories."

**Time:** 55 seconds

---

### SLIDE 12 — Conclusion and Final Model
**Title on slide:** What We Learned

**Left side — 4 key findings (large font, numbered):**

1. Fix the implementation first — the BN bug cost 17 F1 points
2. MobileNetV2 wins at 1,020 images; ResNet50V2 wins at 6,149
3. LR scheduling (ReduceLROnPlateau) produced the steepest gains
4. Regularization must match data scale — more data needs less constraint

**Right side — Final model box (green border):**
```
BEST MODEL
MobileNetV2 — Version 4

Accuracy    = 0.9471
Precision   = 0.9520
Recall      = 0.9471
Macro F1    = 0.9453

Trained on: 1,020 images
Predictions: final_predictions.csv
```

**You say:**
> "The main lesson of this project is that correct implementation matters more
> than data volume. Version 4 — trained on the same 1,020 images as version 1 —
> reaches 0.9453 F1, while version 1 with the bug reached only 0.7776.
> Our final submission is MobileNetV2 under the version 4 configuration.
> Thank you."

**Time:** 50 seconds

---

## Timing Summary

| Slide | Topic | Time |
|-------|-------|------|
| 1 | Title | 0:20 |
| 2 | Problem and Data | 0:45 |
| 3 | System Overview | 0:40 |
| 4 | Journey Map | 0:35 |
| 5 | V1 — The Bug | 1:10 |
| 6 | V3 — Two Fixes | 1:00 |
| 7 | V3-02 — Deeper Head | 0:50 |
| 8 | V4 — Best Result | 1:10 |
| 9 | V5 — Architecture Experiment | 0:55 |
| 10 | Full Results Table | 0:50 |
| 11 | Confusion Matrix | 0:55 |
| 12 | Conclusion + Final Model | 0:50 |
| **Total** | | **~10:00** |

---

## Design Guidelines

### Layout rules
- Maximum **5 bullet points** per slide — prefer 3
- One **hero visual** per slide (chart, image, code block, or table)
- Key numbers always in **large font or colored callout box**
- Code snippets only on slides 6 — keep them to 3–5 lines

### Color palette (matching the flowchart)
- Background: `#1a1a2e` (dark navy) or `#0d0d0d` (near black)
- Primary text: `#ffffff`
- Accent / headings: `#e9c46a` (gold)
- Critical fix highlight: `#c9184a` (red) — used only for the BN fix
- Best result highlight: `#2d6a4f` (green) — used only for v4 result
- Neutral boxes: `#2c2c54`

### Slide transitions
- Simple fade between slides — no animations within slides
- The journey map slide (Slide 4) can use a **left-to-right appear** on each version stop

### Fonts
- Headings: Bold sans-serif (Montserrat, Inter, or Arial Bold)
- Body: Regular weight same family
- Code: Monospace (Consolas or Courier New)

---

## What to Avoid

| Avoid | Do instead |
|-------|-----------|
| Reading from the slide | Put only keywords on slide, speak the explanation |
| Paragraphs of text | 3–5 short bullets maximum |
| Showing all results before context | Follow the version order — results come after the story |
| Apologizing for v5 regression | Frame it as a controlled experiment with a clear lesson |
| Skipping the BN fix explanation | This is your strongest technical talking point — spend time on it |
