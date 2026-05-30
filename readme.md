 10 sections, run top to bottom:

  1. Install & import
  2. Load dataset (tfds)
  3. Data pipeline (resize + augment)
  4. build_model() — shared for both backbones
  5. train_and_evaluate() — phase 1 (freeze) + phase 2 (fine-tune) + metrics
  6. Train MobileNetV2
  7. Train ResNet50V2
  8. Compare both in a table
  9. Confusion matrix
  10. Test set metrics + save final_predictions.csv