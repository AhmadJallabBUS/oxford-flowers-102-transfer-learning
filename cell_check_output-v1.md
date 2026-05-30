 WARNING:absl:Variant folder /root/tensorflow_datasets/oxford_flowers102/2.1.1 has no dataset_info.json
Downloading and preparing dataset Unknown size (download: Unknown size, generated: Unknown size, total: Unknown size) to /root/tensorflow_datasets/oxford_flowers102/2.1.1...
Dl Completed...: 100% 3/3 [00:53<00:00,  9.17s/ url]Dl Size...: 100% 328/328 [00:53<00:00, 15.00 MiB/s]Extraction completed...: 100% 8189/8189 [00:53<00:00, 453.14 file/s]Generating splits...: 100% 3/3 [00:05<00:00,  1.78s/ splits]Generating train examples...:  0/? [00:00<?, ? examples/s]Shuffling /root/tensorflow_datasets/oxford_flowers102/incomplete.6MP7AZ_2.1.1/oxford_flowers102-train.tfrecord-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9]...:   0% 0/1020 [00:00<?, ? examples/s]Generating test examples...:  4788/? [00:02<00:00, 2321.79 examples/s]Shuffling /root/tensorflow_datasets/oxford_flowers102/incomplete.6MP7AZ_2.1.1/oxford_flowers102-test.tfrecord-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9]...:   0% 0/6149 [00:00<?, ? examples/s]Generating validation examples...:  0/? [00:00<?, ? examples/s]Shuffling /root/tensorflow_datasets/oxford_flowers102/incomplete.6MP7AZ_2.1.1/oxford_flowers102-validation.tfrecord-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9]...:   0% 0/1020 [00:00<?, ? examples/s]Dataset oxford_flowers102 downloaded and prepared to /root/tensorflow_datasets/oxford_flowers102/2.1.1. Subsequent calls will reuse this data.
Train: 1020
Val:   1020
Test:  6149
---
  
  Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5
9406464/9406464 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step

===============================
  MobileNetV2
===============================
Phase 1: head training...
Epoch 1/15
32/32 ━━━━━━━━━━━━━━━━━━━━ 161s 5s/step - accuracy: 0.0382 - loss: 4.5908 - val_accuracy: 0.1686 - val_loss: 4.0702
Epoch 2/15
32/32 ━━━━━━━━━━━━━━━━━━━━ 105s 3s/step - accuracy: 0.1833 - loss: 3.6646 - val_accuracy: 0.3892 - val_loss: 3.0689
Epoch 3/15
32/32 ━━━━━━━━━━━━━━━━━━━━ 142s 3s/step - accuracy: 0.3843 - loss: 2.6615 - val_accuracy: 0.5431 - val_loss: 2.2731
Epoch 4/15
32/32 ━━━━━━━━━━━━━━━━━━━━ 140s 4s/step - accuracy: 0.5078 - loss: 2.0061 - val_accuracy: 0.6029 - val_loss: 1.8047
Epoch 5/15
32/32 ━━━━━━━━━━━━━━━━━━━━ 115s 4s/step - accuracy: 0.5892 - loss: 1.6026 - val_accuracy: 0.6706 - val_loss: 1.4976
Epoch 6/15
32/32 ━━━━━━━━━━━━━━━━━━━━ 103s 3s/step - accuracy: 0.6549 - loss: 1.3135 - val_accuracy: 0.6598 - val_loss: 1.4151
Epoch 7/15
32/32 ━━━━━━━━━━━━━━━━━━━━ 142s 4s/step - accuracy: 0.7363 - loss: 1.0328 - val_accuracy: 0.7069 - val_loss: 1.2365
Epoch 8/15
32/32 ━━━━━━━━━━━━━━━━━━━━ 104s 3s/step - accuracy: 0.7676 - loss: 0.9313 - val_accuracy: 0.7353 - val_loss: 1.1505
Epoch 9/15
32/32 ━━━━━━━━━━━━━━━━━━━━ 142s 3s/step - accuracy: 0.7676 - loss: 0.8794 - val_accuracy: 0.7402 - val_loss: 1.0767
Epoch 10/15
32/32 ━━━━━━━━━━━━━━━━━━━━ 178s 4s/step - accuracy: 0.8078 - loss: 0.7082 - val_accuracy: 0.7529 - val_loss: 1.0101
Epoch 11/15
32/32 ━━━━━━━━━━━━━━━━━━━━ 101s 3s/step - accuracy: 0.8255 - loss: 0.6436 - val_accuracy: 0.7490 - val_loss: 1.0229
Epoch 12/15
32/32 ━━━━━━━━━━━━━━━━━━━━ 99s 3s/step - accuracy: 0.8412 - loss: 0.5638 - val_accuracy: 0.7412 - val_loss: 0.9816
Epoch 13/15
32/32 ━━━━━━━━━━━━━━━━━━━━ 102s 3s/step - accuracy: 0.8706 - loss: 0.5315 - val_accuracy: 0.7686 - val_loss: 0.9229
Epoch 14/15
32/32 ━━━━━━━━━━━━━━━━━━━━ 104s 3s/step - accuracy: 0.8824 - loss: 0.4937 - val_accuracy: 0.7745 - val_loss: 0.9129
Epoch 15/15
32/32 ━━━━━━━━━━━━━━━━━━━━ 151s 3s/step - accuracy: 0.8902 - loss: 0.4409 - val_accuracy: 0.7647 - val_loss: 0.8715
Phase 2: fine-tuning...
Epoch 1/20
32/32 ━━━━━━━━━━━━━━━━━━━━ 130s 4s/step - accuracy: 0.5373 - loss: 1.8040 - val_accuracy: 0.7735 - val_loss: 0.8584
Epoch 2/20
32/32 ━━━━━━━━━━━━━━━━━━━━ 138s 4s/step - accuracy: 0.5745 - loss: 1.5662 - val_accuracy: 0.7745 - val_loss: 0.8538
Epoch 3/20
32/32 ━━━━━━━━━━━━━━━━━━━━ 107s 3s/step - accuracy: 0.6569 - loss: 1.2765 - val_accuracy: 0.7706 - val_loss: 0.8564
Epoch 4/20
32/32 ━━━━━━━━━━━━━━━━━━━━ 146s 3s/step - accuracy: 0.7069 - loss: 1.1452 - val_accuracy: 0.7755 - val_loss: 0.8539
Epoch 5/20
32/32 ━━━━━━━━━━━━━━━━━━━━ 150s 5s/step - accuracy: 0.7010 - loss: 1.0694 - val_accuracy: 0.7745 - val_loss: 0.8490
Epoch 6/20
32/32 ━━━━━━━━━━━━━━━━━━━━ 109s 3s/step - accuracy: 0.7392 - loss: 0.9653 - val_accuracy: 0.7735 - val_loss: 0.8521
Epoch 7/20
32/32 ━━━━━━━━━━━━━━━━━━━━ 115s 4s/step - accuracy: 0.7686 - loss: 0.8724 - val_accuracy: 0.7735 - val_loss: 0.8499
Epoch 8/20
32/32 ━━━━━━━━━━━━━━━━━━━━ 112s 3s/step - accuracy: 0.7784 - loss: 0.8291 - val_accuracy: 0.7686 - val_loss: 0.8468
Epoch 9/20
32/32 ━━━━━━━━━━━━━━━━━━━━ 109s 3s/step - accuracy: 0.8245 - loss: 0.7311 - val_accuracy: 0.7676 - val_loss: 0.8465
Epoch 10/20
32/32 ━━━━━━━━━━━━━━━━━━━━ 146s 3s/step - accuracy: 0.8049 - loss: 0.7466 - val_accuracy: 0.7696 - val_loss: 0.8432
Epoch 11/20
32/32 ━━━━━━━━━━━━━━━━━━━━ 110s 3s/step - accuracy: 0.8294 - loss: 0.6927 - val_accuracy: 0.7686 - val_loss: 0.8391
Epoch 12/20
32/32 ━━━━━━━━━━━━━━━━━━━━ 182s 5s/step - accuracy: 0.8500 - loss: 0.6420 - val_accuracy: 0.7735 - val_loss: 0.8338
Epoch 13/20
32/32 ━━━━━━━━━━━━━━━━━━━━ 110s 3s/step - accuracy: 0.8471 - loss: 0.6293 - val_accuracy: 0.7696 - val_loss: 0.8293
Epoch 14/20
32/32 ━━━━━━━━━━━━━━━━━━━━ 110s 3s/step - accuracy: 0.8471 - loss: 0.6044 - val_accuracy: 0.7667 - val_loss: 0.8271
Epoch 15/20
32/32 ━━━━━━━━━━━━━━━━━━━━ 119s 4s/step - accuracy: 0.8647 - loss: 0.5630 - val_accuracy: 0.7686 - val_loss: 0.8228
Epoch 16/20
32/32 ━━━━━━━━━━━━━━━━━━━━ 106s 3s/step - accuracy: 0.8745 - loss: 0.5337 - val_accuracy: 0.7696 - val_loss: 0.8132
Epoch 17/20
32/32 ━━━━━━━━━━━━━━━━━━━━ 142s 3s/step - accuracy: 0.8588 - loss: 0.5211 - val_accuracy: 0.7676 - val_loss: 0.8068
Epoch 18/20
32/32 ━━━━━━━━━━━━━━━━━━━━ 146s 3s/step - accuracy: 0.8637 - loss: 0.5049 - val_accuracy: 0.7706 - val_loss: 0.8001
Epoch 19/20
32/32 ━━━━━━━━━━━━━━━━━━━━ 111s 3s/step - accuracy: 0.8725 - loss: 0.5189 - val_accuracy: 0.7725 - val_loss: 0.7951
Epoch 20/20
32/32 ━━━━━━━━━━━━━━━━━━━━ 153s 3s/step - accuracy: 0.9039 - loss: 0.4487 - val_accuracy: 0.7765 - val_loss: 0.7900

Val Results for MobileNetV2:
  accuracy  : 0.7765
  precision : 0.8175
  recall    : 0.7765
  f1        : 0.7776

 Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5
94668760/94668760 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step

===============================
  ResNet50V2
===============================
Phase 1: head training...
Epoch 1/15
32/32 ━━━━━━━━━━━━━━━━━━━━ 410s 13s/step - accuracy: 0.0686 - loss: 4.4890 - val_accuracy: 0.2284 - val_loss: 3.6411
Epoch 2/15
32/32 ━━━━━━━━━━━━━━━━━━━━ 379s 12s/step - accuracy: 0.2657 - loss: 3.2028 - val_accuracy: 0.4392 - val_loss: 2.5476
Epoch 3/15
32/32 ━━━━━━━━━━━━━━━━━━━━ 384s 12s/step - accuracy: 0.4373 - loss: 2.3067 - val_accuracy: 0.5510 - val_loss: 1.9542
Epoch 4/15
32/32 ━━━━━━━━━━━━━━━━━━━━ 394s 12s/step - accuracy: 0.5529 - loss: 1.7544 - val_accuracy: 0.6235 - val_loss: 1.6320
Epoch 5/15
32/32 ━━━━━━━━━━━━━━━━━━━━ 377s 12s/step - accuracy: 0.6235 - loss: 1.4459 - val_accuracy: 0.6657 - val_loss: 1.3938
Epoch 6/15
32/32 ━━━━━━━━━━━━━━━━━━━━ 370s 12s/step - accuracy: 0.6951 - loss: 1.1394 - val_accuracy: 0.6765 - val_loss: 1.2620
Epoch 7/15
32/32 ━━━━━━━━━━━━━━━━━━━━ 369s 12s/step - accuracy: 0.7471 - loss: 0.9930 - val_accuracy: 0.6961 - val_loss: 1.1991
Epoch 8/15
32/32 ━━━━━━━━━━━━━━━━━━━━ 385s 12s/step - accuracy: 0.7637 - loss: 0.8634 - val_accuracy: 0.7020 - val_loss: 1.1259
Epoch 9/15
32/32 ━━━━━━━━━━━━━━━━━━━━ 403s 12s/step - accuracy: 0.7667 - loss: 0.8244 - val_accuracy: 0.7225 - val_loss: 1.0791
Epoch 10/15
32/32 ━━━━━━━━━━━━━━━━━━━━ 367s 12s/step - accuracy: 0.8196 - loss: 0.6510 - val_accuracy: 0.7147 - val_loss: 1.0583
Epoch 11/15
32/32 ━━━━━━━━━━━━━━━━━━━━ 385s 12s/step - accuracy: 0.8284 - loss: 0.6351 - val_accuracy: 0.7373 - val_loss: 1.0009
Epoch 12/15
32/32 ━━━━━━━━━━━━━━━━━━━━ 381s 12s/step - accuracy: 0.8559 - loss: 0.5378 - val_accuracy: 0.7549 - val_loss: 0.9445
Epoch 13/15
32/32 ━━━━━━━━━━━━━━━━━━━━ 405s 12s/step - accuracy: 0.8627 - loss: 0.5039 - val_accuracy: 0.7382 - val_loss: 0.9588
Epoch 14/15
32/32 ━━━━━━━━━━━━━━━━━━━━ 393s 12s/step - accuracy: 0.8853 - loss: 0.4317 - val_accuracy: 0.7647 - val_loss: 0.8907
Epoch 15/15
32/32 ━━━━━━━━━━━━━━━━━━━━ 393s 12s/step - accuracy: 0.8882 - loss: 0.3923 - val_accuracy: 0.7637 - val_loss: 0.9228
Phase 2: fine-tuning...
Epoch 1/20
32/32 ━━━━━━━━━━━━━━━━━━━━ 435s 13s/step - accuracy: 0.6500 - loss: 1.3407 - val_accuracy: 0.7598 - val_loss: 0.9437
Epoch 2/20
32/32 ━━━━━━━━━━━━━━━━━━━━ 445s 13s/step - accuracy: 0.7088 - loss: 1.1365 - val_accuracy: 0.7578 - val_loss: 0.9803
Epoch 3/20
32/32 ━━━━━━━━━━━━━━━━━━━━ 434s 13s/step - accuracy: 0.7853 - loss: 0.9051 - val_accuracy: 0.7559 - val_loss: 0.9898
Epoch 4/20
32/32 ━━━━━━━━━━━━━━━━━━━━ 443s 14s/step - accuracy: 0.7912 - loss: 0.8537 - val_accuracy: 0.7549 - val_loss: 0.9879
Epoch 5/20
32/32 ━━━━━━━━━━━━━━━━━━━━ 418s 13s/step - accuracy: 0.8402 - loss: 0.7124 - val_accuracy: 0.7529 - val_loss: 0.9842
Epoch 6/20
32/32 ━━━━━━━━━━━━━━━━━━━━ 444s 13s/step - accuracy: 0.8373 - loss: 0.6786 - val_accuracy: 0.7618 - val_loss: 0.9712

Val Results for ResNet50V2:
  accuracy  : 0.7618
  precision : 0.7918
  recall    : 0.7618
  f1        : 0.7521
