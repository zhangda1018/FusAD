#!/bin/bash

# PSM Anomaly Detection using FusAD
python FusAD.py \
  --task anomaly_detection \
  --data_path PSM \
  --root_path data/PSM \
  --data_file PSM_train.csv \
  --seq_len 100 \
  --emb_dim 128 \
  --depth 3 \
  --patch_size 8 \
  --batch_size 32 \
  --train_epochs 10 \
  --learning_rate 0.0001 \
  --anomaly_ratio 1.0 \
  --enc_in 25 \
  --ASM True \
  --IFM True \
  --adaptive_filter True \
  --seed 42
