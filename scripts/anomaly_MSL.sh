#!/bin/bash

# MSL Anomaly Detection using FusAD
python FusAD.py \
  --task anomaly_detection \
  --data_path MSL \
  --root_path data/MSL \
  --data_file MSL_train.csv \
  --seq_len 100 \
  --emb_dim 128 \
  --depth 3 \
  --patch_size 8 \
  --batch_size 32 \
  --train_epochs 10 \
  --learning_rate 0.0001 \
  --anomaly_ratio 2.0 \
  --enc_in 55 \
  --ASM True \
  --IFM True \
  --adaptive_filter True \
  --seed 42
