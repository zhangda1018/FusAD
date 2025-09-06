for len in 96 192 336 720
do
  python -u FusAD.py --task forecasting \
  --root_path /datasets/Forecasting/ETT-small \
  --pred_len $len \
  --data ETTm2 \
  --data_path ETTm2.csv \
  --seq_len 512 \
  --emb_dim 128 \
  --depth 1 \
  --batch_size 512 \
  --dropout 0.5 \
  --patch_size 64 \
  --train_epochs 20 \
  --pretrain_epochs 10
done
