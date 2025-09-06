for len in 96 192 336 720
do
  python -u FusAD.py --task forecasting \
  --root_path /datasets/Forecasting/traffic \
  --pred_len $len \
  --data custom \
  --data_path traffic.csv \
  --seq_len 512 \
  --emb_dim 128 \
  --depth 2 \
  --batch_size 16 \
  --dropout 0.5 \
  --patch_size 32 \
  --train_epochs 10 \
  --pretrain_epochs 10
done
