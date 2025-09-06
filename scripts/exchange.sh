for len in 96 192 336 720
do
  python -u FusAD.py --task forecasting \
  --root_path /datasets/Forecasting/exchange_rate \
  --pred_len $len \
  --data custom \
  --data_path exchange_rate.csv \
  --seq_len 64 \
  --emb_dim 64 \
  --depth 3 \
  --batch_size 64 \
  --dropout 0.5 \
  --patch_size 64 \
  --train_epochs 20 \
  --pretrain_epochs 10
done
