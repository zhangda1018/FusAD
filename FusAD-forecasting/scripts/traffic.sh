for len in 96 192 336 720
do
  python -u FusAD_Forecasting.py \
  --root_path ./data/traffic \
  --pred_len $len \
  --data custom \
  --data_path traffic.csv \
  --seq_len 512 \
  --emb_dim 64 \
  --depth 3 \
  --batch_size 16 \
  --dropout 0.5 \
  --patch_size 32 \
  --train_epochs 20 \
  --pretrain_epochs 10 \
  --ASM True \
  --IFM True \
  --adaptive_filter True
done
