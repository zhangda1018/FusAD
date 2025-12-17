for len in 96 192 336 720
do
  python -u FusAD_Forecasting.py \
  --root_path /gemini/space/yifq/zhaozy/zhangda/code/TSLANet-main/electricity \
  --pred_len $len \
  --data custom \
  --data_path electricity.csv \
  --seq_len 512 \
  --emb_dim 64 \
  --depth 3 \
  --batch_size 64 \
  --dropout 0.5 \
  --patch_size 32 \
  --train_epochs 1 \
  --pretrain_epochs 1 \
  --ASM True \
  --IFM True \
  --adaptive_filter True
done
