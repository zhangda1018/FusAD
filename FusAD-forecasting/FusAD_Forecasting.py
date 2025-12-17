"""
FusAD-Forecasting: Joint Adaptive Frequency-Aware Network for Time Series Forecasting
基于FusAD分类模型改进，适配时间序列预测任务
"""
import argparse
import datetime
import os

import lightning as L
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError

from data_factory import data_provider
from utils import save_copy_of_files, random_masking_3D, str2bool
from Component.ASM import Adaptive_Spectral_Module
from Component.IFM import IFM


class FusAD_layer(L.LightningModule):
    """
    FusAD层：结合自适应频谱模块(ASM)和交互融合模块(IFM)
    """
    def __init__(self, dim, mlp_ratio=3., drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 use_IFM=True, use_ASM=True, adaptive_filter=True):
        super().__init__()
        self.use_IFM = use_IFM
        self.use_ASM = use_ASM
        
        self.norm1 = norm_layer(dim)
        self.asm = Adaptive_Spectral_Module(dim, drop=drop, adaptive_filter=adaptive_filter)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ifm = IFM(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        if self.use_IFM and self.use_ASM:
            x = x + self.drop_path(self.ifm(self.norm2(self.asm(self.norm1(x)))))
        elif self.use_IFM:
            x = x + self.drop_path(self.ifm(self.norm2(x)))
        elif self.use_ASM:
            x = x + self.drop_path(self.asm(self.norm1(x)))
        return x


class FusAD(nn.Module):
    """
    FusAD主模型：用于时间序列预测
    """
    def __init__(self, args):
        super(FusAD, self).__init__()
        self.args = args
        self.patch_size = args.patch_size
        self.stride = self.patch_size // 2
        num_patches = int((args.seq_len - self.patch_size) / self.stride + 1)

        # 输入嵌入层
        self.input_layer = nn.Linear(self.patch_size, args.emb_dim)

        # 随机深度衰减规则
        dpr = [x.item() for x in torch.linspace(0, args.dropout, args.depth)]

        # FusAD层堆叠
        self.jafa_blocks = nn.ModuleList([
            FusAD_layer(
                dim=args.emb_dim, 
                drop=args.dropout, 
                drop_path=dpr[i],
                use_IFM=args.IFM,
                use_ASM=args.ASM,
                adaptive_filter=args.adaptive_filter
            )
            for i in range(args.depth)]
        )

        # 输出层
        self.out_layer = nn.Linear(args.emb_dim * num_patches, args.pred_len)

    def pretrain(self, x_in):
        """预训练前向传播（掩码重建）"""
        x = rearrange(x_in, 'b l m -> b m l')
        x_patched = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x_patched = rearrange(x_patched, 'b m n p -> (b m) n p')

        xb_mask, _, self.mask, _ = random_masking_3D(x_patched, mask_ratio=self.args.mask_ratio)
        self.mask = self.mask.bool()
        xb_mask = self.input_layer(xb_mask)

        for jafa_blk in self.jafa_blocks:
            xb_mask = jafa_blk(xb_mask)

        return xb_mask, self.input_layer(x_patched)

    def forward(self, x):
        """正常前向传播（预测）"""
        B, L, M = x.shape

        # 实例归一化
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x /= stdev

        # Patch嵌入
        x = rearrange(x, 'b l m -> b m l')
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')
        x = self.input_layer(x)

        # FusAD层处理
        for jafa_blk in self.jafa_blocks:
            x = jafa_blk(x)

        # 输出映射
        outputs = self.out_layer(x.reshape(B * M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        # 反归一化
        outputs = outputs * stdev
        outputs = outputs + means

        return outputs


class model_pretraining(L.LightningModule):
    """预训练模型包装类"""
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.model = FusAD(args)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-6)
        return optimizer

    def _calculate_loss(self, batch, mode="train"):
        batch_x, batch_y, _, _ = batch
        _, _, C = batch_x.shape
        batch_x = batch_x.float().to(self.device)

        preds, target = self.model.pretrain(batch_x)

        loss = (preds - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * self.model.mask).sum() / self.model.mask.sum()

        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")


class model_training(L.LightningModule):
    """训练模型包装类"""
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.model = FusAD(args)
        self.criterion = nn.MSELoss()
        self.mse = MeanSquaredError()
        self.mae = MeanAbsoluteError()
        self.preds = []
        self.trues = []

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-6)
        scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, patience=2, verbose=True
            ),
            'monitor': 'val_mse',
            'interval': 'epoch',
            'frequency': 1
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def _calculate_loss(self, batch, mode="train"):
        batch_x, batch_y, _, _ = batch
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        outputs = self.model(batch_x)
        outputs = outputs[:, -self.args.pred_len:, :]
        batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
        loss = self.criterion(outputs, batch_y)

        pred = outputs.detach().cpu()
        true = batch_y.detach().cpu()

        mse = self.mse(pred.contiguous(), true.contiguous())
        mae = self.mae(pred.contiguous(), true.contiguous())

        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_mse", mse, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_mae", mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss, pred, true

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        loss, preds, trues = self._calculate_loss(batch, mode="test")
        self.preds.append(preds)
        self.trues.append(trues)
        return {'test_loss': loss, 'pred': preds, 'true': trues}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)

    def on_test_epoch_end(self):
        preds = torch.cat(self.preds)
        trues = torch.cat(self.trues)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mse = self.mse(preds.contiguous(), trues.contiguous())
        mae = self.mae(preds.contiguous(), trues.contiguous())
        print(f"Final Test Results - MAE: {mae}, MSE: {mse}")


def pretrain_model(args, train_loader, val_loader, checkpoint_path, pretrain_checkpoint_callback):
    """执行预训练"""
    PRETRAIN_MAX_EPOCHS = args.pretrain_epochs
    trainer = L.Trainer(
        default_root_dir=checkpoint_path,
        accelerator="auto",
        devices=1,
        num_sanity_val_steps=0,
        max_epochs=PRETRAIN_MAX_EPOCHS,
        callbacks=[
            pretrain_checkpoint_callback,
            LearningRateMonitor("epoch"),
            TQDMProgressBar(refresh_rate=500)
        ],
    )
    trainer.logger._log_graph = False
    trainer.logger._default_hp_metric = None

    L.seed_everything(args.seed)
    model = model_pretraining(args)
    trainer.fit(model, train_loader, val_loader)

    return model, pretrain_checkpoint_callback.best_model_path


def train_model(args, train_loader, val_loader, test_loader, checkpoint_path, 
                checkpoint_callback, pretrained_model_path):
    """执行训练"""
    trainer = L.Trainer(
        default_root_dir=checkpoint_path,
        accelerator="auto",
        num_sanity_val_steps=0,
        devices=1,
        max_epochs=args.train_epochs,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor("epoch"),
            TQDMProgressBar(refresh_rate=500)
        ],
    )
    trainer.logger._log_graph = False
    trainer.logger._default_hp_metric = None

    L.seed_everything(args.seed)
    if args.load_from_pretrained and pretrained_model_path:
        model = model_training.load_from_checkpoint(pretrained_model_path, args=args)
    else:
        model = model_training(args)
    trainer.fit(model, train_loader, val_loader)

    # 加载最佳检查点
    model = model_training.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, args=args)
    
    # 测试
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    
    mse_result = {"test": test_result[0]["test_mse"], "val": val_result[0]["test_mse"]}
    mae_result = {"test": test_result[0]["test_mae"], "val": val_result[0]["test_mae"]}

    return model, mse_result, mae_result


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='FusAD-Forecasting')

    # 数据参数
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='data/ETT-small',
                        help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')

    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]')

    # 预测长度参数
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # 训练参数
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--pretrain_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
    parser.add_argument('--seed', type=int, default=42)

    # 模型参数
    parser.add_argument('--emb_dim', type=int, default=64, help='dimension of model')
    parser.add_argument('--depth', type=int, default=3, help='num of layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout value')
    parser.add_argument('--patch_size', type=int, default=64, help='size of patches')
    parser.add_argument('--mask_ratio', type=float, default=0.4)

    # FusAD组件开关
    parser.add_argument('--load_from_pretrained', type=str2bool, default=True, help='False: without pretraining')
    parser.add_argument('--IFM', type=str2bool, default=True, help='使用交互融合模块')
    parser.add_argument('--ASM', type=str2bool, default=True, help='使用自适应频谱模块')
    parser.add_argument('--adaptive_filter', type=str2bool, default=True, help='使用自适应滤波')

    args = parser.parse_args()

    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')

    # 运行描述
    run_description = f"{args.data_path.split('.')[0]}_emb{args.emb_dim}_d{args.depth}_ps{args.patch_size}"
    run_description += f"_pl{args.pred_len}_bs{args.batch_size}_mr{args.mask_ratio}"
    run_description += f"_ASM_{args.ASM}_AF_{args.adaptive_filter}_IFM_{args.IFM}_preTr_{args.load_from_pretrained}"
    run_description += f"_{datetime.datetime.now().strftime('%H_%M')}"
    print(f"========== {run_description} ===========")

    CHECKPOINT_PATH = f"lightning_logs/{run_description}"
    pretrain_checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
        save_top_k=1,
        filename='pretrain-{epoch}',
        monitor='val_loss',
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
        save_top_k=1,
        monitor='val_mse',
        mode='min'
    )

    # 保存当前脚本备份
    save_copy_of_files(checkpoint_callback)

    # 确保GPU操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 加载数据集
    train_data, train_loader = data_provider(args, flag='train')
    vali_data, val_loader = data_provider(args, flag='val')
    test_data, test_loader = data_provider(args, flag='test')
    print("Dataset loaded ...")

    # 预训练
    if args.load_from_pretrained:
        pretrained_model, best_model_path = pretrain_model(
            args, train_loader, val_loader, CHECKPOINT_PATH, pretrain_checkpoint_callback
        )
    else:
        best_model_path = ''

    # 训练
    model, mse_result, mae_result = train_model(
        args, train_loader, val_loader, test_loader, 
        CHECKPOINT_PATH, checkpoint_callback, best_model_path
    )
    print("MSE results", mse_result)
    print("MAE results", mae_result)

    # 保存结果到Excel
    df = pd.DataFrame({
        'MSE': mse_result,
        'MAE': mae_result
    })
    df.to_excel(os.path.join(CHECKPOINT_PATH, f"results_{datetime.datetime.now().strftime('%H_%M')}.xlsx"))

    # 追加结果到文本文件
    os.makedirs("textOutput", exist_ok=True)
    f = open(f"textOutput/FusAD_{os.path.basename(args.data_path)}.txt", 'a')
    f.write(run_description + "  \n")
    f.write('MSE:{}, MAE:{}'.format(mse_result, mae_result))
    f.write('\n')
    f.write('\n')
    f.close()
