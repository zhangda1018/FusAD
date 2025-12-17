import torch
import torch.nn as nn
from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_
from einops import rearrange

from models.ASM import Adaptive_Spectral_Module
from models.IFM import IFM


class FusAD_layer(nn.Module):
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


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        
        self.d_model = configs.d_model
        self.depth = configs.e_layers
        self.dropout = configs.dropout
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        
        self.num_patches = (self.seq_len - self.patch_size) // self.stride + 1
        
        self.input_layer = nn.Linear(self.patch_size, self.d_model)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.d_model), requires_grad=True)
        self.pos_drop = nn.Dropout(p=self.dropout)
        trunc_normal_(self.pos_embed, std=.02)
        
        dpr = [x.item() for x in torch.linspace(0, self.dropout, self.depth)]
        
        # FusAD层堆叠
        self.jafa_blocks = nn.ModuleList([
            FusAD_layer(
                dim=self.d_model, 
                drop=self.dropout, 
                drop_path=dpr[i],
                use_IFM=True,
                use_ASM=True,
                adaptive_filter=True
            )
            for i in range(self.depth)]
        )
        
        # 输出层 - 重建原始输入
        self.output_layer = nn.Linear(self.d_model, self.patch_size)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        前向传播 - 重建输入序列 (向量化优化版本)
        Args:
            x_enc: [B, L, C] 输入序列
        Returns:
            outputs: [B, L, C] 重建序列
        """
        B, L, C = x_enc.shape
        
        # 将所有通道合并为批次维度: [B, L, C] -> [B*C, L]
        x = x_enc.permute(0, 2, 1).reshape(B * C, L)  # [B*C, L]
        
        # Patch嵌入: [B*C, L] -> [B*C, num_patches, patch_size]
        x_patched = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)  # [B*C, num_patches, patch_size]
        
        # 线性投影
        x_embed = self.input_layer(x_patched)  # [B*C, num_patches, d_model]
        
        # 添加位置嵌入
        x_embed = x_embed + self.pos_embed
        x_embed = self.pos_drop(x_embed)
        
        # FusAD层处理
        for jafa_blk in self.jafa_blocks:
            x_embed = jafa_blk(x_embed)
        
        # 输出重建
        x_recon = self.output_layer(x_embed)  # [B*C, num_patches, patch_size]
        
        # 使用fold操作进行快速重建 (替代for循环)
        # 将patches重组为完整序列
        x_recon_t = x_recon.permute(0, 2, 1)  # [B*C, patch_size, num_patches]
        
        # 使用fold操作
        outputs = torch.nn.functional.fold(
            x_recon_t,
            output_size=(1, L),
            kernel_size=(1, self.patch_size),
            stride=(1, self.stride)
        )  # [B*C, 1, 1, L]
        
        # 计算重叠计数
        ones = torch.ones_like(x_recon_t)
        count = torch.nn.functional.fold(
            ones,
            output_size=(1, L),
            kernel_size=(1, self.patch_size),
            stride=(1, self.stride)
        )  # [B*C, 1, 1, L]
        
        # 平均重叠
        outputs = outputs / count.clamp(min=1)
        outputs = outputs.squeeze(1).squeeze(1)  # [B*C, L]
        
        # 恢复原始形状: [B*C, L] -> [B, L, C]
        outputs = outputs.reshape(B, C, L).permute(0, 2, 1)  # [B, L, C]
        
        return outputs
