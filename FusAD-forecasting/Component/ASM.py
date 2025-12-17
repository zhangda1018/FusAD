import torch
from timm.models.layers import trunc_normal_
import torch.nn as nn


class Adaptive_Spectral_Module(nn.Module):
    """
    自适应频谱模块 (Adaptive Spectral Module)
    结合FFT频域分析和类小波CNN分支，用于时间序列预测任务
    """
    def __init__(self, dim, drop=0.2, adaptive_filter=True):
        super().__init__()
        self.adaptive_filter = adaptive_filter
        
        # FFT权重参数
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)
        
        # 自适应阈值参数
        self.high_threshold_param = nn.Parameter(torch.rand(1))
        self.low_threshold_param = nn.Parameter(torch.rand(1))
        
        # 类小波CNN分支
        self.wavelet_conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.wavelet_conv2 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def create_adaptive_high_freq_mask(self, x_fft):
        """创建自适应高低频掩码"""
        B, _, _ = x_fft.shape
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)
        flat_energy = energy.view(B, -1)
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]
        median_energy = median_energy.view(B, 1)
        epsilon = 1e-6
        normalized_energy = energy / (median_energy + epsilon)
        
        high_freq_mask = ((normalized_energy > self.high_threshold_param).float() - self.high_threshold_param).detach() + self.high_threshold_param
        low_freq_mask = ((normalized_energy < self.low_threshold_param).float() - self.low_threshold_param).detach() + self.low_threshold_param
        adaptive_mask = high_freq_mask + low_freq_mask
        adaptive_mask = adaptive_mask.unsqueeze(-1)
        return adaptive_mask

    def apply_window(self, x):
        """应用Hann窗函数减少频谱泄露"""
        time_dim = x.size(1)
        window = torch.hann_window(time_dim, periodic=True).to(x.device)
        return x * window.unsqueeze(0).unsqueeze(-1)

    def forward(self, x_in):
        B, N, C = x_in.shape
        dtype = x_in.dtype
        x = x_in.to(torch.float32)
        
        # 应用窗函数
        x = self.apply_window(x)
        
        # FFT分支
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        if self.adaptive_filter:
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)
            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high
            x_weighted += x_weighted2

        x_ifft = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')

        # 类小波CNN分支
        x_wavelet = x.permute(0, 2, 1)
        x_wavelet = self.wavelet_conv1(x_wavelet)
        x_wavelet = self.act(x_wavelet)
        x_wavelet = self.drop(x_wavelet)
        x_wavelet = self.wavelet_conv2(x_wavelet)
        x_wavelet = self.act(x_wavelet)
        x_wavelet = self.drop(x_wavelet)
        x_wavelet = x_wavelet.permute(0, 2, 1)

        # 融合FFT和小波分支结果
        x_combined = x_ifft + x_wavelet

        x_combined = x_combined.to(dtype)
        x_combined = x_combined.view(B, N, C)
        return x_combined
