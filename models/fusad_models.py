import torch
import torch.nn as nn
import lightning as L
from einops import rearrange
from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_


class IFM(L.LightningModule):
    """Interactive Fusion Module"""
    def __init__(self, in_features, hidden_features, kernel_size=5, dropout=0.5, drop=0.):
        super().__init__()

        # Define the components that were in Interactor
        self.alpha = nn.Sequential(
            nn.Conv1d(in_features, hidden_features, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_features, in_features, kernel_size=3, padding=1),
            nn.Tanh()
        )
        self.beta = nn.Sequential(
            nn.Conv1d(in_features, hidden_features, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_features, in_features, kernel_size=3, padding=1),
            nn.Tanh()
        )
        self.miu = nn.Sequential(
            nn.Conv1d(in_features, hidden_features, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_features, in_features, kernel_size=3, padding=1),
            nn.Tanh()
        )

        self.sigma = nn.Sequential(
            nn.Conv1d(in_features, hidden_features, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_features, in_features, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # adaptive filter parameters
        self.affine_alpha = nn.Parameter(torch.ones(1, 1, in_features))
        self.affine_beta = nn.Parameter(torch.zeros(1, 1, in_features))
        self.drop_path = DropPath(drop) if drop > 0. else nn.Identity()

    def forward(self, x, f):
        B, L, M = x.shape
        x_freq = f

        x = x.transpose(1, 2).contiguous()  
        f = f.transpose(1, 2).contiguous()

        alpha = self.alpha(x)
        beta = self.beta(x)
        miu = self.miu(f)
        sigma = self.sigma(f)

        interactive_x = alpha * x + beta
        interactive_f = miu * f + sigma

        x_res = self.affine_alpha * interactive_x + self.affine_beta * interactive_f

        x_res = x_res.transpose(1, 2).contiguous()
        return self.drop_path(x_res)


class ASM(nn.Module):
    """Adaptive Spectral Module"""
    def __init__(self, in_dim, hidden_dim):
        super(ASM, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        
        # 1D convolution for spectral transformation
        self.spectral_conv = nn.Conv1d(in_dim, hidden_dim, kernel_size=1)
        self.activation = nn.GELU()
        self.output_conv = nn.Conv1d(hidden_dim, in_dim, kernel_size=1)
        
        # Learnable parameters for frequency domain operations
        self.frequency_weights = nn.Parameter(torch.ones(in_dim))
        
    def forward(self, x):
        # x shape: [B, L, D]
        B, L, D = x.shape
        
        # FFT to frequency domain
        x_freq = torch.fft.rfft(x, dim=1)
        
        # Apply learnable frequency weights
        x_freq = x_freq * self.frequency_weights.unsqueeze(0).unsqueeze(0)
        
        # Back to time domain
        x_time = torch.fft.irfft(x_freq, n=L, dim=1)
        
        # Apply convolution in time domain
        x_time = x_time.transpose(1, 2)  # [B, D, L]
        x_conv = self.spectral_conv(x_time)
        x_conv = self.activation(x_conv)
        x_conv = self.output_conv(x_conv)
        x_conv = x_conv.transpose(1, 2)  # [B, L, D]
        
        return x_conv


class FusAD_layer(nn.Module):
    """FusAD Layer with time-frequency fusion"""
    def __init__(self, emb_dim, depth, dropout_rate=0.1, adaptive_filter=True):
        super(FusAD_layer, self).__init__()
        
        self.emb_dim = emb_dim
        self.depth = depth
        self.adaptive_filter = adaptive_filter
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        
        # Interactive Fusion Module
        self.ifm = IFM(emb_dim, emb_dim * 2, dropout=dropout_rate)
        
        # Adaptive Spectral Module
        self.asm = ASM(emb_dim, emb_dim * 2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Position-wise feed forward
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(emb_dim * 4, emb_dim)
        )
        
    def forward(self, x):
        # Time domain processing
        x_time = self.norm1(x)
        
        # Frequency domain processing  
        x_freq = torch.fft.rfft(x_time, dim=1)
        x_freq_real = torch.fft.irfft(x_freq, n=x_time.size(1), dim=1)
        
        if self.adaptive_filter:
            # Apply ASM for frequency domain adaptation
            x_freq_processed = self.asm(x_freq_real)
            
            # Interactive fusion between time and frequency
            x_fused = self.ifm(x_time, x_freq_processed)
        else:
            x_fused = x_time
            
        # Residual connection
        x = x + self.dropout(x_fused)
        
        # Feed forward with residual connection
        x_ffn = self.norm2(x)
        x_ffn = self.ffn(x_ffn)
        x = x + self.dropout(x_ffn)
        
        return x


class FusAD_Base(nn.Module):
    """Base FusAD model for all tasks"""
    def __init__(self, seq_len, emb_dim, depth, num_classes=None, pred_len=None, 
                 dropout_rate=0.1, adaptive_filter=True):
        super(FusAD_Base, self).__init__()
        
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.depth = depth
        self.adaptive_filter = adaptive_filter
        
        # Input embedding
        self.input_projection = nn.Linear(seq_len, emb_dim)
        
        # FusAD layers
        self.fusad_layers = nn.ModuleList([
            FusAD_layer(emb_dim, depth, dropout_rate, adaptive_filter)
            for _ in range(depth)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(emb_dim)
        
    def forward(self, x):
        # Input projection
        x = self.input_projection(x.transpose(1, 2)).transpose(1, 2)
        
        # Apply FusAD layers
        for layer in self.fusad_layers:
            x = layer(x)
            
        # Final normalization
        x = self.norm(x)
        
        return x


class FusAD_Classification(FusAD_Base):
    """FusAD model for classification tasks"""
    def __init__(self, seq_len, emb_dim, depth, num_classes, dropout_rate=0.1, adaptive_filter=True):
        super(FusAD_Classification, self).__init__(
            seq_len, emb_dim, depth, num_classes, None, dropout_rate, adaptive_filter
        )
        
        self.num_classes = num_classes
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(emb_dim, num_classes)
        )
        
    def forward(self, x):
        # Get base features
        x = super().forward(x)  # [B, L, D]
        
        # Apply classifier
        x = x.transpose(1, 2)  # [B, D, L] 
        logits = self.classifier(x)  # [B, num_classes]
        
        return logits


class FusAD_Forecasting(FusAD_Base):
    """FusAD model for forecasting tasks"""
    def __init__(self, seq_len, emb_dim, depth, pred_len, enc_in, dropout_rate=0.1, adaptive_filter=True):
        super(FusAD_Forecasting, self).__init__(
            seq_len, emb_dim, depth, None, pred_len, dropout_rate, adaptive_filter
        )
        
        self.pred_len = pred_len
        self.enc_in = enc_in
        
        # Override input projection for multivariate time series
        self.input_projection = nn.Linear(enc_in, emb_dim)
        
        # Forecasting head
        self.forecasting_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(emb_dim * 2, pred_len)
        )
        
    def forward(self, x):
        # x shape: [B, seq_len, enc_in]
        B, L, D = x.shape
        
        # Project input features to embedding dimension
        x = self.input_projection(x)  # [B, L, emb_dim]
        
        # Apply FusAD layers
        for layer in self.fusad_layers:
            x = layer(x)
            
        # Final normalization
        x = self.norm(x)
        
        # Forecasting head - predict next pred_len steps for each feature
        predictions = self.forecasting_head(x)  # [B, L, pred_len]
        
        # Take mean across sequence length and reshape properly
        predictions = predictions.mean(dim=1)  # [B, pred_len]
        predictions = predictions.unsqueeze(-1).repeat(1, 1, self.enc_in)  # [B, pred_len, enc_in]
        
        return predictions


class FusAD_AnomalyDetection(FusAD_Base):
    """FusAD model for anomaly detection tasks"""
    def __init__(self, seq_len, emb_dim, depth, enc_in, dropout_rate=0.1, adaptive_filter=True):
        super(FusAD_AnomalyDetection, self).__init__(
            seq_len, emb_dim, depth, None, None, dropout_rate, adaptive_filter
        )
        
        self.enc_in = enc_in
        
        # Override input projection for multivariate time series
        self.input_projection = nn.Linear(enc_in, emb_dim)
        
        # Reconstruction head for anomaly detection
        self.reconstruction_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(emb_dim * 2, enc_in)
        )
        
        # Normalization parameters (similar to GPT4TS approach)
        self.affine_weight = nn.Parameter(torch.ones(1, 1, enc_in))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, enc_in))
        
    def forward(self, x):
        # x shape: [B, seq_len, enc_in]
        B, L, D = x.shape
        
        # Segmented normalization (inspired by One-Fits-All GPT4TS)
        means = x.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_norm = (x - means) / stdev
        
        # Project input features to embedding dimension
        x_emb = self.input_projection(x_norm)  # [B, L, emb_dim]
        
        # Apply FusAD layers
        for layer in self.fusad_layers:
            x_emb = layer(x_emb)
            
        # Final normalization
        x_emb = self.norm(x_emb)
        
        # Reconstruction
        x_rec = self.reconstruction_head(x_emb)  # [B, L, enc_in]
        
        # Denormalization
        x_rec = x_rec * (stdev * self.affine_weight + 1e-10) + (means * self.affine_bias)
        
        return x_rec
