import torch
import torch.nn as nn
import pytorch_lightning as L

class IFM(L.LightningModule):
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
        self.niu = nn.Sequential(
            nn.Conv1d(in_features, hidden_features, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_features, in_features, kernel_size=3, padding=1),
            nn.Tanh()
        )

        # Define the additional components of IFM
        self.rou = nn.Conv1d(in_features, hidden_features, 1)
        self.omega = nn.Conv1d(in_features, hidden_features, 3, 1, 1)
        self.conv_final = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x1 = x.transpose(1, 2)
        x2 = x1.clone()

        alpha_x2 = torch.exp(self.alpha(x2))
        beta_x1 = torch.exp(self.beta(x1))
        if alpha_x2.shape != x1.shape:
            raise ValueError(f"Shape mismatch: alpha_x2 {alpha_x2.shape}, x1 {x1.shape}")
        if beta_x1.shape != x2.shape:
            raise ValueError(f"Shape mismatch: beta_x1 {beta_x1.shape}, x2 {x2.shape}")

        x3 = x1 * alpha_x2
        x4 = x2 * beta_x1
        x6 = self.miu(x3) + x4
        x5 = self.niu(x4) + x3

        x1_1 = self.rou(x5)
        x1_1 = self.act(x1_1)
        x1_2 = self.drop(x1_1)

        x2_1 = self.omega(x6)
        x2_1 = self.act(x2_1)
        x2_2 = self.drop(x2_1)

        out1 = x1_1 * x2_2
        out2 = x2_1 * x1_2

        x = self.conv_final(out1 + out2)
        x = x.transpose(1, 2)  # Transpose back to original format.
        return x
