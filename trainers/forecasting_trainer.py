import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError

from models.fusad_models import FusAD_Forecasting


class model_training_forecasting(L.LightningModule):
    """PyTorch Lightning module for FusAD forecasting training"""
    
    def __init__(self, seq_len, pred_len, enc_in, emb_dim, depth, 
                 learning_rate=1e-3, dropout_rate=0.1, adaptive_filter=True):
        super().__init__()
        
        self.save_hyperparameters()
        
        # Model parameters
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.emb_dim = emb_dim
        self.depth = depth
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.adaptive_filter = adaptive_filter
        
        # Initialize model
        self.model = FusAD_Forecasting(
            seq_len=seq_len,
            emb_dim=emb_dim,
            depth=depth,
            pred_len=pred_len,
            enc_in=enc_in,
            dropout_rate=dropout_rate,
            adaptive_filter=adaptive_filter
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Metrics
        self.train_mse = MeanSquaredError()
        self.train_mae = MeanAbsoluteError()
        self.val_mse = MeanSquaredError()
        self.val_mae = MeanAbsoluteError()
        
    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        
        # Forward pass
        outputs = self(batch_x)
        
        # Compute loss
        loss = self.criterion(outputs, batch_y)
        
        # Compute metrics
        mse = self.train_mse(outputs, batch_y)
        mae = self.train_mae(outputs, batch_y)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_mse', mse, prog_bar=True)
        self.log('train_mae', mae, prog_bar=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        
        # Forward pass
        outputs = self(batch_x)
        
        # Compute loss
        loss = self.criterion(outputs, batch_y)
        
        # Compute metrics
        mse = self.val_mse(outputs, batch_y)
        mae = self.val_mae(outputs, batch_y)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mse', mse, prog_bar=True)
        self.log('val_mae', mae, prog_bar=True)
        
        return loss
        
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)


def train_model_forecasting(args, train_loader, val_loader, test_loader):
    """Train FusAD forecasting model"""
    
    # Initialize model
    model = model_training_forecasting(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        enc_in=args.enc_in,
        emb_dim=args.emb_dim,
        depth=args.depth,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout_rate,
        adaptive_filter=args.adaptive_filter
    )
    
    # Setup callbacks
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
    
    CHECKPOINT_PATH = f"lightning_logs/{args.run_description}"
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
        save_top_k=1,
        filename='forecasting-{epoch}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min'
    )
    
    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=args.train_epochs,
        callbacks=[checkpoint_callback, LearningRateMonitor(), TQDMProgressBar()],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1
    )
    
    # Train model
    print("Starting forecasting training...")
    trainer.fit(model, train_loader, val_loader)
    
    # Load best model
    best_model_path = checkpoint_callback.best_model_path
    best_model = model_training_forecasting.load_from_checkpoint(best_model_path)
    
    return best_model, trainer


def test_forecasting(model, trainer, test_loader):
    """Test FusAD forecasting model"""
    
    # Test the model
    test_results = trainer.test(model, test_loader)
    
    # Collect predictions and ground truth
    predictions = []
    ground_truth = []
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            if torch.cuda.is_available():
                batch_x = batch_x.cuda()
                
            outputs = model(batch_x)
            
            predictions.append(outputs.cpu().numpy())
            ground_truth.append(batch_y.cpu().numpy())
    
    import numpy as np
    predictions = np.concatenate(predictions, axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)
    
    # Calculate final metrics
    mse = np.mean((predictions - ground_truth) ** 2)
    mae = np.mean(np.abs(predictions - ground_truth))
    
    print(f"Test Results:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    
    return test_results, predictions, ground_truth
