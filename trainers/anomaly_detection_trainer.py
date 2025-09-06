import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from models.fusad_models import FusAD_AnomalyDetection


class model_training_anomaly_detection(L.LightningModule):
    """PyTorch Lightning module for FusAD anomaly detection training"""
    
    def __init__(self, seq_len, enc_in, emb_dim, depth, 
                 learning_rate=1e-3, dropout_rate=0.1, adaptive_filter=True):
        super().__init__()
        
        self.save_hyperparameters()
        
        # Model parameters
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.emb_dim = emb_dim
        self.depth = depth
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.adaptive_filter = adaptive_filter
        
        # Initialize model
        self.model = FusAD_AnomalyDetection(
            seq_len=seq_len,
            emb_dim=emb_dim,
            depth=depth,
            enc_in=enc_in,
            dropout_rate=dropout_rate,
            adaptive_filter=adaptive_filter
        )
        
        # Loss function - MSE for reconstruction
        self.criterion = nn.MSELoss()
        
    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        batch_x, _ = batch  # For anomaly detection, we only need input data
        
        # Forward pass - reconstruct input
        reconstructed = self(batch_x)
        
        # Compute reconstruction loss
        loss = self.criterion(reconstructed, batch_x)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        batch_x, _ = batch
        
        # Forward pass
        reconstructed = self(batch_x)
        
        # Compute reconstruction loss
        loss = self.criterion(reconstructed, batch_x)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        
        return loss
        
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)


def train_model_anomaly_detection(args, train_loader, val_loader, test_loader):
    """Train FusAD anomaly detection model"""
    
    # Initialize model
    model = model_training_anomaly_detection(
        seq_len=args.seq_len,
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
        filename='anomaly-{epoch}-{val_loss:.4f}',
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
    print("Starting anomaly detection training...")
    trainer.fit(model, train_loader, val_loader)
    
    # Load best model
    best_model_path = checkpoint_callback.best_model_path
    best_model = model_training_anomaly_detection.load_from_checkpoint(best_model_path)
    
    return best_model, trainer


def test_anomaly_detection(model, trainer, test_loader, anomaly_ratio=1.0):
    """Test FusAD anomaly detection model and compute anomaly scores"""
    
    # Collect predictions and ground truth
    reconstruction_errors = []
    true_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch_x, batch_y = batch
            if torch.cuda.is_available():
                batch_x = batch_x.cuda()
                
            # Get reconstructions
            reconstructed = model(batch_x)
            
            # Calculate reconstruction error for each sample
            # MSE per sample
            errors = torch.mean((batch_x - reconstructed) ** 2, dim=(1, 2))
            
            reconstruction_errors.extend(errors.cpu().numpy())
            true_labels.extend(batch_y.cpu().numpy())
    
    reconstruction_errors = np.array(reconstruction_errors)
    true_labels = np.array(true_labels)
    
    # Set threshold using percentile based on anomaly ratio
    threshold = np.percentile(reconstruction_errors, 100 - anomaly_ratio)
    
    # Predict anomalies based on threshold
    predicted_labels = (reconstruction_errors > threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average='binary'
    )
    
    print(f"Anomaly Detection Test Results:")
    print(f"Threshold: {threshold:.6f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'threshold': threshold,
        'reconstruction_errors': reconstruction_errors,
        'true_labels': true_labels,
        'predicted_labels': predicted_labels
    }
