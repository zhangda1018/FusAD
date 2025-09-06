import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
from timm.loss import LabelSmoothingCrossEntropy
from torchmetrics.classification import MulticlassF1Score

from models.fusad_models import FusAD_Classification
from utils import random_masking_3D


class model_training_classification(L.LightningModule):
    """PyTorch Lightning module for FusAD classification training"""
    
    def __init__(self, num_sensors, seq_len, emb_dim, depth, num_classes, 
                 pretrain_lr=1e-3, train_lr=1e-3, dropout_rate=0.1, 
                 adaptive_filter=True, load_from_pretrained=True, 
                 masking_ratio=0.4):
        super().__init__()
        
        self.save_hyperparameters()
        
        # Model parameters
        self.num_sensors = num_sensors
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.depth = depth
        self.num_classes = num_classes
        self.pretrain_lr = pretrain_lr
        self.train_lr = train_lr
        self.dropout_rate = dropout_rate
        self.adaptive_filter = adaptive_filter
        self.load_from_pretrained = load_from_pretrained
        self.masking_ratio = masking_ratio
        
        # Initialize model
        self.model = FusAD_Classification(
            seq_len=seq_len,
            emb_dim=emb_dim, 
            depth=depth,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            adaptive_filter=adaptive_filter
        )
        
        # Loss functions
        self.pretrain_criterion = nn.MSELoss()
        self.train_criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        
        # Metrics
        self.train_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')
        
        # Training phase tracking
        self.pretraining = True
        
    def set_pretraining(self, pretraining):
        """Set training phase"""
        self.pretraining = pretraining
        
    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        if self.pretraining:
            # Pretraining with masked reconstruction
            x_masked, mask = random_masking_3D(x, self.masking_ratio)
            x_reconstructed = self(x_masked)
            
            # Only compute loss on masked regions
            loss = self.pretrain_criterion(x_reconstructed * mask, x * mask)
            self.log('pretrain_loss', loss, prog_bar=True)
            return loss
        else:
            # Classification training
            logits = self(x)
            loss = self.train_criterion(logits, y)
            
            # Calculate F1 score
            preds = torch.argmax(logits, dim=1)
            f1 = self.train_f1(preds, y)
            
            self.log('train_loss', loss, prog_bar=True)
            self.log('train_f1', f1, prog_bar=True)
            return loss
            
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        if self.pretraining:
            # Validation for pretraining
            x_masked, mask = random_masking_3D(x, self.masking_ratio)
            x_reconstructed = self(x_masked)
            loss = self.pretrain_criterion(x_reconstructed * mask, x * mask)
            self.log('val_loss', loss, prog_bar=True)
            return loss
        else:
            # Validation for classification
            logits = self(x)
            loss = self.train_criterion(logits, y)
            
            # Calculate F1 score
            preds = torch.argmax(logits, dim=1)
            f1 = self.val_f1(preds, y)
            
            self.log('val_loss', loss, prog_bar=True)
            self.log('val_f1', f1, prog_bar=True)
            return loss
            
    def configure_optimizers(self):
        if self.pretraining:
            return optim.Adam(self.parameters(), lr=self.pretrain_lr)
        else:
            return optim.Adam(self.parameters(), lr=self.train_lr)


def train_model_classification(args, pretrain_loader, train_loader, val_loader, test_loader, 
                             NUM_SENSORS, SEQ_LEN, NUM_CLASSES):
    """Train FusAD classification model"""
    
    # Initialize model
    model = model_training_classification(
        num_sensors=NUM_SENSORS,
        seq_len=SEQ_LEN,
        emb_dim=args.emb_dim,
        depth=args.depth,
        num_classes=NUM_CLASSES,
        pretrain_lr=args.pretrain_lr,
        train_lr=args.train_lr,
        dropout_rate=args.dropout_rate,
        adaptive_filter=args.adaptive_filter,
        load_from_pretrained=args.load_from_pretrained,
        masking_ratio=args.masking_ratio
    )
    
    # Setup callbacks
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
    
    CHECKPOINT_PATH = f"lightning_logs/{args.run_description}"
    
    pretrain_checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
        save_top_k=1,
        filename='pretrain-{epoch}',
        monitor='val_loss',
        mode='min'
    )
    
    train_checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
        save_top_k=1,
        filename='train-{epoch}-{val_f1:.3f}',
        monitor='val_f1',
        mode='max'
    )
    
    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=args.pretrain_epochs,
        callbacks=[pretrain_checkpoint_callback, LearningRateMonitor(), TQDMProgressBar()],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1
    )
    
    # Pretraining phase
    if args.load_from_pretrained:
        print("Starting pretraining...")
        model.set_pretraining(True)
        trainer.fit(model, pretrain_loader, val_loader)
        
        # Load best pretrained model
        best_pretrain_path = pretrain_checkpoint_callback.best_model_path
        model = model_training_classification.load_from_checkpoint(
            best_pretrain_path,
            num_sensors=NUM_SENSORS,
            seq_len=SEQ_LEN,
            emb_dim=args.emb_dim,
            depth=args.depth,
            num_classes=NUM_CLASSES,
            pretrain_lr=args.pretrain_lr,
            train_lr=args.train_lr,
            dropout_rate=args.dropout_rate,
            adaptive_filter=args.adaptive_filter,
            load_from_pretrained=args.load_from_pretrained,
            masking_ratio=args.masking_ratio
        )
    
    # Fine-tuning phase
    print("Starting fine-tuning...")
    model.set_pretraining(False)
    
    # Update trainer for fine-tuning
    trainer = L.Trainer(
        max_epochs=args.num_epochs,
        callbacks=[train_checkpoint_callback, LearningRateMonitor(), TQDMProgressBar()],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1
    )
    
    trainer.fit(model, train_loader, val_loader)
    
    # Load best model and test
    best_train_path = train_checkpoint_callback.best_model_path
    best_model = model_training_classification.load_from_checkpoint(best_train_path)
    
    return best_model, trainer


def test_classification(model, trainer, test_loader):
    """Test FusAD classification model"""
    
    # Test the model
    test_results = trainer.test(model, test_loader)
    
    # Get predictions for detailed analysis
    predictions = []
    true_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
                
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(y.cpu().numpy())
    
    # Generate classification report  
    from sklearn.metrics import classification_report
    report = classification_report(true_labels, predictions)
    print("Classification Report:")
    print(report)
    
    return test_results, predictions, true_labels
