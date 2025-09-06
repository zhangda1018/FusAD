import argparse
import datetime
import os
import torch

# Import modular components
from trainers.classification_trainer import train_model_classification, test_classification
from trainers.forecasting_trainer import train_model_forecasting, test_forecasting
from trainers.anomaly_detection_trainer import train_model_anomaly_detection, test_anomaly_detection

from dataloader import get_datasets
from data_factory import data_provider
from utils import save_copy_of_files, str2bool


def main():
    """Main execution function for FusAD unified framework"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='FusAD: Unified Framework for Classification, Forecasting, and Anomaly Detection')
    
    # Task selection
    parser.add_argument('--task', type=str, required=True, 
                        choices=['classification', 'forecasting', 'anomaly_detection'],
                        help='Task type: classification, forecasting, or anomaly_detection')
    
    # Common model parameters
    parser.add_argument('--seq_len', type=int, default=256, help='input sequence length')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--pretrain_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--train_lr', type=float, default=1e-3)
    parser.add_argument('--pretrain_lr', type=float, default=1e-3)
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--masking_ratio', type=float, default=0.4)
    parser.add_argument('--dropout_rate', type=float, default=0.15)
    parser.add_argument('--patch_size', type=int, default=8)
    parser.add_argument('--load_from_pretrained', type=str2bool, default=True, help='False: without pretraining')
    parser.add_argument('--IFM', type=str2bool, default=True)
    parser.add_argument('--ASM', type=str2bool, default=True)
    parser.add_argument('--adaptive_filter', type=str2bool, default=True)
    
    # Classification specific parameters
    parser.add_argument('--data_path', type=str, default=r'data/hhar')
    
    # Forecasting specific parameters
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='data/ETT-small',
                        help='root path of the data file')
    parser.add_argument('--data_file', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')  
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--mask_ratio', type=float, default=0.5, help='mask ratio')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    
    # Anomaly detection specific parameters  
    parser.add_argument('--anomaly_ratio', type=float, default=1.0, help='prior anomaly ratio (%)')
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size (number of features)')
    
    # Additional parameters required by data_factory
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
    parser.add_argument('--freq', type=str, default='h', help='frequency for time features encoding')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--percent', type=int, default=100, help='percent of data to use')
    parser.add_argument('--max_len', type=int, default=-1, help='max length of sequence')
    parser.add_argument('--train_all', type=str2bool, default=False, help='train on all data')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='seasonal patterns')
    
    # Root path for data (used by forecasting and anomaly detection)
    parser.add_argument('--root_path', type=str, default='data/', help='root path of data file')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda:{}'.format(0))
    
    # Create run description
    if args.task == 'classification':
        run_description = f"{os.path.basename(args.data_path)}_dim{args.emb_dim}_depth{args.depth}___"
        run_description += f"ASM_{args.ASM}__AF_{args.adaptive_filter}__IFM_{args.IFM}__preTr_{args.load_from_pretrained}_"
        run_description += f"{datetime.datetime.now().strftime('%H_%M_%S')}"
        DATASET_PATH = args.data_path
        MAX_EPOCHS = args.num_epochs
    elif args.task == 'forecasting':
        run_description = f"{args.data_file.split('.')[0]}_emb{args.emb_dim}_d{args.depth}_ps{args.patch_size}"
        run_description += f"_pl{args.pred_len}_bs{args.batch_size}_mr{args.mask_ratio}"
        run_description += f"_ASM_{args.ASM}_AF_{args.adaptive_filter}_IFM_{args.IFM}_preTr_{args.load_from_pretrained}"
        run_description += f"_{datetime.datetime.now().strftime('%H_%M')}"
    else:  # anomaly_detection
        run_description = f"anomaly_{args.data_path}_emb{args.emb_dim}_d{args.depth}_sl{args.seq_len}"
        run_description += f"_bs{args.batch_size}_ar{args.anomaly_ratio}"
        run_description += f"_ASM_{args.ASM}_AF_{args.adaptive_filter}_IFM_{args.IFM}"
        run_description += f"_{datetime.datetime.now().strftime('%H_%M')}"
    
    args.run_description = run_description
    print(f"========== {run_description} ===========")

    # Task-specific execution
    if args.task == 'classification':
        execute_classification(args)
    elif args.task == 'forecasting':
        execute_forecasting(args)
    elif args.task == 'anomaly_detection':
        execute_anomaly_detection(args)


def execute_classification(args):
    """Execute classification task"""
    print("Starting FusAD Classification Task...")
    
    # Load datasets and data loaders
    DATASET_PATH = args.data_path
    train_loader, val_loader, test_loader = get_datasets(DATASET_PATH, args)
    
    # For classification, pretrain_loader is the same as train_loader
    pretrain_loader = train_loader
    
    # Get dataset info
    sample_data, sample_label = next(iter(train_loader))
    NUM_SENSORS = sample_data.shape[-1]
    SEQ_LEN = sample_data.shape[1] 
    NUM_CLASSES = len(torch.unique(sample_label))
    
    print(f"Dataset: {args.data_path}")
    print(f"Number of sensors: {NUM_SENSORS}")
    print(f"Sequence length: {SEQ_LEN}")
    print(f"Number of classes: {NUM_CLASSES}")
    
    # Train model
    model, trainer = train_model_classification(
        args, pretrain_loader, train_loader, val_loader, test_loader,
        NUM_SENSORS, SEQ_LEN, NUM_CLASSES
    )
    
    # Test model
    test_results, predictions, true_labels = test_classification(model, trainer, test_loader)
    
    print("Classification task completed!")


def execute_forecasting(args):
    """Execute forecasting task"""
    print("Starting FusAD Forecasting Task...")
    
    # Prepare data
    train_data, train_loader = data_provider(args, 'train')
    val_data, val_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')
    
    print(f"Dataset: {args.data}")
    print(f"Input length: {args.seq_len}")
    print(f"Prediction length: {args.pred_len}")
    
    # Train model
    model, trainer = train_model_forecasting(args, train_loader, val_loader, test_loader)
    
    # Test model  
    test_results, predictions, ground_truth = test_forecasting(model, trainer, test_loader)
    
    print("Forecasting task completed!")


def execute_anomaly_detection(args):
    """Execute anomaly detection task"""
    print("Starting FusAD Anomaly Detection Task...")
    
    # Prepare data
    train_data, train_loader = data_provider(args, 'train')
    val_data, val_loader = data_provider(args, 'val') 
    test_data, test_loader = data_provider(args, 'test')
    
    print(f"Dataset: {args.data_path}")
    print(f"Input length: {args.seq_len}")
    print(f"Number of features: {args.enc_in}")
    print(f"Anomaly ratio: {args.anomaly_ratio}%")
    
    # Train model
    model, trainer = train_model_anomaly_detection(args, train_loader, val_loader, test_loader)
    
    # Test model
    results = test_anomaly_detection(model, trainer, test_loader, args.anomaly_ratio)
    
    print("Anomaly detection task completed!")


if __name__ == '__main__':
    main()
