# FusAD: Time-Frequency Fusion with Adaptive Denoising for General Time Series Analysis

<p align="center">
<a href="http://arxiv.org/abs/2512.14078">
    <img src="https://img.shields.io/badge/arXiv-2512.14078-green" /></a>
<a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/Framework-PyTorch-orange.svg" /></a>
</p>

## 📖 Introduction

**FusAD** (Fusion Adaptive Network) is a unified deep learning framework for time series analysis, supporting three major tasks:

- 🔍 **Anomaly Detection** 
- 📊 **Classification** 
- 📈 **Forecasting** 

D. Zhang, B. Li, Z. Zhao, F. Nie, J. Gao, and X. Li, "[FusAD: Time-Frequency Fusion with Adaptive Denoising for General Time Series Analysis](https://arxiv.org/abs/2512.14078)," *ICDE*, 2026.


## 📊 Supported Datasets

### Forecasting and Anomaly Detection
Forecasting and Anomaly Detection datasets are downloaded from [TimesNet](https://github.com/thuml/Time-Series-Library).

### Classification
UCR and UEA classification datasets are available at [UCR Time Series Classification Archive](https://www.timeseriesclassification.com).


## 📁 Project Structure

```
FusAD/
├── FusAD-Anomaly-Detection/    # Anomaly detection task
│   ├── models/
│   │   ├── FusAD.py            # Main model
│   │   ├── ASM.py              # Adaptive Spectral Module
│   │   └── IFM.py              # Interactor-Fusion Module
│   ├── exp/                    # Experiment classes
│   ├── data_provider/          # Data loading utilities
│   ├── scripts/                # Training scripts (MSL, SMAP, SMD, PSM, SWAT)
│   ├── utils/                  # Utility functions
│   └── run.py                  # Main entry point
│
├── FusAD-classification/       # Classification task
│   ├── Component/
│   │   ├── ASM.py
│   │   ├── IFM.py
│   │   └── Patch.py
│   ├── Dataload/               # Data loading
│   ├── utils/                  # Utilities
│   └── main.py                 # Main entry point
│
├── FusAD-forecasting/          # Forecasting task
│   ├── Component/
│   │   ├── ASM.py
│   │   └── IFM.py
│   ├── scripts/                # Training scripts (ETT, Weather, Traffic, etc.)
│   ├── data_factory.py         # Data factory
│   ├── data_loader.py          # Data loader
│   └── FusAD_Forecasting.py    # Main entry point
│
└── README.md
```

## 🚀 Quick Start

### Requirements

```bash
pip install torch lightning timm einops pandas numpy scikit-learn
```

### Anomaly Detection

```bash
cd FusAD-Anomaly-Detection

# Train on MSL dataset
bash scripts/MSL.sh

# Or run directly
python run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --model FusAD \
  --data MSL \
  --root_path ./data/MSL \
  --seq_len 100 \
  --d_model 256 \
  --e_layers 3 \
  --batch_size 128
```

### Classification

```bash
cd FusAD-classification

python main.py \
  --model_id YourDataset \
  --data_path /path/to/dataset \
  --emb_dim 64 \
  --depth 3 \
  --batch_size 1024 \
  --num_epochs 300
```

### Forecasting

```bash
cd FusAD-forecasting

# Train on ETTh1 dataset
bash scripts/ETTh1.sh

# Or run directly
python FusAD_Forecasting.py \
  --root_path ./data/ETT-small \
  --data ETTh1 \
  --data_path ETTh1.csv \
  --seq_len 512 \
  --pred_len 96 \
  --emb_dim 64 \
  --depth 3 \
  --batch_size 512
```



## 📝 Citation

If you find FusAD useful in your research, please consider citing:

```bibtex
@article{zhang2026fusad,
  title={Time-Frequency Fusion with Adaptive Denoising for General Time Series Analysis},
  author={Zhang, Da and Li, Bingyu and Zhao, Zhiyuan and Nie, Feiping and Gao, Junyu and Li, Xuelong},
  journal={arXiv preprint arXiv:2512.14078},
  year={2025}   
}
```

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgements

Our codebase is based on the following Github repositories. Thanks to the following public repositories:
- [TSLANet](https://github.com/emadeldeen24/TSLANet)

- [PatchTST](https://github.com/yuqinie98/PatchTST)

- [TimesNet](https://github.com/thuml/Time-Series-Library)  

Note: This is a research level repository and might contain issues/bugs. Please contact the authors for any query.
