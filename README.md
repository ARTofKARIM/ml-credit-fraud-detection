# Credit Card Fraud Detection

An anomaly detection system for identifying fraudulent credit card transactions using multiple machine learning approaches including Isolation Forest, Local Outlier Factor (LOF), and Autoencoder neural networks.

## Overview

Credit card fraud represents a significant financial threat. This project implements and compares three distinct anomaly detection strategies to identify fraudulent transactions in highly imbalanced datasets.

## Architecture

```
ml-credit-fraud-detection/
├── src/
│   ├── data_loader.py          # Data ingestion and train/test splitting
│   ├── preprocessing.py        # Scaling, PCA, SMOTE resampling
│   ├── feature_engineering.py  # Derived feature creation
│   ├── models.py               # Isolation Forest & LOF detectors
│   ├── autoencoder.py          # Keras autoencoder for anomaly detection
│   ├── evaluation.py           # Metrics computation and model comparison
│   └── visualization.py        # ROC curves, confusion matrices, t-SNE
├── config/
│   └── config.yaml             # Model and pipeline configuration
├── tests/
│   └── test_models.py          # Unit tests for detection models
└── main.py                     # Pipeline entry point
```

## Models

| Model | Approach | Strengths |
|-------|----------|-----------|
| Isolation Forest | Tree-based isolation | Fast, handles high dimensions |
| LOF | Density-based neighbors | Good for local anomalies |
| Autoencoder | Reconstruction error | Learns complex patterns |

## Installation

```bash
git clone https://github.com/mouachiqab/ml-credit-fraud-detection.git
cd ml-credit-fraud-detection
pip install -r requirements.txt
```

## Usage

```bash
# Run full pipeline with all models
python main.py --data data/creditcard.csv --model all

# Run specific model
python main.py --data data/creditcard.csv --model isolation_forest

# Disable SMOTE oversampling
python main.py --data data/creditcard.csv --model all --no-smote
```

## Results

The models are evaluated on precision, recall, F1-score, ROC-AUC, and PR-AUC metrics. The autoencoder typically achieves the best balance between precision and recall on highly imbalanced fraud datasets.

## Technologies

- Python 3.9+
- scikit-learn (Isolation Forest, LOF)
- TensorFlow/Keras (Autoencoder)
- imbalanced-learn (SMOTE)
- pandas, NumPy, Matplotlib, Seaborn








