"""Data loading and initial exploration module for credit card fraud detection."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
import os


class CreditCardDataLoader:
    """Handles loading, exploration, and splitting of credit card transaction data."""

    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load(self, filepath=None):
        """Load the credit card dataset from CSV file."""
        path = filepath or self.config["data"]["path"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found at {path}")
        self.data = pd.read_csv(path)
        print(f"Dataset loaded: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        return self.data

    def describe(self):
        """Generate summary statistics of the dataset."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")
        summary = {
            "shape": self.data.shape,
            "dtypes": self.data.dtypes.value_counts().to_dict(),
            "missing_values": self.data.isnull().sum().sum(),
            "fraud_ratio": self.data[self.config["data"]["target_column"]].mean(),
            "class_distribution": self.data[self.config["data"]["target_column"]].value_counts().to_dict(),
        }
        print(f"Dataset shape: {summary['shape']}")
        print(f"Missing values: {summary['missing_values']}")
        print(f"Fraud ratio: {summary['fraud_ratio']:.4f}")
        return summary

    def get_feature_stats(self):
        """Compute basic statistics for each feature."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")
        return self.data.describe().T

    def split(self, test_size=None, random_state=None):
        """Split data into training and testing sets with stratification."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")
        target = self.config["data"]["target_column"]
        test_sz = test_size or self.config["data"]["test_size"]
        rand_st = random_state or self.config["data"]["random_state"]

        X = self.data.drop(columns=[target])
        y = self.data[target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_sz, random_state=rand_st, stratify=y
        )
        print(f"Train set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        return self.X_train, self.X_test, self.y_train, self.y_test
