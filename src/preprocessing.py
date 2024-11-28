"""Feature preprocessing pipeline for credit card fraud detection."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE


class DataPreprocessor:
    """Preprocessing pipeline including scaling, feature selection, and resampling."""

    def __init__(self, scaling_method="standard", n_components=None):
        self.scaling_method = scaling_method
        self.n_components = n_components
        self.scaler = None
        self.pca = None
        self.selected_features = None

    def fit_scaler(self, X):
        """Fit the scaler on training data."""
        if self.scaling_method == "standard":
            self.scaler = StandardScaler()
        elif self.scaling_method == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")
        self.scaler.fit(X)
        return self

    def transform_scale(self, X):
        """Apply scaling transformation."""
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler() first.")
        scaled = self.scaler.transform(X)
        return pd.DataFrame(scaled, columns=X.columns, index=X.index)

    def select_features_by_correlation(self, X, y, threshold=0.1):
        """Select features based on correlation with target variable."""
        correlations = pd.DataFrame(X).corrwith(pd.Series(y)).abs()
        self.selected_features = correlations[correlations > threshold].index.tolist()
        print(f"Selected {len(self.selected_features)} features with correlation > {threshold}")
        return X[self.selected_features]

    def remove_correlated_features(self, X, threshold=0.95):
        """Remove highly correlated features to reduce multicollinearity."""
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]
        print(f"Removing {len(to_drop)} highly correlated features")
        return X.drop(columns=to_drop)

    def apply_pca(self, X):
        """Apply PCA dimensionality reduction."""
        if self.n_components is None:
            self.n_components = min(X.shape[1], 15)
        self.pca = PCA(n_components=self.n_components)
        X_pca = self.pca.fit_transform(X)
        explained_var = sum(self.pca.explained_variance_ratio_)
        print(f"PCA: {self.n_components} components, {explained_var:.2%} variance explained")
        return pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(self.n_components)])

    def apply_smote(self, X, y, random_state=42):
        """Apply SMOTE oversampling to handle class imbalance."""
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        print(f"SMOTE applied: {X.shape[0]} -> {X_resampled.shape[0]} samples")
        print(f"Class distribution after SMOTE: {pd.Series(y_resampled).value_counts().to_dict()}")
        return X_resampled, y_resampled

    def preprocess_pipeline(self, X_train, X_test, y_train, use_smote=True):
        """Run the full preprocessing pipeline."""
        self.fit_scaler(X_train)
        X_train_scaled = self.transform_scale(X_train)
        X_test_scaled = self.transform_scale(X_test)

        if use_smote:
            X_train_scaled, y_train = self.apply_smote(X_train_scaled, y_train)

        return X_train_scaled, X_test_scaled, y_train
