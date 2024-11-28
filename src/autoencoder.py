"""Autoencoder-based anomaly detection for credit card fraud."""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks


class FraudAutoencoder:
    """Autoencoder model for unsupervised anomaly detection."""

    def __init__(self, input_dim, encoding_dim=14, hidden_layers=None):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers or [64, 32]
        self.model = None
        self.threshold = None
        self.history = None

    def build(self):
        """Build the autoencoder architecture."""
        encoder_dims = self.hidden_layers + [self.encoding_dim]
        decoder_dims = list(reversed(self.hidden_layers))

        inputs = keras.Input(shape=(self.input_dim,))
        x = inputs
        for dim in encoder_dims:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)

        for dim in decoder_dims:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.BatchNormalization()(x)

        outputs = layers.Dense(self.input_dim, activation="linear")(x)
        self.model = keras.Model(inputs, outputs, name="fraud_autoencoder")
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
        return self

    def train(self, X_train, X_val=None, epochs=50, batch_size=32):
        """Train the autoencoder on normal transactions."""
        cb = [
            callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
        ]
        val_data = (X_val, X_val) if X_val is not None else None
        self.history = self.model.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_data,
            callbacks=cb,
            verbose=1,
        )
        return self.history

    def get_reconstruction_errors(self, X):
        """Compute reconstruction error for each sample."""
        reconstructed = self.model.predict(X, verbose=0)
        errors = np.mean(np.square(X - reconstructed), axis=1)
        return errors

    def optimize_threshold(self, X_val, y_val):
        """Find optimal threshold using validation set."""
        errors = self.get_reconstruction_errors(X_val)
        best_f1 = 0
        best_threshold = 0
        for percentile in range(90, 100):
            threshold = np.percentile(errors, percentile)
            predictions = (errors > threshold).astype(int)
            tp = np.sum((predictions == 1) & (y_val == 1))
            fp = np.sum((predictions == 1) & (y_val == 0))
            fn = np.sum((predictions == 0) & (y_val == 1))
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        self.threshold = best_threshold
        print(f"Optimal threshold: {self.threshold:.6f} (F1: {best_f1:.4f})")
        return self.threshold

    def predict(self, X):
        """Predict anomalies using reconstruction error threshold."""
        if self.threshold is None:
            raise ValueError("Threshold not set. Call optimize_threshold() first.")
        errors = self.get_reconstruction_errors(X)
        return (errors > self.threshold).astype(int)

    def get_scores(self, X):
        """Get anomaly scores (reconstruction errors)."""
        return self.get_reconstruction_errors(X)
