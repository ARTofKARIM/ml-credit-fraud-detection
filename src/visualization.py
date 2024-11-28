"""Visualization module for fraud detection results."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.manifold import TSNE


class FraudVisualizer:
    """Generates publication-quality plots for fraud detection analysis."""

    def __init__(self, output_dir="results/plots/"):
        self.output_dir = output_dir
        plt.style.use("seaborn-v0_8-whitegrid")

    def plot_roc_curves(self, y_true, model_scores, model_names, save=True):
        """Plot ROC curves for multiple models on the same figure."""
        fig, ax = plt.subplots(figsize=(8, 6))
        for scores, name in zip(model_scores, model_names):
            fpr, tpr, _ = roc_curve(y_true, scores)
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_true, scores)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curves - Model Comparison", fontsize=14)
        ax.legend(loc="lower right", fontsize=10)
        if save:
            fig.savefig(f"{self.output_dir}roc_curves.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_confusion_matrix(self, y_true, y_pred, model_name, save=True):
        """Plot confusion matrix as a heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Normal", "Fraud"], yticklabels=["Normal", "Fraud"])
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_title(f"Confusion Matrix - {model_name}", fontsize=14)
        if save:
            safe_name = model_name.lower().replace(" ", "_")
            fig.savefig(f"{self.output_dir}cm_{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_feature_importance(self, importances, feature_names, top_n=15, save=True):
        """Plot top N feature importances."""
        indices = np.argsort(importances)[-top_n:]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(top_n), importances[indices], color="steelblue")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel("Importance", fontsize=12)
        ax.set_title("Top Feature Importances", fontsize=14)
        if save:
            fig.savefig(f"{self.output_dir}feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_tsne_embedding(self, X, y, n_samples=2000, save=True):
        """Plot t-SNE 2D embedding colored by fraud/normal."""
        if X.shape[0] > n_samples:
            indices = np.random.choice(X.shape[0], n_samples, replace=False)
            X_sample = X[indices] if isinstance(X, np.ndarray) else X.iloc[indices]
            y_sample = y[indices] if isinstance(y, np.ndarray) else y.iloc[indices]
        else:
            X_sample, y_sample = X, y
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_embedded = tsne.fit_transform(X_sample if isinstance(X_sample, np.ndarray) else X_sample.values)
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(X_embedded[:, 0], X_embedded[:, 1],
                           c=y_sample, cmap="coolwarm", alpha=0.6, s=10)
        plt.colorbar(scatter, ax=ax, label="Class (0=Normal, 1=Fraud)")
        ax.set_title("t-SNE Visualization of Transactions", fontsize=14)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        if save:
            fig.savefig(f"{self.output_dir}tsne_embedding.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_score_distribution(self, scores, y_true, model_name, save=True):
        """Plot anomaly score distributions for normal vs fraud transactions."""
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(scores[y_true == 0], bins=50, alpha=0.6, label="Normal", color="steelblue", density=True)
        ax.hist(scores[y_true == 1], bins=50, alpha=0.6, label="Fraud", color="red", density=True)
        ax.set_xlabel("Anomaly Score", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"Score Distribution - {model_name}", fontsize=14)
        ax.legend(fontsize=10)
        if save:
            safe_name = model_name.lower().replace(" ", "_")
            fig.savefig(f"{self.output_dir}scores_{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
