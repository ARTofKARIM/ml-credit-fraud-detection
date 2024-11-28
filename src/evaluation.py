"""Model evaluation and comparison module."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
)


class ModelEvaluator:
    """Evaluates and compares anomaly detection models."""

    def __init__(self):
        self.results = {}

    def evaluate(self, y_true, y_pred, scores, model_name):
        """Compute comprehensive evaluation metrics for a model."""
        report = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, scores)
        pr_auc = average_precision_score(y_true, scores)

        metrics = {
            "model_name": model_name,
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "f1_score": report["1"]["f1-score"],
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "confusion_matrix": cm,
            "classification_report": report,
        }
        self.results[model_name] = metrics
        print(f"\n{'='*50}")
        print(f"Model: {model_name}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"PR-AUC: {metrics['pr_auc']:.4f}")
        return metrics

    def get_roc_curve(self, y_true, scores):
        """Compute ROC curve data points."""
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        return fpr, tpr, thresholds

    def get_pr_curve(self, y_true, scores):
        """Compute Precision-Recall curve data points."""
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        return precision, recall, thresholds

    def comparison_table(self):
        """Generate a comparison table of all evaluated models."""
        rows = []
        for name, metrics in self.results.items():
            rows.append({
                "Model": name,
                "Precision": f"{metrics['precision']:.4f}",
                "Recall": f"{metrics['recall']:.4f}",
                "F1-Score": f"{metrics['f1_score']:.4f}",
                "ROC-AUC": f"{metrics['roc_auc']:.4f}",
                "PR-AUC": f"{metrics['pr_auc']:.4f}",
            })
        return pd.DataFrame(rows)

    def best_model(self, metric="f1_score"):
        """Return the name of the best performing model."""
        best_name = max(self.results, key=lambda k: self.results[k][metric])
        return best_name, self.results[best_name]
