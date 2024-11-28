"""Main entry point for the credit card fraud detection pipeline."""

import argparse
import yaml
import os
from src.data_loader import CreditCardDataLoader
from src.feature_engineering import FeatureEngineer
from src.preprocessing import DataPreprocessor
from src.models import ModelFactory
from src.autoencoder import FraudAutoencoder
from src.evaluation import ModelEvaluator


def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection Pipeline")
    parser.add_argument("--data", type=str, help="Path to dataset CSV")
    parser.add_argument("--model", type=str, choices=["all", "isolation_forest", "lof", "autoencoder"],
                        default="all", help="Model to train")
    parser.add_argument("--no-smote", action="store_true", help="Disable SMOTE oversampling")
    args = parser.parse_args()

    config = load_config()

    # Load data
    loader = CreditCardDataLoader()
    loader.load(args.data)
    loader.describe()
    X_train, X_test, y_train, y_test = loader.split()

    # Feature engineering
    engineer = FeatureEngineer()
    X_train = engineer.engineer_features(X_train)
    X_test = engineer.engineer_features(X_test)

    # Preprocessing
    preprocessor = DataPreprocessor(scaling_method="robust")
    X_train_processed, X_test_processed, y_train_processed = preprocessor.preprocess_pipeline(
        X_train, X_test, y_train, use_smote=not args.no_smote
    )

    # Evaluation
    evaluator = ModelEvaluator()

    # Train traditional models
    if args.model in ["all", "isolation_forest"]:
        iso_forest = ModelFactory.create("isolation_forest", **config["models"]["isolation_forest"])
        iso_forest.fit(X_train_processed)
        y_pred = iso_forest.predict(X_test_processed)
        scores = iso_forest.get_scores(X_test_processed)
        evaluator.evaluate(y_test, y_pred, scores, "Isolation Forest")

    if args.model in ["all", "lof"]:
        lof = ModelFactory.create("lof", **config["models"]["lof"])
        lof.fit(X_train_processed)
        y_pred = lof.predict(X_test_processed)
        scores = lof.get_scores(X_test_processed)
        evaluator.evaluate(y_test, y_pred, scores, "LOF")

    if args.model in ["all", "autoencoder"]:
        ae_config = config["models"]["autoencoder"]
        ae = FraudAutoencoder(X_train_processed.shape[1], encoding_dim=ae_config["encoding_dim"])
        ae.build()
        ae.train(X_train_processed.values if hasattr(X_train_processed, 'values') else X_train_processed,
                 epochs=ae_config["epochs"], batch_size=ae_config["batch_size"])
        ae.optimize_threshold(X_test_processed.values if hasattr(X_test_processed, 'values') else X_test_processed, y_test)
        y_pred = ae.predict(X_test_processed.values if hasattr(X_test_processed, 'values') else X_test_processed)
        scores = ae.get_scores(X_test_processed.values if hasattr(X_test_processed, 'values') else X_test_processed)
        evaluator.evaluate(y_test, y_pred, scores, "Autoencoder")

    # Summary
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(evaluator.comparison_table().to_string(index=False))
    best_name, _ = evaluator.best_model()
    print(f"\nBest model: {best_name}")


if __name__ == "__main__":
    main()
