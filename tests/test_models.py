"""Unit tests for anomaly detection models."""

import unittest
import numpy as np
from src.models import IsolationForestDetector, LOFDetector, ModelFactory


class TestIsolationForest(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.normal_data = np.random.randn(200, 5)
        self.anomalous_data = np.random.randn(10, 5) + 5

    def test_fit_predict(self):
        detector = IsolationForestDetector(n_estimators=50, contamination=0.05)
        detector.fit(self.normal_data)
        predictions = detector.predict(self.normal_data)
        self.assertEqual(predictions.shape[0], 200)
        self.assertTrue(set(predictions).issubset({0, 1}))

    def test_anomaly_scores(self):
        detector = IsolationForestDetector(n_estimators=50)
        detector.fit(self.normal_data)
        scores = detector.get_scores(self.normal_data)
        self.assertEqual(scores.shape[0], 200)

    def test_detects_anomalies(self):
        detector = IsolationForestDetector(n_estimators=100, contamination=0.1)
        X = np.vstack([self.normal_data, self.anomalous_data])
        detector.fit(X)
        scores_normal = detector.get_scores(self.normal_data).mean()
        scores_anomalous = detector.get_scores(self.anomalous_data).mean()
        self.assertGreater(scores_anomalous, scores_normal)


class TestLOF(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X_train = np.random.randn(200, 5)

    def test_fit_predict(self):
        detector = LOFDetector(n_neighbors=10, contamination=0.05)
        detector.fit(self.X_train)
        predictions = detector.predict(self.X_train)
        self.assertEqual(predictions.shape[0], 200)


class TestModelFactory(unittest.TestCase):
    def test_create_isolation_forest(self):
        model = ModelFactory.create("isolation_forest", n_estimators=50)
        self.assertIsInstance(model, IsolationForestDetector)

    def test_create_lof(self):
        model = ModelFactory.create("lof", n_neighbors=10)
        self.assertIsInstance(model, LOFDetector)

    def test_invalid_model(self):
        with self.assertRaises(ValueError):
            ModelFactory.create("invalid_model")

    def test_list_models(self):
        models = ModelFactory.list_models()
        self.assertIn("isolation_forest", models)
        self.assertIn("lof", models)


if __name__ == "__main__":
    unittest.main()
