# tests/test_xgboost_model.py

import os
import sys
import unittest

import numpy as np
from sklearn.datasets import make_classification

# Add project root so import works
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.models.xgboost_model import XGBoostModel


class TestXGBoostModel(unittest.TestCase):
    def setUp(self):
        # Generate synthetic dataset
        self.X, self.y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_classes=2,
            random_state=42,
        )
        self.model = XGBoostModel()

    def test_training(self):
        # Check that training completes without error
        self.model.train(self.X, self.y)
        self.assertIsNotNone(self.model.model)

    def test_prediction_shape(self):
        self.model.train(self.X, self.y)
        preds = self.model.predict(self.X)
        self.assertEqual(len(preds), len(self.y))

    def test_prediction_values(self):
        self.model.train(self.X, self.y)
        preds = self.model.predict(self.X)
        self.assertTrue(np.all(np.isin(preds, [0, 1])))

    def test_predict_proba_range(self):
        self.model.train(self.X, self.y)
        proba = self.model.predict_proba(self.X)
        self.assertTrue(np.all((proba >= 0) & (proba <= 1)))


if __name__ == "__main__":
    unittest.main()
