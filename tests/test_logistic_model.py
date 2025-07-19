import unittest
import numpy as np
from sklearn.datasets import make_classification

import sys
import os

# Add project root so import works
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.models.logistic_model import LogisticModel  # adjust path if needed

class TestLogisticModel(unittest.TestCase):
    def setUp(self):
        # Generate synthetic binary classification data
        self.X, self.y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_classes=2,
            random_state=42
        )
        self.model = LogisticModel()

    def test_train_model(self):
        # Should train without errors
        self.model.train(self.X, self.y)
        self.assertIsNotNone(self.model.model)

    def test_predict_shape(self):
        self.model.train(self.X, self.y)
        predictions = self.model.predict(self.X)
        self.assertEqual(predictions.shape, self.y.shape)

    def test_predict_values(self):
        self.model.train(self.X, self.y)
        predictions = self.model.predict(self.X)
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))

    def test_predict_proba_range(self):
        self.model.train(self.X, self.y)
        probabilities = self.model.predict_proba(self.X)
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))
        self.assertEqual(probabilities.shape, self.y.shape)

if __name__ == '__main__':
    unittest.main()
