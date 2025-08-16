import os
import sys
import unittest

# Add project root so import works
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_preprocessing.preprocessing_pipeline import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):

    def setUp(self):
        self.processor = DataPreprocessor(
            "data/raw/Fraud_Data.csv",
            "data/raw/creditcard.csv",
            "data/raw/IpAddress_to_Country.csv",
        )

    def test_load_data(self):
        # Just test loading data runs without error
        self.processor.load_data()
        self.assertIsNotNone(self.processor.fraud_df)
        self.assertIsNotNone(self.processor.credit_df)
        self.assertIsNotNone(self.processor.ip_country_df)


if __name__ == "__main__":
    unittest.main()
