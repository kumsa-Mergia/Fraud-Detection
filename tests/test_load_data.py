import os
import sys
import unittest
import pandas as pd
from faker import Faker

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_preprocessing.preprocessing_pipeline import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        # Initialize Faker instance
        fake = Faker()

        # Generate dummy data for fraud_df
        fraud_data = {
            'TransactionID': [fake.uuid4() for _ in range(10)],
            'Amount': [fake.random_number(digits=5) for _ in range(10)],
            'IsFraud': [fake.boolean() for _ in range(10)],
        }
        self.fraud_df = pd.DataFrame(fraud_data)

        # Generate dummy data for credit_df
        credit_data = {
            'CreditScore': [fake.random_int(min=300, max=850) for _ in range(10)],
            'CreditLimit': [fake.random_number(digits=6) for _ in range(10)],
            'Balance': [fake.random_number(digits=5) for _ in range(10)],
        }
        self.credit_df = pd.DataFrame(credit_data)

        # Generate dummy data for ip_country_df
        ip_country_data = {
            'IPAddress': [fake.ipv4() for _ in range(10)],
            'Country': [fake.country() for _ in range(10)],
        }
        self.ip_country_df = pd.DataFrame(ip_country_data)

        # Mock the DataPreprocessor to use the dummy data
        self.processor = DataPreprocessor(
            fraud_file_path=None,
            credit_file_path=None,
            ip_country_file_path=None,
        )
        self.processor.fraud_df = self.fraud_df
        self.processor.credit_df = self.credit_df
        self.processor.ip_country_df = self.ip_country_df

    def test_load_data(self):
        # Just test loading data runs without error
        self.processor.load_data()
        self.assertIsNotNone(self.processor.fraud_df)
        self.assertIsNotNone(self.processor.credit_df)
        self.assertIsNotNone(self.processor.ip_country_df)


if __name__ == "__main__":
    unittest.main()
    