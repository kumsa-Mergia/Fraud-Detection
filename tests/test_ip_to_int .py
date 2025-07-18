import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing.preprocessing_pipeline import DataPreprocessor

class TestIPConversion(unittest.TestCase):

    def setUp(self):
        # We just need to instantiate the class with dummy paths because
        # we're only testing ip_to_int here.
        self.processor = DataPreprocessor(
            fraud_path="dummy.csv",
            credit_path="dummy.csv",
            ip_country_path="dummy.csv"
        )

    def test_ip_to_int(self):
        ip = "192.168.1.1"
        expected = (192 << 24) + (168 << 16) + (1 << 8) + 1
        result = self.processor.ip_to_int(ip)
        self.assertEqual(result, expected)

    def test_ip_to_int_invalid(self):
        ip = "not.an.ip"
        result = self.processor.ip_to_int(ip)
        self.assertTrue(result != result)  # NaN check: NaN != NaN

if __name__ == "__main__":
    unittest.main()
