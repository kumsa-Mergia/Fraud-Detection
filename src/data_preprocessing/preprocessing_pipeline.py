import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os


class DataPreprocessor:
    def __init__(self, fraud_path, credit_path, ip_country_path):
        self.fraud_path = fraud_path
        self.credit_path = credit_path
        self.ip_country_path = ip_country_path

    def load_data(self):
        self.fraud_df = pd.read_csv(self.fraud_path)
        self.credit_df = pd.read_csv(self.credit_path)
        self.ip_country_df = pd.read_csv(self.ip_country_path)

    def clean_fraud_data(self):
        df = self.fraud_df.copy()

        # Handle missing values
        df.dropna(inplace=True)

        # Remove duplicates
        df.drop_duplicates(inplace=True)

        # Convert timestamps to datetime
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])

        # Feature Engineering
        df['hour_of_day'] = df['purchase_time'].dt.hour
        df['day_of_week'] = df['purchase_time'].dt.dayofweek
        df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600.0

        # Convert IP to integer
        df['ip_address_int'] = df['ip_address'].apply(self.ip_to_int)
        self.ip_country_df['lower_bound_ip_address'] = self.ip_country_df['lower_bound_ip_address'].apply(self.ip_to_int)
        self.ip_country_df['upper_bound_ip_address'] = self.ip_country_df['upper_bound_ip_address'].apply(self.ip_to_int)

        # Merge with geolocation
        df = self.merge_ip_country(df)

        # Encode categorical features
        df = pd.get_dummies(df, columns=['source', 'browser', 'sex'], drop_first=True)

        # Select features to keep
        features_to_keep = ['purchase_value', 'age', 'hour_of_day', 'day_of_week', 'time_since_signup', 'country'] + \
                           [col for col in df.columns if col.startswith(('source_', 'browser_', 'sex_'))] + ['class']
        df = df[features_to_keep]

        # Encode country
        df = pd.get_dummies(df, columns=['country'], drop_first=True)

        # Normalize numerical features
        scaler = StandardScaler()
        num_cols = ['purchase_value', 'age', 'hour_of_day', 'day_of_week', 'time_since_signup']
        df[num_cols] = scaler.fit_transform(df[num_cols])

        self.cleaned_fraud_df = df

        # Save cleaned fraud data
        os.makedirs("data", exist_ok=True)
        df.to_csv("../data/processed/cleaned_fraud_data.csv", index=False)

    def clean_credit_data(self):
        df = self.credit_df.copy()

        # Handle missing values
        df.dropna(inplace=True)

        # Remove duplicates
        df.drop_duplicates(inplace=True)

        # Normalize and scale
        scaler = StandardScaler()
        df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])

        self.cleaned_credit_df = df

        # Save cleaned credit data
        os.makedirs("data", exist_ok=True)
        df.to_csv("../data/processed/cleaned_credit_data.csv", index=False)

    def ip_to_int(self, ip):
        try:
            parts = list(map(int, ip.split('.')))
            return (parts[0] << 24) + (parts[1] << 16) + (parts[2] << 8) + parts[3]
        except:
            return np.nan

    def merge_ip_country(self, df):
        df['country'] = df['ip_address_int'].apply(self.find_country)
        return df

    def find_country(self, ip_int):
        match = self.ip_country_df[(self.ip_country_df['lower_bound_ip_address'] <= ip_int) &
                                   (self.ip_country_df['upper_bound_ip_address'] >= ip_int)]
        if not match.empty:
            return match.iloc[0]['country']
        return 'Unknown'

    def apply_smote(self):
        X = self.cleaned_fraud_df.drop('class', axis=1)
        y = self.cleaned_fraud_df['class']
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=['class'])], axis=1)


if __name__ == "__main__":
    processor = DataPreprocessor(
        fraud_path="../../data/raw/Fraud_Data.csv",
        credit_path="../../data/raw/creditcard.csv",
        ip_country_path="../../data/raw/IpAddress_to_Country.csv"
    )

    processor.load_data()
    processor.clean_fraud_data()
    processor.clean_credit_data()

    # Save SMOTE-applied data if needed
    balanced_df = processor.apply_smote()
    balanced_df.to_csv("../../data/processed/cleaned_balanced_fraud_data.csv", index=False)
