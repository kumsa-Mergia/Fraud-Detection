# src/models/base_model.py
from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, model_name):
        self.model = None
        self.model_name = model_name

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass


    @abstractmethod
    def predict_proba(self, X_test):
        pass