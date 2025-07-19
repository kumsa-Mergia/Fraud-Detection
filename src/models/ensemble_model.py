from sklearn.ensemble import RandomForestClassifier
from src.models.base_model import BaseModel

class RandomForestModel(BaseModel):
    def __init__(self):
        super().__init__("Random Forest")
        self.model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]