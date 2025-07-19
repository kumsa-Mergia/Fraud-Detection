from xgboost import XGBClassifier


class XGBoostModel:
    def __init__(self, **kwargs):
        """
        Initializes the XGBoost classifier with default or user-defined hyperparameters.
        """
        self.model = None
        self.model_name = "XGBoost"
        self.kwargs = kwargs  # Store additional user-defined parameters

    def train(self, X_train, y_train):
        # Compute class imbalance
        negative = (y_train == 0).sum()
        positive = (y_train == 1).sum()
        scale_pos_weight = negative / positive

        # Instantiate the model here so scale_pos_weight is dynamic
        self.model = XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbosity=0,
            **self.kwargs
        )

        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]
