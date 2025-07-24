import shap
import matplotlib.pyplot as plt

class SHAPExplainer:
    def __init__(self, model, X_train, X_test, model_type="tree"):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None

    def fit(self):
        if self.model_type == "tree":
            self.explainer = shap.TreeExplainer(self.model)
        elif self.model_type == "linear":
            self.explainer = shap.LinearExplainer(self.model, self.X_train)
        else:
            self.explainer = shap.Explainer(self.model, self.X_train)
        
        self.shap_values = self.explainer(self.X_test)
        return self.shap_values

    def plot_summary(self, plot_type="bar"):
        if self.shap_values is not None:
            shap.summary_plot(self.shap_values, self.X_test, plot_type=plot_type)
        else:
            print("Please run .fit() before plotting.")

    def plot_force(self, index=0):
        if self.shap_values is not None:
            shap.plots.force(self.shap_values[index])
        else:
            print("Please run .fit() before plotting.")
