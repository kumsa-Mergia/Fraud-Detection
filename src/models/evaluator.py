import matplotlib.pyplot as plt
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, f1_score,
                             precision_recall_curve)


class Evaluator:
    @staticmethod
    def evaluate(model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        auc_pr = average_precision_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        print(f"\n--- Evaluation: {model.model_name} ---")
        print(f"AUC-PR: {auc_pr:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        plt.figure(figsize=(6, 4))
        plt.plot(recall, precision, label=f"{model.model_name} PR Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve for {model.model_name}")
        plt.legend()
        plt.show()

        return {"auc_pr": auc_pr, "f1": f1, "confusion_matrix": cm}
