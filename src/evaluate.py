import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from src.config import METRICS_FILE
from src.utils import save_json

def evaluate_model(model, test_df):
    X_test = test_df.drop(columns=["label", "difficulty", "target"])
    y_test = test_df["target"]

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }

    save_json(metrics, METRICS_FILE)

    results_df = X_test.copy()
    results_df["actual"] = y_test.values
    results_df["predicted"] = y_pred
    results_df["threat_probability"] = y_prob

    return metrics, results_df