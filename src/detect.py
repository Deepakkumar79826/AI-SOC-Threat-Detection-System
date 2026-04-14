import pandas as pd
from src.config import ALERTS_FILE, PREDICTIONS_FILE

def assign_severity(prob: float) -> str:
    if prob >= 0.90:
        return "Critical"
    elif prob >= 0.75:
        return "High"
    elif prob >= 0.50:
        return "Medium"
    else:
        return "Low"

def generate_alerts(results_df: pd.DataFrame):
    df = results_df.copy()
    df["alert"] = df["predicted"].apply(lambda x: "Attack" if x == 1 else "Normal")
    df["severity"] = df["threat_probability"].apply(assign_severity)

    predictions_df = df.copy()
    predictions_df.to_csv(PREDICTIONS_FILE, index=False)

    alerts_df = df[df["predicted"] == 1].copy()
    alerts_df = alerts_df.sort_values(by="threat_probability", ascending=False)
    alerts_df.to_csv(ALERTS_FILE, index=False)

    return predictions_df, alerts_df