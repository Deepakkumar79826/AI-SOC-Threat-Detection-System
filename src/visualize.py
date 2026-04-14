import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.config import IMAGES_DIR

def plot_class_distribution(train_df: pd.DataFrame):
    plt.figure(figsize=(6, 4))
    sns.countplot(x="target", data=train_df)
    plt.title("Class Distribution (0 = Normal, 1 = Attack)")
    plt.xlabel("Target")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "class_distribution.png")
    plt.close()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "confusion_matrix.png")
    plt.close()

def plot_alert_severity(alerts_df: pd.DataFrame):
    if alerts_df.empty:
        return
    plt.figure(figsize=(7, 4))
    sns.countplot(x="severity", data=alerts_df, order=["Low", "Medium", "High", "Critical"])
    plt.title("Alert Severity Distribution")
    plt.xlabel("Severity")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "alert_severity.png")
    plt.close()

def plot_feature_importance(model, X_columns):
    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]

    cat_features = preprocessor.named_transformers_["cat"].get_feature_names_out(["protocol_type", "service", "flag"])
    num_features = [col for col in X_columns if col not in ["protocol_type", "service", "flag"]]
    all_features = list(num_features) + list(cat_features)

    importances = classifier.feature_importances_
    importance_df = pd.DataFrame({
        "feature": all_features,
        "importance": importances
    }).sort_values(by="importance", ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x="importance", y="feature")
    plt.title("Top 15 Important Features")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "feature_importance.png")
    plt.close()