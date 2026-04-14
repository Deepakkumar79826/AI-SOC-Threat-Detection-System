from src.data_loader import load_data, convert_to_binary_label
from src.train import train_model
from src.evaluate import evaluate_model
from src.detect import generate_alerts
from src.visualize import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_alert_severity,
    plot_feature_importance,
)
from src.utils import print_section

def main():
    print_section("STEP 1: LOADING DATA")
    train_df, test_df = load_data()
    train_df = convert_to_binary_label(train_df)
    test_df = convert_to_binary_label(test_df)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    print_section("STEP 2: TRAINING MODEL")
    model = train_model(train_df)
    print("Model trained and saved successfully.")

    print_section("STEP 3: EVALUATION")
    metrics, results_df = evaluate_model(model, test_df)
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1_score']:.4f}")

    print_section("STEP 4: ALERT GENERATION")
    predictions_df, alerts_df = generate_alerts(results_df)
    print("Total predictions:", len(predictions_df))
    print("Total alerts generated:", len(alerts_df))

    print_section("STEP 5: VISUALIZATION")
    plot_class_distribution(train_df)
    plot_confusion_matrix(results_df["actual"], results_df["predicted"])
    plot_alert_severity(alerts_df)
    X_columns = train_df.drop(columns=["label", "difficulty", "target"]).columns.tolist()
    plot_feature_importance(model, X_columns)
    print("Graphs saved in images/")

    print_section("PIPELINE COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    main()