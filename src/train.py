import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.config import MODEL_FILE, PREPROCESSOR_FILE, RANDOM_STATE
from src.preprocess import build_preprocessor, prepare_features

def train_model(train_df):
    X_train, y_train = prepare_features(train_df)
    preprocessor, _ = build_preprocessor(train_df)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    # Save complete pipeline
    joblib.dump(pipeline, MODEL_FILE)
    joblib.dump(preprocessor, PREPROCESSOR_FILE)

    return pipeline