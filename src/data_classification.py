import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# import file 
from data_ingestion import data_ingestion
from preprocessing import build_preprocessor


# =====================
# CONFIG
# =====================
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

mlflow.set_experiment("UTS_Classification_MD_Assignment")


def train():
    # Load data
    df = data_ingestion()


    TARGET_COL = "placement_status"

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Preprocessing
    preprocessor = build_preprocessor(X_train)

    # Model candidates
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier()
    }

    best_score = 0
    best_model = None
    best_name = ""

    for name, model in models.items():
        with mlflow.start_run(run_name=name):

            pipeline = Pipeline([
                ("preprocessing", preprocessor),
                ("model", model)
            ])

            # Train
            pipeline.fit(X_train, y_train)

            # Predict
            preds = pipeline.predict(X_test)

            # Metrics
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="weighted")

            # Logging
            mlflow.log_param("model_name", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)

            mlflow.sklearn.log_model(pipeline, "model")

            print(f"{name} -> Acc: {acc:.4f}, F1: {f1:.4f}")

            # Save best
            if acc > best_score:
                best_score = acc
                best_model = pipeline
                best_name = name

    # Save best model ke .pkl
    best_class_model_path = MODEL_DIR / "best_classmodel.pkl"
    joblib.dump(best_model, best_class_model_path)

    print(f"\nBest model: {best_name}")
    print(f"Saved to: {best_class_model_path}")


if __name__ == "__main__":
    train()