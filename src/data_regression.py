import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


from sklearn.linear_model import Ridge as RidgeRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# import file
from data_ingestion import data_ingestion
from preprocessing import build_preprocessor


# =====================
# CONFIG
# =====================
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

mlflow.set_experiment("UTS_Regression_MD_Assignment")


def train():
    # Load data
    df = data_ingestion()

    TARGET_COL = "salary_lpa"

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
        "Ridge": RidgeRegression(max_iter=1000),
        "DecisionTree": DecisionTreeRegressor(),
        "RandomForest": RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
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
            mae = mean_absolute_error(y_test, preds)
            mse = mean_squared_error(y_test, preds)
            rmse = mse ** 0.5
            r2 = r2_score(y_test, preds)

            # Logging
            mlflow.log_param("model_name", name)
            mlflow.log_metric("mean_absolute_error", mae)
            mlflow.log_metric("mean_squared_error", mse)
            mlflow.log_metric("root_mean_squared_error", rmse)
            mlflow.log_metric("r2_score", r2)

            mlflow.sklearn.log_model(pipeline, "model")

            print(f"{name} -> MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

            # Save best
            if r2 > best_score:
                best_score = r2
                best_model = pipeline
                best_name = name

    # Save best model ke .pkl
    best_reg_model_path = MODEL_DIR / "best_regmodel.pkl"
    joblib.dump(best_model, best_reg_model_path, compress=3)

    print(f"\nBest model: {best_name}")
    print(f"Saved to: {best_reg_model_path}")


if __name__ == "__main__":
    train()