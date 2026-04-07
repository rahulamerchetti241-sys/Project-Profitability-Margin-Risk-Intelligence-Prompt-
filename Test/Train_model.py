import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = "project_profitability_data.csv"
MODEL_PATH = "profit_margin_model.joblib"
META_PATH = "model_meta.joblib"

FEATURES = [
    "budget",
    "duration_months",
    "labor_cost",
    "material_cost",
    "overhead",
    "actual_cost",
    "delay_pct",
    "resource_utilization_pct",
]
TARGET = "profit_margin_pct"


def train():
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"Model trained. MAE: {mae:.2f}, R2: {r2:.3f}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump({"features": FEATURES, "target": TARGET}, META_PATH)
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved metadata to: {META_PATH}")


if __name__ == "__main__":
    train()