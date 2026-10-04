import sys
import pandas as pd
import joblib


MODEL_PATH = "models/churn_random_forest.joblib"


def predict(input_path):
    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    threshold = bundle["threshold"]

    df = pd.read_csv(input_path)

    customer_ids = df["customerID"] if "customerID" in df.columns else None

    X = df.drop(
        columns=["customerID"],
        errors="ignore"
    )

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    result = pd.DataFrame({
        "Churn_Probability": probabilities,
        "Churn_Prediction": predictions
    })

    if customer_ids is not None:
        result.insert(
            0,
            "customerID",
            customer_ids
        )

    return result


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(
            "Usage: python predict.py <input_csv>"
        )
        sys.exit(1)

    input_path = sys.argv[1]

    predictions = predict(input_path)

    print(predictions.head())