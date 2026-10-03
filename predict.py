import sys
import joblib
import pandas as pd


MODEL_PATH = "models/churn_random_forest.joblib"
THRESHOLD = 0.3


def prepare_data(df):
    df = df.copy()

    # Customer ID не используется моделью
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Приводим TotalCharges к числовому типу
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

        # Для клиентов с tenure = 0
        # TotalCharges принимаем равным 0
        if "tenure" in df.columns:
            df.loc[
                df["tenure"] == 0,
                "TotalCharges"
            ] = 0

    return df


def predict(input_path):
    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(input_path)

    X = prepare_data(df)

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    result = df.copy()

    result["Churn_Probability"] = probabilities
    result["Churn_Prediction"] = predictions

    return result


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python predict.py <input.csv>")
        sys.exit(1)

    input_path = sys.argv[1]

    result = predict(input_path)

    print(
        result[
            [
                "Churn_Probability",
                "Churn_Prediction"
            ]
        ]
    )