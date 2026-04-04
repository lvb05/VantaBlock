import csv
import joblib
import numpy as np
from pathlib import Path


def load_artifacts(models_dir: Path):
    model = joblib.load(models_dir / "best_model.pkl")
    scaler = joblib.load(models_dir / "scaler.pkl")
    metadata = joblib.load(models_dir / "model_metadata.pkl")
    feature_names = metadata.get("feature_names", [])
    if not feature_names:
        raise ValueError("model_metadata.pkl is missing feature_names")
    return model, scaler, metadata, feature_names


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    models_dir = base_dir.parent / "models"

    input_csv = base_dir / "features.csv"
    output_csv = base_dir / "predictions.csv"

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    model, scaler, metadata, feature_names = load_artifacts(models_dir)

    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("Input CSV is empty")

    missing = [name for name in feature_names if name not in rows[0]]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    feature_values = []
    for row in rows:
        feature_values.append([float(row[name]) for name in feature_names])

    features_scaled = scaler.transform(np.array(feature_values, dtype=float))

    predictions = model.predict(features_scaled)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features_scaled)
        prob_bot = probabilities[:, 0]
        prob_human = probabilities[:, 1]
    else:
        prob_bot = None
        prob_human = None

    output_rows = []
    for row, prediction in zip(rows, predictions):
        out_row = {k: v for k, v in row.items() if k != "is_human"}
        out_row["predicted_human"] = int(prediction)
        out_row["predicted_label"] = "HUMAN" if prediction == 1 else "BOT"
        output_rows.append(out_row)

    if prob_human is not None:
        for out_row, bot_prob, human_prob in zip(output_rows, prob_bot, prob_human):
            out_row["confidence_bot"] = float(bot_prob)
            out_row["confidence_human"] = float(human_prob)

    fieldnames = list(output_rows[0].keys())
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    total = len(output_rows)
    human_count = sum(1 for row in output_rows if int(row["predicted_human"]) == 1)
    bot_count = total - human_count

    print("Prediction completed")
    print(f"Model: {metadata.get('model_name', 'unknown')}")
    print(f"Input: {input_csv}")
    print(f"Output: {output_csv}")
    print(f"Total rows: {total}")
    print(f"Predicted HUMAN: {human_count}")
    print(f"Predicted BOT: {bot_count}")


if __name__ == "__main__":
    main()
