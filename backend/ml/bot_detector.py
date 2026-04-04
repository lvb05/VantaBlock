#!/usr/bin/env python3
"""
Minimal Bot Detector – High‑Impact Feature Set
===============================================
Usage:
  python bot_detector.py --data ./sessions          # train
  python bot_detector.py --predict ./session.json   # predict with saved model
  python bot_detector.py --data ./sessions --predict ./session.json
"""

import os
import sys
import json
import math
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
import joblib

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ---------- Feature extraction ----------
def _dist(p1, p2):
    return math.hypot(p1["x"] - p2["x"], p1["y"] - p2["y"])

def extract_features(session: dict) -> dict:
    events = session.get("events", [])
    start_ms = session.get("start_time_ms", 0)
    if not events:
        return {f: 0.0 for f in feature_names}

    # 1. time_to_first_action
    first_action = None
    for e in events:
        if e["type"] in ("mousedown", "keydown", "click", "scroll"):
            first_action = e
            break
    time_to_first = (first_action["timestamp_ms"] - start_ms) if first_action else 0.0

    # 2. inter_event_time_std
    timestamps = [e["timestamp_ms"] for e in events]
    gaps = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
    inter_std = float(np.std(gaps)) if gaps else 0.0

    # 3. mouse_path_efficiency
    efficiencies = []
    for e in events:
        if e["type"] == "mousemove":
            path = e["data"].get("path", [])
            if len(path) >= 2:
                total_len = sum(_dist(path[i], path[i+1]) for i in range(len(path)-1))
                direct = _dist(path[0], path[-1])
                if total_len > 0:
                    efficiencies.append(direct / total_len)
    eff = np.mean(efficiencies) if efficiencies else 1.0   # neutral if no mouse

    # 4. velocity_variance
    velocities = []
    for e in events:
        if e["type"] == "mousemove":
            path = e["data"].get("path", [])
            for i in range(len(path)-1):
                d = _dist(path[i], path[i+1])
                dt = max(path[i+1]["t_ms"] - path[i]["t_ms"], 1)
                velocities.append(d / dt)
    vel_var = float(np.var(velocities)) if velocities else 0.0

    # 5. hover_time_before_click
    hover_times = []
    # map target -> last mousemove timestamp on that target
    last_hover = {}
    for e in events:
        if e["type"] == "mousemove":
            target = e["data"].get("target_element")
            if target:
                last_hover[target] = e["timestamp_ms"]
        elif e["type"] == "click":
            target = e["data"].get("target_element")
            if target and target in last_hover:
                hover_times.append(e["timestamp_ms"] - last_hover[target])
    avg_hover = np.mean(hover_times) if hover_times else 0.0

    # 6. scroll_variance
    scroll_deltas = []
    for e in events:
        if e["type"] == "scroll":
            steps = e["data"].get("steps", [])
            if len(steps) >= 2:
                for i in range(len(steps)-1):
                    delta = steps[i+1]["y"] - steps[i]["y"]
                    scroll_deltas.append(abs(delta))
    scroll_var = float(np.var(scroll_deltas)) if scroll_deltas else 0.0

    # 7. error_behavior: backspace rate
    keydowns = [e for e in events if e["type"] == "keydown"]
    backspaces = sum(1 for e in keydowns if e["data"].get("key") == "Backspace")
    backspace_rate = backspaces / max(len(keydowns), 1)

    # 8. consistency_score (variance of z-scores of the above 7 features)
    # We'll compute later after we have training statistics; for now return raw.
    # We'll compute z-scores in the pipeline using StandardScaler, then compute variance.
    # So we return raw features and compute consistency in the training step.
    # But for prediction we need to compute using stored mean/std.
    # We'll handle consistency as a separate post-processing step after scaling.

    return {
        "time_to_first_action": time_to_first,
        "inter_event_time_std": inter_std,
        "mouse_path_efficiency": eff,
        "velocity_variance": vel_var,
        "hover_time_before_click": avg_hover,
        "scroll_variance": scroll_var,
        "error_behavior": backspace_rate,
    }

feature_names = [
    "time_to_first_action",
    "inter_event_time_std",
    "mouse_path_efficiency",
    "velocity_variance",
    "hover_time_before_click",
    "scroll_variance",
    "error_behavior",
]

def add_consistency_score(X_df, scaler=None):
    """
    Compute consistency score as variance of z-scored features.
    If scaler is provided, use it to transform; else compute on-the-fly.
    Returns array of consistency scores.
    """
    # Z-score each feature
    if scaler is not None:
        X_scaled = scaler.transform(X_df[feature_names])
    else:
        # compute temporary scaler
        from sklearn.preprocessing import StandardScaler
        temp_scaler = StandardScaler()
        X_scaled = temp_scaler.fit_transform(X_df[feature_names])
    # variance across features for each row
    consistency = np.var(X_scaled, axis=1)
    return consistency

# ---------- Data loader ----------
def load_dataset(data_dir: str) -> pd.DataFrame:
    rows = []
    files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
    if not files:
        sys.exit(f"No JSON files in {data_dir}")

    for fname in files:
        fname_lower = fname.lower()
        if fname_lower.startswith("human"):
            label = 0
        elif fname_lower.startswith("agent") or fname_lower.startswith("bot"):
            label = 1
        else:
            continue   # skip unknown naming

        with open(os.path.join(data_dir, fname)) as f:
            session = json.load(f)
        feats = extract_features(session)
        feats["label"] = label
        feats["_filename"] = fname
        rows.append(feats)

    df = pd.DataFrame(rows).fillna(0)
    return df

# ---------- Training ----------
def train(data_dir, model_path="bot_model.pkl"):
    print(f"Loading data from {data_dir}...")
    df = load_dataset(data_dir)
    if df.empty:
        sys.exit("No labeled data found.")

    n_human = (df["label"] == 0).sum()
    n_bot = (df["label"] == 1).sum()
    print(f"Sessions: {len(df)} (human={n_human}, bot={n_bot})")

    X_raw = df[feature_names]
    y = df["label"]

    # Add consistency score as 8th feature
    consistency = add_consistency_score(X_raw)
    X = X_raw.copy()
    X["consistency_score"] = consistency

    # Save feature list (now 8)
    final_features = feature_names + ["consistency_score"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validation on Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=min(5, n_human, n_bot), shuffle=True, random_state=42)
    scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring="roc_auc")
    print(f"Random Forest CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    # Train final Random Forest
    rf.fit(X_scaled, y)

    # Optional XGBoost
    if XGB_AVAILABLE:
        xgb = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42, use_label_encoder=False, eval_metric="logloss")
        xgb_scores = cross_val_score(xgb, X_scaled, y, cv=cv, scoring="roc_auc")
        print(f"XGBoost CV AUC: {xgb_scores.mean():.4f} ± {xgb_scores.std():.4f}")
        xgb.fit(X_scaled, y)
    else:
        xgb = None
        print("XGBoost not installed – using only Random Forest")

    # Feature importances
    importances = pd.Series(rf.feature_importances_, index=final_features).sort_values(ascending=False)
    print("\nFeature importances (Random Forest):")
    for feat, imp in importances.items():
        print(f"  {feat:<28} {imp:.4f}")

    # Save model components
    model_data = {
        "rf": rf,
        "xgb": xgb,
        "scaler": scaler,
        "features": final_features,
    }
    joblib.dump(model_data, model_path)
    print(f"\nModel saved to {model_path}")

    # Quick evaluation on training data
    pred_prob = rf.predict_proba(X_scaled)[:, 1]
    if xgb:
        xgb_prob = xgb.predict_proba(X_scaled)[:, 1]
        pred_prob = (pred_prob + xgb_prob) / 2
    pred = (pred_prob > 0.5).astype(int)
    print("\nTraining set performance:")
    print(classification_report(y, pred, target_names=["Human", "Bot"]))

    return model_data

# ---------- Prediction ----------
def predict(session_path, model_path="bot_model.pkl"):
    if not os.path.exists(model_path):
        sys.exit(f"Model not found: {model_path}")

    model_data = joblib.load(model_path)
    rf = model_data["rf"]
    xgb = model_data.get("xgb")
    scaler = model_data["scaler"]
    final_features = model_data["features"]

    with open(session_path) as f:
        session = json.load(f)

    feats = extract_features(session)
    X_raw = pd.DataFrame([feats])[feature_names]

    # Add consistency score using saved scaler (for z-scoring)
    # We need to compute z-scores of the 7 base features using the same scaler that was fit on training.
    # But note: scaler was fit on 8 features including consistency. We need to transform the 7 base features only.
    # Extract the sub-scaler for the 7 base features from the original scaler.
    # Simpler: recompute consistency using a temporary scaler fit on the 7 features? That would not match training.
    # Instead, we stored the scaler for the 8-feature matrix. For a single sample, we need to compute consistency using training mean/std of the 7 base features.
    # We can retrieve the mean and std from the scaler for the first 7 features.
    mean_7 = scaler.mean_[:7]
    std_7 = scaler.scale_[:7]
    # z-score the 7 base features
    z_7 = (X_raw.iloc[0].values - mean_7) / std_7
    consistency = np.var(z_7)   # variance of those 7 z-scores
    X = X_raw.copy()
    X["consistency_score"] = consistency

    # Ensure correct column order
    X = X[final_features]
    X_scaled = scaler.transform(X)

    # Predict probabilities
    rf_prob = rf.predict_proba(X_scaled)[0, 1]   # bot probability
    if xgb:
        xgb_prob = xgb.predict_proba(X_scaled)[0, 1]
        prob = (rf_prob + xgb_prob) / 2
    else:
        prob = rf_prob

    human_pct = (1 - prob) * 100
    bot_pct = prob * 100
    verdict = "BOT / AGENT" if prob > 0.5 else "HUMAN"

    print("\n" + "=" * 60)
    print("  PREDICTION RESULT")
    print("=" * 60)
    print(f"  Session: {session.get('session_id', session_path)}")
    print(f"  Human probability: {human_pct:.1f}%")
    print(f"  Bot probability:   {bot_pct:.1f}%")
    print(f"  Verdict: {verdict}")
    print(f"  Confidence: {max(human_pct, bot_pct):.1f}%")
    print("=" * 60)

    # Signal breakdown using the 7 base features (optional)
    print("\n  Feature values:")
    for f in feature_names:
        val = feats[f]
        status = "⚠️ bot-like" if ((f in ["mouse_path_efficiency"] and val > 0.95) or
                                   (f in ["time_to_first_action"] and val < 50) or
                                   (f in ["inter_event_time_std"] and val < 30) or
                                   (f in ["velocity_variance"] and val < 50) or
                                   (f in ["hover_time_before_click"] and val < 50) or
                                   (f in ["scroll_variance"] and val < 100) or
                                   (f in ["error_behavior"] and val == 0)) else "✓ human-like"
        print(f"    {f:<25} {val:>10.2f}   {status}")

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Minimal high-impact bot detector")
    parser.add_argument("--data", help="Folder with human_*.json and agent_*.json")
    parser.add_argument("--predict", help="Single session JSON to classify")
    parser.add_argument("--model", default="bot_model.pkl", help="Model file path")
    args = parser.parse_args()

    if not args.data and not args.predict:
        parser.print_help()
        return

    if args.data:
        train(args.data, args.model)

    if args.predict:
        predict(args.predict, args.model)

if __name__ == "__main__":
    main()