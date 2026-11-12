"""
Generate random customers and predict which segment they belong to.

Examples:
    python predict_customer.py                       # 5 random customers
    python predict_customer.py --n 10 --seed 42      # 10 random, reproducible
    python predict_customer.py --recency 5 --frequency 12 --monetary 4200   # your own customer

Run train_model.py first so models/rfm_kmeans.joblib exists.
"""

import argparse

import joblib
import numpy as np
import pandas as pd


def load_bundle(path="models/rfm_kmeans.joblib"):
    return joblib.load(path)


def generate_random_customers(bundle, n=5, seed=None):
    """
    Sample random (Recency, Frequency, Monetary) values from a multivariate normal
    fitted on the log-transformed training data. That keeps the correlations between
    R, F and M, so the fake customers look like real ones instead of three
    independent random numbers.
    """
    rng = np.random.default_rng(seed)
    log_vals = rng.multivariate_normal(bundle["log_mean"], bundle["log_cov"], size=n)
    vals = np.expm1(log_vals)

    return pd.DataFrame(
        {
            "Recency": np.clip(np.round(vals[:, 0]), 1, bundle["max_recency"]).astype(int),
            "Frequency": np.clip(np.round(vals[:, 1]), 1, None).astype(int),
            "Monetary": np.clip(np.round(vals[:, 2], 2), 1, None),
        }
    )


def predict_segments(bundle, customers):
    """Assign each customer to the nearest KMeans centroid (same preprocessing as training)."""
    features = bundle["features"]
    X = bundle["scaler"].transform(np.log1p(customers[features]))
    clusters = bundle["kmeans"].predict(X)

    out = customers.copy()
    out["Segment"] = [bundle["segment_names"][int(c)] for c in clusters]
    out["Distance_To_Centroid"] = bundle["kmeans"].transform(X).min(axis=1).round(2)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="how many random customers")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--recency", type=float, help="days since last purchase")
    parser.add_argument("--frequency", type=float, help="number of orders")
    parser.add_argument("--monetary", type=float, help="total spend")
    parser.add_argument("--model", default="models/rfm_kmeans.joblib")
    args = parser.parse_args()

    bundle = load_bundle(args.model)

    manual = [args.recency, args.frequency, args.monetary]
    if all(v is not None for v in manual):
        customers = pd.DataFrame(
            {"Recency": [args.recency], "Frequency": [args.frequency], "Monetary": [args.monetary]}
        )
    elif any(v is not None for v in manual):
        parser.error("give all three of --recency, --frequency and --monetary, or none of them")
    else:
        customers = generate_random_customers(bundle, args.n, args.seed)

    result = predict_segments(bundle, customers)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
