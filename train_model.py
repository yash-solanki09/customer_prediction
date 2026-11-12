"""
Train the RFM customer segmentation model.

Usage:
    python train_model.py --data data/online_retail.csv

Outputs (created automatically):
    models/rfm_kmeans.joblib      -> scaler + KMeans + segment names (used by predict_customer.py)
    outputs/segment_profiles.csv  -> average R/F/M per segment
    outputs/elbow_plot.png
    outputs/segments_scatter.png
"""

import argparse
import datetime as dt
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["Recency", "Frequency", "Monetary"]


def load_and_clean(path):
    df = pd.read_csv(path, encoding="iso-8859-1")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # drop rows with no customer, then cancelled invoices (start with 'C') and bad rows
    df = df.dropna(subset=["CustomerID"]).copy()
    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()

    df["Sales"] = df["Quantity"] * df["UnitPrice"]
    return df


def build_rfm(df):
    # snapshot = one day after the last transaction in the data
    snapshot_date = df["InvoiceDate"].max() + dt.timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Sales", "sum"),
    )
    return rfm


def name_segments(centers):
    """
    KMeans cluster numbers are arbitrary (cluster 1 isn't 'best' just because it's 1),
    so I name segments from the centroids instead of hard-coding them.
    centers: array of shape (k, 3) in scaled [Recency, Frequency, Monetary] space.
    """
    k = len(centers)
    if k != 4:
        return {i: f"Segment {i}" for i in range(k)}

    # good customer = low recency, high frequency, high monetary
    score = -centers[:, 0] + centers[:, 1] + centers[:, 2]
    order = list(np.argsort(score)[::-1])

    names = {int(order[0]): "Best Customers", int(order[3]): "Churned / At-Risk"}
    mid = order[1:3]
    loyal = mid[int(np.argmax(centers[mid, 1]))]  # higher frequency of the two
    other = mid[0] if mid[1] == loyal else mid[1]
    names[int(loyal)] = "Loyal Customers"
    names[int(other)] = "New / Promising"
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/online_retail.csv")
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    Path("models").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)

    print("Loading and cleaning data...")
    df = load_and_clean(args.data)
    rfm = build_rfm(df)
    print(f"{len(rfm)} customers after cleaning")

    # log transform because R/F/M are heavily right-skewed, then scale
    rfm_log = np.log1p(rfm[FEATURES])
    scaler = StandardScaler()
    X = scaler.fit_transform(rfm_log)

    # k selection: elbow + silhouette
    inertia, sil = {}, {}
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        inertia[k] = km.inertia_
        sil[k] = silhouette_score(X, km.labels_)
    print("Silhouette by k:", {k: round(v, 3) for k, v in sil.items()})

    kmeans = KMeans(n_clusters=args.k, random_state=42, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(X)
    segment_names = name_segments(kmeans.cluster_centers_)
    rfm["Segment"] = rfm["Cluster"].map(segment_names)

    profiles = (
        rfm.groupby("Segment")
        .agg(
            Customers=("Cluster", "size"),
            Recency_Mean=("Recency", "mean"),
            Frequency_Mean=("Frequency", "mean"),
            Monetary_Mean=("Monetary", "mean"),
        )
        .sort_values("Monetary_Mean", ascending=False)
        .round(2)
    )
    print("\nSegment profiles:")
    print(profiles)
    profiles.to_csv("outputs/segment_profiles.csv")

    # plots
    plt.figure(figsize=(8, 5))
    plt.plot(list(inertia.keys()), list(inertia.values()), marker="o")
    plt.axvline(args.k, color="r", linestyle="--", label=f"Chosen k={args.k}")
    plt.title("Elbow Method")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/elbow_plot.png", dpi=150)
    plt.close()

    plt.figure(figsize=(11, 6))
    sns.scatterplot(data=rfm, x="Recency", y="Monetary", hue="Segment", style="Segment", alpha=0.6, s=60)
    plt.yscale("log")
    plt.title("Customer Segments: Recency vs Monetary")
    plt.xlabel("Recency (days since last purchase)")
    plt.ylabel("Monetary (total spend, log scale)")
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig("outputs/segments_scatter.png", dpi=150)
    plt.close()

    # save everything predict_customer.py needs.
    # log_mean / log_cov are used to generate realistic random customers.
    bundle = {
        "scaler": scaler,
        "kmeans": kmeans,
        "segment_names": segment_names,
        "features": FEATURES,
        "log_mean": rfm_log.mean().values,
        "log_cov": np.cov(rfm_log.values.T),
        "max_recency": int(rfm["Recency"].max()),
    }
    joblib.dump(bundle, "models/rfm_kmeans.joblib")
    print("\nSaved model to models/rfm_kmeans.joblib")


if __name__ == "__main__":
    main()
