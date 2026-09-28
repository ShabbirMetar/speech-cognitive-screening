"""Generate TRAIN-only diagnostics for the manual linguistic feature table."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.baseline_logistic import encode_screening_labels, select_feature_columns, training_rows


FEATURE_TABLE_PATH = PROJECT_ROOT / "artifacts" / "features" / "manual_transcript_features.csv"
OUTPUT_DIRECTORY = PROJECT_ROOT / "artifacts" / "results" / "feature_analysis"
CORRELATION_THRESHOLD = 0.90


def cohens_d(impaired: pd.Series, healthy: pd.Series) -> float:
    """Return the standardized impaired-minus-healthy mean difference, or NaN."""

    impaired = impaired.dropna()
    healthy = healthy.dropna()
    if len(impaired) < 2 or len(healthy) < 2:
        return float("nan")
    pooled_variance = (
        (len(impaired) - 1) * impaired.var(ddof=1)
        + (len(healthy) - 1) * healthy.var(ddof=1)
    ) / (len(impaired) + len(healthy) - 2)
    if not np.isfinite(pooled_variance) or pooled_variance <= 0:
        return float("nan")
    return float((impaired.mean() - healthy.mean()) / np.sqrt(pooled_variance))


def feature_missingness_and_variance(features: pd.DataFrame) -> pd.DataFrame:
    """Summarize missingness and observed TRAIN-only variance per feature."""

    return pd.DataFrame(
        {
            "feature": features.columns,
            "missing_count": features.isna().sum().to_numpy(),
            "missing_percentage": (features.isna().mean() * 100).to_numpy(),
            "variance": features.var(ddof=1).to_numpy(),
        }
    ).sort_values("feature").reset_index(drop=True)


def high_correlation_pairs(correlation: pd.DataFrame, threshold: float = CORRELATION_THRESHOLD) -> pd.DataFrame:
    """List unique Pearson pairs whose absolute correlation meets the threshold."""

    pairs: list[dict[str, float | str]] = []
    for left_index, left_name in enumerate(correlation.columns):
        for right_name in correlation.columns[left_index + 1 :]:
            value = correlation.loc[left_name, right_name]
            if pd.notna(value) and abs(value) >= threshold:
                pairs.append(
                    {
                        "feature_a": left_name,
                        "feature_b": right_name,
                        "pearson_correlation": float(value),
                        "absolute_pearson_correlation": float(abs(value)),
                    }
                )
    return pd.DataFrame(pairs).sort_values(
        "absolute_pearson_correlation", ascending=False
    ).reset_index(drop=True) if pairs else pd.DataFrame(
        columns=["feature_a", "feature_b", "pearson_correlation", "absolute_pearson_correlation"]
    )


def feature_summary_by_label(features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Describe each numeric feature by TRAIN binary label and standardized effect."""

    binary_labels = encode_screening_labels(labels)
    rows: list[dict[str, float | int | str]] = []
    for feature_name in features.columns:
        healthy = features.loc[binary_labels.eq(0), feature_name]
        impaired = features.loc[binary_labels.eq(1), feature_name]
        rows.append(
            {
                "feature": feature_name,
                "healthy_n": int(healthy.notna().sum()),
                "healthy_mean": healthy.mean(),
                "healthy_std": healthy.std(ddof=1),
                "healthy_median": healthy.median(),
                "impaired_n": int(impaired.notna().sum()),
                "impaired_mean": impaired.mean(),
                "impaired_std": impaired.std(ddof=1),
                "impaired_median": impaired.median(),
                "cohens_d_impaired_minus_healthy": cohens_d(impaired, healthy),
            }
        )
    return pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)


def save_correlation_plot(correlation: pd.DataFrame, output_path: Path) -> None:
    """Save a readable labelled Pearson correlation heatmap using matplotlib only."""

    figure, axis = plt.subplots(figsize=(18, 16))
    image = axis.imshow(correlation.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_title("TRAIN-only Pearson correlation: linguistic features")
    axis.set_xticks(range(len(correlation.columns)), labels=correlation.columns, rotation=90, fontsize=7)
    axis.set_yticks(range(len(correlation.index)), labels=correlation.index, fontsize=7)
    colorbar = figure.colorbar(image, ax=axis, shrink=0.75)
    colorbar.set_label("Pearson correlation")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    if not FEATURE_TABLE_PATH.is_file():
        print(f"Feature table not found: {FEATURE_TABLE_PATH}")
        return 1
    feature_table = pd.read_csv(FEATURE_TABLE_PATH)
    train_table = training_rows(feature_table)
    feature_names = select_feature_columns(train_table, "ALL")
    features = train_table[feature_names].apply(pd.to_numeric, errors="coerce")
    correlation = features.corr(method="pearson")
    missingness = feature_missingness_and_variance(features)
    pairs = high_correlation_pairs(correlation)
    summary = feature_summary_by_label(features, train_table["screening_label"])

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    missingness.to_csv(OUTPUT_DIRECTORY / "feature_missingness.csv", index=False)
    pairs.to_csv(OUTPUT_DIRECTORY / "highly_correlated_pairs.csv", index=False)
    summary.to_csv(OUTPUT_DIRECTORY / "feature_summary_by_label.csv", index=False)
    save_correlation_plot(correlation, OUTPUT_DIRECTORY / "correlation_matrix.png")

    print("TRAIN-only Linguistic Feature Diagnostics")
    print("======================================")
    print(f"TRAIN participants: {len(train_table)}")
    print("TEST participants used: 0")
    print(f"Feature columns: {len(feature_names)}")
    print(f"Pairs with |Pearson r| >= {CORRELATION_THRESHOLD:.2f}: {len(pairs)}")
    if not pairs.empty:
        print(pairs.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Saved diagnostics to: {OUTPUT_DIRECTORY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
