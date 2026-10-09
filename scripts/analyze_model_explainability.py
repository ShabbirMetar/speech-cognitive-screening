"""TRAIN-only SHAP analysis for the frozen 51-feature screening representation.

SHAP values are calculated on the standardized linear-model log-odds scale.
They explain model behaviour; they do not establish causal or clinical biomarkers.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import GridSearchCV, StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from ml.asr.asr_linguistic_features import asr_feature_columns
from ml.data.development_guard import assert_train_only
from ml.models.baseline_logistic import build_logistic_pipeline, encode_screening_labels
from ml.models.explainability import (
    FROZEN_REPRESENTATION,
    RESEARCH_OPERATING_THRESHOLD,
    coefficient_direction,
    feature_groups,
    format_feature_contribution,
    human_feature_label,
    screening_classification,
    screening_score,
    select_example_indices,
)
from ml.models.modality_comparison import PURE_ACOUSTIC_SUFFIXES, select_pure_acoustic_feature_columns
from scripts.analyze_screening_threshold import _align_to_asr, frozen_feature_columns
from scripts.compare_deployment_representations import asr_rate_feature_columns, validate_asr_rate_provenance


ASR_LINGUISTIC_TRAIN_PATH = PROJECT_ROOT / "artifacts" / "features" / "asr_linguistic_features_train.csv"
DEPLOYMENT_TRAIN_PATH = PROJECT_ROOT / "artifacts" / "features" / "deployment_features_train.csv"
OUTPUT_DIRECTORY = PROJECT_ROOT / "artifacts" / "results" / "explainability"
EXPECTED_DEVELOPMENT_PARTICIPANTS = 320
C_VALUES = (0.1, 1.0, 10.0)
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def validate_frozen_schema(
    asr_table: pd.DataFrame, deployment_table: pd.DataFrame
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Validate and return the exact, already-frozen 51-feature TRAIN schema."""

    asr = assert_train_only(asr_table)
    deployment = _align_to_asr(asr, deployment_table)
    if len(asr) != EXPECTED_DEVELOPMENT_PARTICIPANTS:
        raise AssertionError("Explainability requires exactly 320 TRAIN participants.")

    features = frozen_feature_columns(asr, deployment)
    if len(features) != 51 or len(set(features)) != 51:
        raise AssertionError("The frozen explainability schema must contain 51 unique predictors.")
    forbidden = {"participant_id", "diagnosis", "screening_label", "Split", "age", "gender", "MMSE"}
    if forbidden.intersection(features):
        raise AssertionError("Frozen predictors include bookkeeping, demographic, or label columns.")
    if any("pause_annotation" in feature for feature in features):
        raise AssertionError("Manual transcript pause annotations must not be explainability predictors.")

    asr_features = asr_feature_columns(asr)
    rate_features = asr_rate_feature_columns(deployment)
    ctd_acoustic_features = select_pure_acoustic_feature_columns(deployment, "CTD")
    if features != [*asr_features, *rate_features, *ctd_acoustic_features]:
        raise AssertionError("Feature ordering does not match the frozen construction logic.")
    if (len(asr_features), len(rate_features), len(ctd_acoustic_features)) != (32, 6, 13):
        raise AssertionError("Frozen group counts must be 32 ASR linguistic, 6 rate, and 13 CTD acoustic.")
    if any(feature.removeprefix("ctd_") not in PURE_ACOUSTIC_SUFFIXES for feature in ctd_acoustic_features):
        raise AssertionError("CTD acoustic features must be waveform-only pure-acoustic fields.")
    validate_asr_rate_provenance(asr, deployment)

    groups = feature_groups(features)
    observed_counts = pd.Series(groups).value_counts().to_dict()
    expected_counts = {"ASR_LINGUISTIC": 32, "ASR_RATE": 6, "CTD_ACOUSTIC": 13}
    if observed_counts != expected_counts:
        raise AssertionError(f"Unexpected frozen feature groups: {observed_counts}")
    return deployment, features, groups


def resolve_final_c(features: pd.DataFrame, target: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Resolve the frozen Logistic Regression C with the requested TRAIN-only CV."""

    search = GridSearchCV(
        estimator=build_logistic_pipeline(),
        param_grid={"classifier__C": list(C_VALUES)},
        scoring="f1",
        cv=CV,
        n_jobs=2,
        refit=True,
        return_train_score=False,
    )
    search.fit(features, target)
    resolution = pd.DataFrame(
        {
            "C": [float(value) for value in search.cv_results_["param_classifier__C"].data],
            "mean_cv_f1": search.cv_results_["mean_test_score"],
            "std_cv_f1": search.cv_results_["std_test_score"],
            "rank": search.cv_results_["rank_test_score"],
        }
    ).sort_values("C").reset_index(drop=True)
    return float(search.best_params_["classifier__C"]), resolution


def linear_shap_values(classifier: object, transformed_features: np.ndarray) -> tuple[np.ndarray, float]:
    """Calculate binary linear-model SHAP values on the model's log-odds scale."""

    # Keep all 320 transformed TRAIN rows as the background reference rather
    # than accepting SHAP's smaller default background sample.
    masker = shap.maskers.Independent(transformed_features, max_samples=len(transformed_features))
    explainer = shap.LinearExplainer(classifier, masker)
    values = np.asarray(explainer.shap_values(transformed_features), dtype=float)
    if values.ndim == 3:
        # Current SHAP may represent binary outputs as [row, feature, class].
        values = values[:, :, 1]
    if values.ndim != 2:
        raise AssertionError(f"Expected a two-dimensional SHAP matrix; received {values.shape}.")
    expected = np.asarray(explainer.expected_value, dtype=float).reshape(-1)
    if expected.size == 1:
        base_value = float(expected[0])
    elif expected.size == 2:
        base_value = float(expected[1])
    else:
        raise AssertionError("Expected one binary Logistic Regression SHAP base value.")
    return values, base_value


def _plot_global_bar(global_importance: pd.DataFrame, output_path: Path) -> None:
    top = global_importance.head(20).iloc[::-1]
    figure, axis = plt.subplots(figsize=(10, 8))
    axis.barh(top["feature_label"], top["mean_abs_shap"], color="#4C78A8")
    axis.set_title("Top global SHAP importance (TRAIN, linear model)")
    axis.set_xlabel("Mean absolute SHAP value (log-odds scale)")
    axis.set_ylabel("Feature")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _plot_beeswarm(shap_values: np.ndarray, transformed_features: np.ndarray, labels: list[str], output_path: Path) -> None:
    shap.summary_plot(
        shap_values,
        features=transformed_features,
        feature_names=labels,
        max_display=20,
        show=False,
        plot_size=(10, 8),
    )
    plt.title("Global SHAP distribution (TRAIN, standardized inputs)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _contributor_payload(
    feature_names: list[str],
    shap_row: np.ndarray,
    observed_row: pd.Series,
    reference_medians: pd.Series,
    positive: bool,
) -> str:
    """Encode the top signed participant contributions for a portable CSV field."""

    candidates = (
        [index for index, value in enumerate(shap_row) if value > 0]
        if positive
        else [index for index, value in enumerate(shap_row) if value < 0]
    )
    ordered = sorted(candidates, key=lambda index: abs(float(shap_row[index])), reverse=True)[:3]
    payload: list[dict[str, object]] = []
    for index in ordered:
        feature = feature_names[index]
        observed = pd.to_numeric(pd.Series([observed_row[feature]]), errors="coerce").iloc[0]
        reference = float(reference_medians[feature])
        payload.append(
            {
                "feature": feature,
                "feature_label": human_feature_label(feature),
                "shap_log_odds": float(shap_row[index]),
                "observed_value": None if pd.isna(observed) else float(observed),
                "train_reference_median": reference,
                "explanation": format_feature_contribution(feature, float(shap_row[index]), observed, reference),
            }
        )
    return json.dumps(payload, ensure_ascii=False)


def build_example_explanations(
    train: pd.DataFrame,
    feature_names: list[str],
    raw_features: pd.DataFrame,
    reference_medians: pd.Series,
    scores: np.ndarray,
    shap_values: np.ndarray,
) -> pd.DataFrame:
    """Build deterministic, label-blind example rows selected only by model score."""

    selected = select_example_indices(train["participant_id"].tolist(), scores)
    rows: list[dict[str, object]] = []
    for selection_rule, index in selected.items():
        score = float(scores[index])
        rows.append(
            {
                "selection_rule": selection_rule,
                "participant_id": train.iloc[index]["participant_id"],
                "raw_model_score": score,
                "cognitive_speech_screening_score": screening_score(score),
                "classification_at_0_53": screening_classification(score),
                "top_positive_contributors": _contributor_payload(
                    feature_names, shap_values[index], raw_features.iloc[index], reference_medians, positive=True
                ),
                "top_negative_contributors": _contributor_payload(
                    feature_names, shap_values[index], raw_features.iloc[index], reference_medians, positive=False
                ),
            }
        )
    examples = pd.DataFrame(rows)
    if len(examples) != 3 or examples["participant_id"].duplicated().any():
        raise AssertionError("Example explanations must have three unique, deterministic TRAIN participants.")
    return examples


def main() -> int:
    for path in (ASR_LINGUISTIC_TRAIN_PATH, DEPLOYMENT_TRAIN_PATH):
        if not path.is_file():
            print(f"Required TRAIN feature table not found: {path}")
            return 1

    deployment, feature_names, groups = validate_frozen_schema(
        pd.read_csv(ASR_LINGUISTIC_TRAIN_PATH), pd.read_csv(DEPLOYMENT_TRAIN_PATH)
    )
    train = assert_train_only(deployment)
    raw_features = train[feature_names].apply(pd.to_numeric, errors="coerce")
    if list(raw_features.columns) != feature_names:
        raise AssertionError("Raw feature ordering must match the frozen model feature ordering.")
    target = encode_screening_labels(train["screening_label"]).to_numpy()

    selected_c, c_resolution = resolve_final_c(raw_features, target)
    pipeline = build_logistic_pipeline().set_params(classifier__C=selected_c)
    pipeline.fit(raw_features, target)
    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    classifier = pipeline.named_steps["classifier"]
    imputed_features = imputer.transform(raw_features)
    transformed_features = scaler.transform(imputed_features)
    shap_values, base_value = linear_shap_values(classifier, transformed_features)

    if shap_values.shape != (len(train), len(feature_names)):
        raise AssertionError("SHAP shape must be 320 rows by 51 frozen features.")
    if not np.isfinite(shap_values).all():
        raise AssertionError("SHAP values must not contain NaN or infinite values.")
    if len(classifier.coef_.ravel()) != len(feature_names):
        raise AssertionError("Coefficient count must match the frozen feature count.")
    if classifier.n_features_in_ != len(feature_names):
        raise AssertionError("Classifier input count must match the frozen feature count.")
    logits = classifier.decision_function(transformed_features)
    reconstructed_logits = base_value + shap_values.sum(axis=1)
    max_reconstruction_error = float(np.max(np.abs(logits - reconstructed_logits)))
    if not np.allclose(logits, reconstructed_logits, rtol=1e-6, atol=1e-6):
        raise AssertionError("SHAP values do not reconstruct the Logistic Regression log-odds output.")
    raw_scores = pipeline.predict_proba(raw_features)[:, 1]
    if not np.allclose(raw_scores, 1.0 / (1.0 + np.exp(-logits)), rtol=1e-8, atol=1e-8):
        raise AssertionError("Raw classifier scores must match the fitted linear-model logits.")
    display_scores = np.asarray([screening_score(score) for score in raw_scores])
    if not np.all((display_scores >= 0.0) & (display_scores <= 100.0)):
        raise AssertionError("Cognitive Speech Screening Scores must remain in the 0–100 range.")

    coefficients = classifier.coef_.ravel()
    coefficient_table = pd.DataFrame(
        {
            "feature": feature_names,
            "feature_label": [human_feature_label(feature) for feature in feature_names],
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
            "direction": [coefficient_direction(value) for value in coefficients],
        }
    ).sort_values("absolute_coefficient", ascending=False).reset_index(drop=True)
    global_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "feature_label": [human_feature_label(feature) for feature in feature_names],
            "feature_group": [groups[feature] for feature in feature_names],
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "model_coefficient": coefficients,
            "coefficient_direction": [coefficient_direction(value) for value in coefficients],
        }
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    global_importance.insert(0, "rank", np.arange(1, len(global_importance) + 1))
    if len(global_importance) != 51 or set(global_importance["feature"]) != set(feature_names):
        raise AssertionError("Global importance must contain all and only frozen predictors.")

    group_importance = (
        global_importance.groupby("feature_group", sort=False)["mean_abs_shap"]
        .sum()
        .reindex(["ASR_LINGUISTIC", "ASR_RATE", "CTD_ACOUSTIC"])
        .rename("sum_mean_abs_shap")
        .reset_index()
    )
    group_importance["percentage_of_total_abs_shap"] = (
        100.0 * group_importance["sum_mean_abs_shap"] / group_importance["sum_mean_abs_shap"].sum()
    )
    if group_importance["sum_mean_abs_shap"].isna().any():
        raise AssertionError("Feature-group mapping must cover every frozen predictor.")

    reference_medians = pd.Series(imputer.statistics_, index=feature_names)
    examples = build_example_explanations(
        train, feature_names, raw_features, reference_medians, raw_scores, shap_values
    )
    expected_classification = examples["raw_model_score"].map(screening_classification)
    if not examples["classification_at_0_53"].equals(expected_classification):
        raise AssertionError("Example classifications must use the raw-score 0.53 research threshold.")
    valid_features = set(feature_names)
    for column in ("top_positive_contributors", "top_negative_contributors"):
        for payload in examples[column]:
            if not {item["feature"] for item in json.loads(payload)}.issubset(valid_features):
                raise AssertionError("Participant-level explanations contain a non-frozen feature.")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    c_resolution.to_csv(OUTPUT_DIRECTORY / "hyperparameter_resolution.csv", index=False)
    global_importance.to_csv(OUTPUT_DIRECTORY / "global_shap_importance.csv", index=False)
    coefficient_table.to_csv(OUTPUT_DIRECTORY / "standardized_coefficients.csv", index=False)
    group_importance.to_csv(OUTPUT_DIRECTORY / "feature_group_importance.csv", index=False)
    examples.to_csv(OUTPUT_DIRECTORY / "example_explanations.csv", index=False)
    _plot_global_bar(global_importance, OUTPUT_DIRECTORY / "global_shap_bar.png")
    _plot_beeswarm(
        shap_values,
        transformed_features,
        [human_feature_label(feature) for feature in feature_names],
        OUTPUT_DIRECTORY / "global_shap_beeswarm.png",
    )
    metadata = {
        "representation": FROZEN_REPRESENTATION,
        "development_participants": int(len(train)),
        "official_test_participants_used": 0,
        "feature_count": int(len(feature_names)),
        "feature_order": feature_names,
        "feature_groups": groups,
        "classifier": "LogisticRegression(max_iter=2000, random_state=42)",
        "selected_C": selected_c,
        "C_candidates": list(C_VALUES),
        "selection_cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring=F1",
        "preprocessing": ["SimpleImputer(strategy=median)", "StandardScaler"],
        "score_source": "raw Logistic Regression classifier score",
        "score_display": "Cognitive Speech Screening Score = raw model score * 100; display transformation only",
        "research_operating_threshold": RESEARCH_OPERATING_THRESHOLD,
        "shap_method": "shap.LinearExplainer on imputed and standardized model inputs",
        "shap_value_scale": "log-odds (linear Logistic Regression decision-function scale)",
        "max_log_odds_reconstruction_error": max_reconstruction_error,
        "missing_feature_counts": {feature: int(raw_features[feature].isna().sum()) for feature in feature_names if raw_features[feature].isna().any()},
        "limitations": "TRAIN-only explanatory analysis; SHAP and coefficients describe model behaviour, not causal or clinical biomarkers.",
    }
    (OUTPUT_DIRECTORY / "explainability_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("Model Explainability Analysis")
    print("=============================")
    print(f"Development participants: {len(train)}")
    print("Official TEST participants used: 0")
    print(f"Frozen representation: {FROZEN_REPRESENTATION}")
    print(f"Features: {len(feature_names)}")
    print("Classifier: Logistic Regression")
    print(f"Selected C: {selected_c:g}")
    print("CV F1 by C:")
    for row in c_resolution.itertuples(index=False):
        print(f"  C={row.C:g}: {row.mean_cv_f1:.3f} ± {row.std_cv_f1:.3f}")
    print("\nFrozen feature groups:")
    for group in ("ASR_LINGUISTIC", "ASR_RATE", "CTD_ACOUSTIC"):
        names = [feature for feature in feature_names if groups[feature] == group]
        print(f"  {group} ({len(names)}): {', '.join(names)}")
    print("\nTop global features (mean absolute SHAP; log-odds scale):")
    for row in global_importance.head(10).itertuples(index=False):
        print(f"  {row.feature}: {row.mean_abs_shap:.4f} ({row.coefficient_direction})")
    print("\nFeature-group SHAP contribution:")
    for row in group_importance.itertuples(index=False):
        print(f"  {row.feature_group}: {row.sum_mean_abs_shap:.4f} ({row.percentage_of_total_abs_shap:.1f}%)")
    print(f"\nResearch operating threshold: {RESEARCH_OPERATING_THRESHOLD:.2f}")
    print("Example explanation scores (TRAIN demonstrations only):")
    for row in examples.itertuples(index=False):
        print(f"  {row.selection_rule}: {row.raw_model_score:.4f} ({row.classification_at_0_53})")
    print(f"\nSHAP log-odds reconstruction max error: {max_reconstruction_error:.3e}")
    print(f"Saved outputs: {OUTPUT_DIRECTORY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
