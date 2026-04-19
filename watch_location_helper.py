from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from mhealth_activity import Recording


WATCH_KEYS = ("ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz", "temperature", "altitude")
AXIS_GROUPS = {
    "acc": ("ax", "ay", "az"),
    "gyr": ("gx", "gy", "gz"),
    "mag": ("mx", "my", "mz"),
}


def parse_trace_id(path: Path) -> int:
    match = re.search(r"(\d{3})\.pkl$", path.name)
    if match is None:
        raise ValueError(f"Could not parse trace id from {path.name}")
    return int(match.group(1))


def downsample(values: Iterable[float], max_len: int = 3000) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= max_len:
        return arr
    idx = np.linspace(0, len(arr) - 1, max_len).astype(int)
    return arr[idx]


def signal_features(values: np.ndarray, samplerate: float) -> dict[str, float]:
    names = (
        "mean",
        "std",
        "median",
        "q05",
        "q25",
        "q75",
        "q95",
        "min",
        "max",
        "range",
        "rms",
        "mad",
        "zcr",
        "dom_freq",
        "dom_power",
        "spec_entropy",
    )
    if len(values) == 0:
        return {name: 0.0 for name in names}

    mean = float(np.mean(values))
    feats = {
        "mean": mean,
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "rms": float(np.sqrt(np.mean(values**2))),
        "mad": float(np.mean(np.abs(values - mean))),
        "zcr": float(np.mean(np.diff(np.signbit(values)) != 0)) if len(values) > 1 else 0.0,
    }
    feats["range"] = feats["max"] - feats["min"]

    if len(values) > 32 and samplerate > 0:
        freqs, power = signal.welch(values - mean, fs=samplerate, nperseg=min(256, len(values)))
        power = np.maximum(power, 1e-12)
        feats["dom_freq"] = float(freqs[np.argmax(power[1:]) + 1] if len(power) > 1 else 0.0)
        feats["dom_power"] = float(np.max(power))
        power_share = power / np.sum(power)
        feats["spec_entropy"] = float(-(power_share * np.log(power_share)).sum())
    else:
        feats["dom_freq"] = 0.0
        feats["dom_power"] = 0.0
        feats["spec_entropy"] = 0.0

    return feats


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 4 or len(b) < 4:
        return 0.0
    corr = np.corrcoef(a, b)[0, 1]
    return 0.0 if not np.isfinite(corr) else float(corr)


def extract_watch_location_features(recording: Recording) -> dict[str, float]:
    features: dict[str, float] = {}
    if "ax" in recording.data:
        features["duration_s"] = float(recording.data["ax"].total_time)

    grouped_axes: dict[str, list[np.ndarray]] = {name: [] for name in AXIS_GROUPS}

    for key in WATCH_KEYS:
        if key not in recording.data:
            continue

        trace = recording.data[key]
        values = downsample(trace.values)
        for name, value in signal_features(values, trace.samplerate).items():
            features[f"{key}_{name}"] = value
        features[f"{key}_samplerate"] = float(trace.samplerate)
        features[f"{key}_gap"] = float(trace.max_update_gap)

        for prefix, axis_names in AXIS_GROUPS.items():
            if key in axis_names:
                max_len = 4000 if prefix != "mag" else 1500
                grouped_axes[prefix].append(downsample(trace.values, max_len=max_len))

    for prefix, axis_names in AXIS_GROUPS.items():
        axes = grouped_axes[prefix]
        if len(axes) != 3:
            continue

        usable_len = min(len(axis) for axis in axes)
        stacked = np.vstack([axis[:usable_len] for axis in axes])
        magnitude = np.sqrt((stacked**2).sum(axis=0))
        samplerate = float(np.mean([recording.data[name].samplerate for name in axis_names if name in recording.data]))

        for name, value in signal_features(downsample(magnitude), samplerate).items():
            features[f"{prefix}mag_{name}"] = value

        features[f"{prefix}_corr_xy"] = safe_corr(stacked[0], stacked[1])
        features[f"{prefix}_corr_xz"] = safe_corr(stacked[0], stacked[2])
        features[f"{prefix}_corr_yz"] = safe_corr(stacked[1], stacked[2])

    return features


def load_dataset(data_dir: Path, labeled: bool) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None, np.ndarray | None]:
    rows: list[dict[str, float]] = []
    ids: list[int] = []
    labels: list[int] = []
    groups: list[int] = []

    for path in sorted(data_dir.glob("*.pkl")):
        recording = Recording(str(path))
        rows.append(extract_watch_location_features(recording))
        ids.append(parse_trace_id(path))
        if labeled:
            labels.append(int(recording.labels["watch_loc"]))
            groups.append(int(recording.labels["path_idx"]))

    frame = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    return (
        frame,
        np.asarray(ids, dtype=int),
        np.asarray(labels, dtype=int) if labeled else None,
        np.asarray(groups, dtype=int) if labeled else None,
    )


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                ExtraTreesClassifier(
                    n_estimators=500,
                    max_depth=12,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )


def align_feature_columns(features: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    aligned = features.reindex(columns=feature_columns, fill_value=np.nan)
    return aligned


def evaluate_model(features: pd.DataFrame, labels: np.ndarray, path_groups: np.ndarray) -> None:
    print("Evaluating watch-location model...")
    model = build_model()

    train_scores: list[float] = []
    stratified_scores: list[float] = []
    stratified_confusion = np.zeros((3, 3), dtype=int)
    stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, valid_idx in stratified_cv.split(features, labels):
        model.fit(features.iloc[train_idx], labels[train_idx])
        train_predictions = model.predict(features.iloc[train_idx])
        predictions = model.predict(features.iloc[valid_idx])
        train_scores.append(accuracy_score(labels[train_idx], train_predictions))
        stratified_scores.append(accuracy_score(labels[valid_idx], predictions))
        stratified_confusion += confusion_matrix(labels[valid_idx], predictions, labels=[0, 1, 2])

    grouped_scores: list[float] = []
    grouped_cv = GroupKFold(n_splits=5)
    for train_idx, valid_idx in grouped_cv.split(features, labels, groups=path_groups):
        model.fit(features.iloc[train_idx], labels[train_idx])
        predictions = model.predict(features.iloc[valid_idx])
        grouped_scores.append(accuracy_score(labels[valid_idx], predictions))

    model.fit(features, labels)
    final_train_predictions = model.predict(features)
    full_train_score = accuracy_score(labels, final_train_predictions)

    class_counts = pd.Series(labels).value_counts().sort_index().to_dict()
    per_class_recall = stratified_confusion.diagonal() / np.maximum(stratified_confusion.sum(axis=1), 1)

    print("Watch-location summary")
    print(f"  Training traces: {len(features)}")
    print(f"  Feature columns: {features.shape[1]}")
    print(f"  Class counts (0=wrist, 1=belt, 2=ankle): {class_counts}")
    print(f"  Full-train accuracy: {full_train_score:.4f}")
    print(f"  Mean fold training accuracy: {np.mean(train_scores):.4f} +/- {np.std(train_scores):.4f}")
    print(f"  Mean 5-fold validation accuracy: {np.mean(stratified_scores):.4f} +/- {np.std(stratified_scores):.4f}")
    print(f"  Mean 5-fold grouped-by-path validation accuracy: {np.mean(grouped_scores):.4f} +/- {np.std(grouped_scores):.4f}")
    print("  Stratified CV confusion matrix (rows=true, cols=pred):")
    print(stratified_confusion)
    print(
        "  Stratified CV per-class recall "
        f"(wrist, belt, ankle): {[round(float(x), 4) for x in per_class_recall]}"
    )


def train_watch_location_model(
    train_dir: Path,
    evaluate: bool = True,
    model_output: Path | None = None,
) -> dict[str, object]:
    print(f"Loading training traces from {train_dir} ...")
    train_x, _, train_y, train_groups = load_dataset(train_dir, labeled=True)

    if evaluate and train_y is not None and train_groups is not None:
        evaluate_model(train_x, train_y, train_groups)

    model = build_model()
    print("Training final model on all training traces...")
    model.fit(train_x, train_y)

    artifact = {
        "model": model,
        "feature_columns": list(train_x.columns),
        "label_map": {0: "wrist", 1: "belt", 2: "ankle"},
        "task": "watch_location",
    }

    if model_output is not None:
        model_output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, model_output)
        print(f"Saved watch-location model to {model_output}")

    return artifact


def load_watch_location_model(model_path: Path) -> dict[str, object]:
    artifact = joblib.load(model_path)
    required_keys = {"model", "feature_columns", "task"}
    missing_keys = required_keys.difference(artifact.keys())
    if missing_keys:
        raise ValueError(f"Saved model at {model_path} is missing keys: {sorted(missing_keys)}")
    if artifact["task"] != "watch_location":
        raise ValueError(f"Saved model at {model_path} is not a watch-location model")
    print(f"Loaded watch-location model from {model_path}")
    return artifact


def predict_watch_locations_from_artifact(
    artifact: dict[str, object],
    test_dir: Path,
    output_csv: Path,
) -> pd.DataFrame:
    print(f"Loading test traces from {test_dir} ...")
    test_x, test_ids, _, _ = load_dataset(test_dir, labeled=False)
    feature_columns = artifact["feature_columns"]
    test_x = align_feature_columns(test_x, feature_columns)

    print("Predicting watch locations for test traces...")
    predicted_watch_locations = artifact["model"].predict(test_x).astype(int)

    predictions = pd.DataFrame(
        {
            "Id": test_ids,
            "watch_loc": predicted_watch_locations,
        }
    ).sort_values("Id")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_csv, index=False)
    print(f"Saved watch-location predictions to {output_csv}")
    return predictions


def predict_watch_locations(
    train_dir: Path,
    test_dir: Path,
    output_csv: Path,
    evaluate: bool,
    model_output: Path | None = None,
) -> None:
    artifact = train_watch_location_model(
        train_dir=train_dir,
        evaluate=evaluate,
        model_output=model_output,
    )
    predict_watch_locations_from_artifact(
        artifact=artifact,
        test_dir=test_dir,
        output_csv=output_csv,
    )


def main() -> None:
    default_train_dir = Path(r"c:\Users\giaco\Desktop\ETH\Mobile_Health\DATA\data\train")
    default_test_dir = Path(r"c:\Users\giaco\Desktop\ETH\Mobile_Health\DATA\data\test")
    default_output = Path("watch_location_predictions.csv")
    default_model_output = Path("groupXX_model_watchloc.joblib")

    parser = argparse.ArgumentParser(
        description="Standalone helper to train a smartwatch location model and predict watch_loc for test traces."
    )
    parser.add_argument("--train-dir", type=Path, default=default_train_dir)
    parser.add_argument("--test-dir", type=Path, default=default_test_dir)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--skip-eval", action="store_true", help="Skip cross-validation and train directly on all training data.")
    parser.add_argument("--save-model", type=Path, default=None, help="Optional path to save the trained watch-location model.")
    parser.add_argument("--load-model", type=Path, default=None, help="Optional path to a previously saved watch-location model.")
    parser.add_argument(
        "--train-and-save-default",
        action="store_true",
        help=f"Convenience flag to save the trained model to {default_model_output}.",
    )
    args = parser.parse_args()

    model_output = args.save_model
    if args.train_and_save_default:
        model_output = default_model_output

    if args.load_model is not None:
        artifact = load_watch_location_model(args.load_model)
        predict_watch_locations_from_artifact(
            artifact=artifact,
            test_dir=args.test_dir,
            output_csv=args.output,
        )
        return

    predict_watch_locations(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_csv=args.output,
        evaluate=not args.skip_eval,
        model_output=model_output,
    )


if __name__ == "__main__":
    main()
