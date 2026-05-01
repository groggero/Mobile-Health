from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score

from mhealth_activity.recording import Recording


ACTIVITY_ORDER = ("standing", "walking", "running", "cycling")
WATCH_MOTION_GROUPS = {
    "watch_acc": ("ax", "ay", "az"),
    "watch_gyr": ("gx", "gy", "gz"),
    "watch_mag": ("mx", "my", "mz"),
}

DEFAULT_WINDOW_S = 15.0
DEFAULT_HOP_S = 5.0
DEFAULT_MIN_ACTIVITY_S = 60.0
DEFAULT_TRAIN_DIR = Path(r"c:\Users\giaco\Desktop\ETH\Mobile_Health\DATA\data\train")
DEFAULT_TEST_DIR = Path(r"c:\Users\giaco\Desktop\ETH\Mobile_Health\DATA\data\test")
DEFAULT_MODEL_PATH = Path("group10_activity_recognizer.joblib")


def normalize_activities(raw_value) -> list[str]:
    if isinstance(raw_value, Mapping):
        return [name for name in ACTIVITY_ORDER if bool(raw_value.get(name, False))]

    if isinstance(raw_value, (list, tuple, set, np.ndarray)):
        parsed = []
        for item in raw_value:
            if isinstance(item, str):
                label = item.strip().lower()
                if label in ACTIVITY_ORDER:
                    parsed.append(label)
            elif isinstance(item, (int, np.integer)) and 0 <= int(item) < len(ACTIVITY_ORDER):
                parsed.append(ACTIVITY_ORDER[int(item)])
        return [name for name in ACTIVITY_ORDER if name in set(parsed)]

    return []


def summarize_signal(values: Sequence[float], samplerate: float) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        raise ValueError("Signal summary requires at least one sample.")

    centered = values - np.mean(values)
    freqs, power = welch(centered, fs=samplerate, nperseg=min(256, len(values)))
    dom_idx = np.argmax(power[1:]) + 1 if len(power) > 1 else 0

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "rms": float(np.sqrt(np.mean(values ** 2))),
        "iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
        "dom_freq": float(freqs[dom_idx]) if len(freqs) else 0.0,
        "dom_power": float(power[dom_idx]) if len(power) else 0.0,
        "spec_energy": float(power.sum()) if len(power) else 0.0,
    }


def extract_window_feature_rows(
    recording: Recording,
    motion_groups: Mapping[str, Sequence[str]] = WATCH_MOTION_GROUPS,
    window_s: float = DEFAULT_WINDOW_S,
    hop_s: float = DEFAULT_HOP_S,
) -> pd.DataFrame:
    base_keys = ("ax", "ay", "az")
    if not all(key in recording.data for key in base_keys):
        return pd.DataFrame()

    base_fs = float(np.mean([recording.data[key].samplerate for key in base_keys]))
    base_len = min(len(recording.data[key].values) for key in base_keys)
    window_n = max(int(window_s * base_fs), 1)
    hop_n = max(int(hop_s * base_fs), 1)

    if base_len < window_n:
        return pd.DataFrame()

    rows: list[dict[str, float]] = []
    for start_idx in range(0, base_len - window_n + 1, hop_n):
        end_idx = start_idx + window_n
        row = {
            "start_s": start_idx / base_fs,
            "end_s": end_idx / base_fs,
            "mid_s": (start_idx + end_idx) / (2 * base_fs),
            "window_s": float(window_s),
            "hop_s": float(hop_s),
        }

        for prefix, keys in motion_groups.items():
            if not all(key in recording.data for key in keys):
                continue

            usable_len = min(len(recording.data[key].values) for key in keys)
            if end_idx > usable_len:
                continue

            stacked = np.vstack(
                [recording.data[key].values[start_idx:end_idx].astype(float) for key in keys]
            )
            magnitude = np.sqrt((stacked ** 2).sum(axis=0))
            samplerate = float(np.mean([recording.data[key].samplerate for key in keys]))

            for feature_name, value in summarize_signal(magnitude, samplerate).items():
                row[f"{prefix}_{feature_name}"] = value

        rows.append(row)

    return pd.DataFrame(rows)


def _bridge_short_gaps(mask: Sequence[bool], max_gap_windows: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool).copy()
    if max_gap_windows <= 0:
        return mask

    n_samples = len(mask)
    idx = 0
    while idx < n_samples:
        if mask[idx]:
            idx += 1
            continue

        gap_end = idx
        while gap_end < n_samples and not mask[gap_end]:
            gap_end += 1

        if idx > 0 and gap_end < n_samples and (gap_end - idx) <= max_gap_windows:
            mask[idx:gap_end] = True

        idx = gap_end

    return mask


def _smooth_labels(labels: Sequence[str]) -> np.ndarray:
    labels = np.asarray(labels, dtype=object)
    if len(labels) < 3:
        return labels.copy()

    smoothed = labels.copy()
    for idx in range(1, len(labels) - 1):
        triad = labels[idx - 1 : idx + 2]
        values, counts = np.unique(triad, return_counts=True)
        smoothed[idx] = values[np.argmax(counts)]
    return smoothed


def _longest_positive_run(mask: Sequence[bool], hop_s: float, window_s: float) -> float:
    best = 0.0
    current = 0
    for flag in mask:
        if flag:
            current += 1
            best = max(best, window_s + max(current - 1, 0) * hop_s)
        else:
            current = 0
    return best


def _activity_gap_windows(activity: str, hop_s: float) -> int:
    if activity in {"walking", "running"}:
        return int(np.floor(8.0 / hop_s))
    return 1


@dataclass
class ActivityRecognizer:
    classifier: ExtraTreesClassifier
    feature_columns: list[str]
    feature_medians: pd.Series
    motion_groups: Mapping[str, Sequence[str]] = field(
        default_factory=lambda: WATCH_MOTION_GROUPS
    )
    window_s: float = DEFAULT_WINDOW_S
    hop_s: float = DEFAULT_HOP_S
    min_activity_s: float = DEFAULT_MIN_ACTIVITY_S

    def label_windows(self, recording: Recording) -> pd.DataFrame:
        window_df = extract_window_feature_rows(
            recording,
            motion_groups=self.motion_groups,
            window_s=self.window_s,
            hop_s=self.hop_s,
        )
        if window_df.empty:
            return window_df

        x = window_df.reindex(columns=self.feature_columns).fillna(self.feature_medians)
        proba = self.classifier.predict_proba(x)
        classes = np.asarray(self.classifier.classes_)
        raw_labels = classes[np.argmax(proba, axis=1)]
        pred_labels = _smooth_labels(raw_labels)

        labeled = window_df.copy()
        labeled["raw_pred_label"] = raw_labels
        labeled["pred_label"] = pred_labels

        for activity_name in ACTIVITY_ORDER:
            labeled[f"prob_{activity_name}"] = 0.0

        for class_idx, class_name in enumerate(classes):
            labeled[f"prob_{class_name}"] = proba[:, class_idx]

        return labeled

    def summarize_labeled_windows(self, labeled_window_df: pd.DataFrame) -> pd.DataFrame:
        if labeled_window_df.empty:
            return pd.DataFrame(
                {
                    "activity": list(ACTIVITY_ORDER),
                    "longest_run_s": [0.0] * len(ACTIVITY_ORDER),
                    "meets_60s_rule": [False] * len(ACTIVITY_ORDER),
                }
            )

        hop_s = float(labeled_window_df["hop_s"].iloc[0])
        window_s = float(labeled_window_df["window_s"].iloc[0])

        rows = []
        for activity_name in ACTIVITY_ORDER:
            raw_mask = labeled_window_df["pred_label"].to_numpy() == activity_name
            bridged_mask = _bridge_short_gaps(
                raw_mask,
                max_gap_windows=_activity_gap_windows(activity_name, hop_s),
            )
            longest_run_s = _longest_positive_run(bridged_mask, hop_s=hop_s, window_s=window_s)
            rows.append(
                {
                    "activity": activity_name,
                    "longest_run_s": float(longest_run_s),
                    "meets_60s_rule": bool(longest_run_s >= self.min_activity_s),
                }
            )

        return pd.DataFrame(rows)

    def predict_activities(self, recording: Recording) -> dict[str, bool]:
        summary_df = self.summarize_labeled_windows(self.label_windows(recording))
        return {
            row.activity: bool(row.meets_60s_rule)
            for row in summary_df.itertuples(index=False)
        }


def _trace_id_from_path(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def build_activity_recognizer(
    train_dir: Path | str,
    motion_groups: Mapping[str, Sequence[str]] = WATCH_MOTION_GROUPS,
    window_s: float = DEFAULT_WINDOW_S,
    hop_s: float = DEFAULT_HOP_S,
    min_activity_s: float = DEFAULT_MIN_ACTIVITY_S,
    standing_fraction: float = 0.15,
    random_state: int = 42,
    n_estimators: int = 120,
    min_samples_leaf: int = 2,
) -> ActivityRecognizer:
    train_dir = Path(train_dir)
    metadata_rows = []
    window_cache: dict[str, pd.DataFrame] = {}
    all_feature_names: set[str] = set()

    for path in sorted(train_dir.glob("*.pkl")):
        recording = Recording(str(path))
        activities = normalize_activities((recording.labels or {}).get("activities", []))
        metadata_rows.append(
            {
                "file": path.name,
                "trace_id": _trace_id_from_path(path),
                "n_activities": len(activities),
                **{activity: activity in activities for activity in ACTIVITY_ORDER},
            }
        )

        window_df = extract_window_feature_rows(
            recording,
            motion_groups=motion_groups,
            window_s=window_s,
            hop_s=hop_s,
        )
        window_cache[path.name] = window_df
        all_feature_names.update(
            column
            for column in window_df.columns
            if column not in {"start_s", "end_s", "mid_s", "window_s", "hop_s"}
        )

    metadata_df = pd.DataFrame(metadata_rows).sort_values("trace_id").reset_index(drop=True)
    single_label_df = metadata_df.loc[metadata_df["n_activities"] == 1].copy()
    single_label_df["activity_label"] = single_label_df[list(ACTIVITY_ORDER)].idxmax(axis=1)

    training_parts = []
    for row in single_label_df.itertuples(index=False):
        window_df = window_cache[row.file]
        if window_df.empty:
            continue
        part = window_df.copy()
        part["file"] = row.file
        part["activity_label"] = row.activity_label
        training_parts.append(part)

    standing_subset = metadata_df.loc[metadata_df["standing"]].copy()
    for row in standing_subset.itertuples(index=False):
        window_df = window_cache[row.file]
        if window_df.empty:
            continue

        motion_score = window_df["watch_acc_std"] + 0.3 * window_df["watch_gyr_std"]
        keep_n = max(1, int(np.ceil(standing_fraction * len(window_df))))
        part = window_df.loc[motion_score.nsmallest(keep_n).index].copy()
        part["file"] = row.file
        part["activity_label"] = "standing"
        training_parts.append(part)

    if not training_parts:
        raise RuntimeError("No training windows were extracted for activity recognition.")

    training_df = pd.concat(training_parts, ignore_index=True)
    feature_columns = sorted(all_feature_names)
    feature_medians = training_df.reindex(columns=feature_columns).median(numeric_only=True)
    x_train = training_df.reindex(columns=feature_columns).fillna(feature_medians)
    y_train = training_df["activity_label"]

    classifier = ExtraTreesClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=1,
    )
    classifier.fit(x_train, y_train)

    return ActivityRecognizer(
        classifier=classifier,
        feature_columns=feature_columns,
        feature_medians=feature_medians,
        motion_groups=motion_groups,
        window_s=window_s,
        hop_s=hop_s,
        min_activity_s=min_activity_s,
    )


def save_activity_recognizer(
    recognizer: ActivityRecognizer,
    model_path: Path | str = DEFAULT_MODEL_PATH,
) -> Path:
    model_path = Path(model_path)
    joblib.dump(recognizer, model_path)
    return model_path


def load_activity_recognizer(
    model_path: Path | str = DEFAULT_MODEL_PATH,
) -> ActivityRecognizer:
    return joblib.load(model_path)


def train_and_save_activity_recognizer(
    train_dir: Path | str = DEFAULT_TRAIN_DIR,
    model_path: Path | str = DEFAULT_MODEL_PATH,
) -> ActivityRecognizer:
    recognizer = build_activity_recognizer(train_dir)
    save_activity_recognizer(recognizer, model_path)
    return recognizer


def evaluate_activity_recognizer(
    recognizer: ActivityRecognizer,
    data_dir: Path | str,
) -> pd.DataFrame:
    data_dir = Path(data_dir)
    rows = []
    for path in sorted(data_dir.glob("*.pkl")):
        recording = Recording(str(path))
        truth_activities = normalize_activities((recording.labels or {}).get("activities", []))
        truth = {activity: activity in truth_activities for activity in ACTIVITY_ORDER}
        pred = recognizer.predict_activities(recording)

        row = {"file": path.name}
        for activity in ACTIVITY_ORDER:
            row[f"true_{activity}"] = bool(truth[activity])
            row[f"pred_{activity}"] = bool(pred[activity])
        rows.append(row)

    return pd.DataFrame(rows)


def activity_f1_summary(prediction_df: pd.DataFrame) -> pd.Series:
    scores = {
        activity: f1_score(
            prediction_df[f"true_{activity}"],
            prediction_df[f"pred_{activity}"],
        )
        for activity in ACTIVITY_ORDER
    }
    scores["macro"] = float(np.mean([scores[activity] for activity in ACTIVITY_ORDER]))
    return pd.Series(scores, dtype=float)


def generate_activity_predictions(
    recognizer: ActivityRecognizer,
    test_dir: Path | str = DEFAULT_TEST_DIR,
) -> pd.DataFrame:
    test_dir = Path(test_dir)
    rows = []

    for path in sorted(test_dir.glob("*.pkl")):
        recording = Recording(str(path))
        predicted_activities = recognizer.predict_activities(recording)
        rows.append(
            {
                "Id": _trace_id_from_path(path),
                **{
                    activity: bool(predicted_activities[activity])
                    for activity in ACTIVITY_ORDER
                },
            }
        )

    return pd.DataFrame(rows).sort_values("Id").reset_index(drop=True)


def run_activity_analysis(
    train_dir: Path | str = DEFAULT_TRAIN_DIR,
    test_dir: Path | str = DEFAULT_TEST_DIR,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    evaluate_train: bool = False,
    prediction_path: Path | str | None = None,
) -> dict[str, object]:
    recognizer = train_and_save_activity_recognizer(train_dir, model_path)
    result: dict[str, object] = {
        "recognizer": recognizer,
        "model_path": Path(model_path).resolve(),
    }

    if evaluate_train:
        evaluation_df = evaluate_activity_recognizer(recognizer, train_dir)
        result["evaluation"] = evaluation_df
        result["f1"] = activity_f1_summary(evaluation_df)

    if prediction_path is not None:
        prediction_df = generate_activity_predictions(recognizer, test_dir)
        prediction_df.to_csv(prediction_path, index=False)
        result["predictions"] = prediction_df
        result["prediction_path"] = Path(prediction_path).resolve()

    return result


if __name__ == "__main__":
    analysis_result = run_activity_analysis(evaluate_train=True)
    print(f"Saved activity recognizer to {analysis_result['model_path']}")
    print(analysis_result["f1"].rename("f1").to_frame())
