from pathlib import Path
from collections import Counter

from mhealth_activity import Recording, Activity


def main() -> None:
    train_dir = Path("data/train")

    if not train_dir.exists():
        raise FileNotFoundError(f"Folder not found: {train_dir.resolve()}")

    activity_counter = Counter()

    pkl_files = sorted(train_dir.glob("*.pkl"))
    print(f"Found {len(pkl_files)} training traces.\n")

    for path in pkl_files:
        recording = Recording(str(path))
        labels = recording.labels

        if labels is None or "activities" not in labels:
            print(f"{path.name}: no activity labels")
            continue

        labeled_ids = labels["activities"]

        present_names = []
        for activity in Activity:
            if activity.value in labeled_ids:
                present_names.append(activity.name.lower())
                activity_counter[activity.name.lower()] += 1

        print(f"{path.name}: {present_names}")

    print("\nSummary:")
    for activity in Activity:
        name = activity.name.lower()
        print(f"{name}: {activity_counter[name]}")


if __name__ == "__main__":
    main()