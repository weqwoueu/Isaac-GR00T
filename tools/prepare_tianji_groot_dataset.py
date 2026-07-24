#!/usr/bin/env python3
"""Add GR00T N1.7 modality metadata to the Tianji Pico LeRobot v2.1 dataset.

The dataset already uses the LeRobot v2 layout, so preparation only adds
``meta/modality.json``. Parquet and video files are validated but never rewritten.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


STATE_KEY = "observation.state"
ACTION_KEY = "action"
VIDEO_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)

EXPECTED_NAMES = [
    "left_ee_x",
    "left_ee_y",
    "left_ee_z",
    "left_ee_qx",
    "left_ee_qy",
    "left_ee_qz",
    "left_ee_qw",
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "left_joint_6",
    "left_joint_7",
    "left_trigger",
    "right_ee_x",
    "right_ee_y",
    "right_ee_z",
    "right_ee_qx",
    "right_ee_qy",
    "right_ee_qz",
    "right_ee_qw",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_joint_6",
    "right_joint_7",
    "right_trigger",
]

MODALITY_JSON = {
    "state": {
        "left_eef": {"start": 0, "end": 7},
        "left_arm": {"start": 7, "end": 14},
        "left_trigger": {"start": 14, "end": 15},
        "right_eef": {"start": 15, "end": 22},
        "right_arm": {"start": 22, "end": 29},
        "right_trigger": {"start": 29, "end": 30},
    },
    "action": {
        "left_eef": {"start": 0, "end": 7},
        "left_arm": {"start": 7, "end": 14},
        "left_trigger": {"start": 14, "end": 15},
        "right_eef": {"start": 15, "end": 22},
        "right_arm": {"start": 22, "end": 29},
        "right_trigger": {"start": 29, "end": 30},
    },
    "video": {
        "cam_high": {"original_key": "observation.images.cam_high"},
        "cam_left_wrist": {"original_key": "observation.images.cam_left_wrist"},
        "cam_right_wrist": {"original_key": "observation.images.cam_right_wrist"},
    },
    "annotation": {
        "human.task_description": {"original_key": "task_index"},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and prepare a Tianji Pico LeRobot v2.1 dataset for GR00T N1.7."
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the dataset without writing any files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Back up and replace existing generated files if their content differs.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
    return records


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_vector_feature(features: dict[str, Any], key: str) -> None:
    require(key in features, f"Missing feature: {key}")
    feature = features[key]
    require(feature.get("dtype") == "float32", f"{key} must use float32")
    require(feature.get("shape") == [30], f"{key} must have shape [30]")
    require(
        feature.get("names") == EXPECTED_NAMES,
        f"{key} names/order do not match the expected Tianji 30D layout",
    )


def validate_first_parquet(path: Path) -> str:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return "skipped (pyarrow is not installed)"

    parquet = pq.ParquetFile(path)
    required_columns = {STATE_KEY, ACTION_KEY, "task_index"}
    missing = required_columns.difference(parquet.schema_arrow.names)
    require(not missing, f"{path} is missing parquet columns: {sorted(missing)}")

    table = parquet.read_row_group(0, columns=sorted(required_columns))
    require(table.num_rows > 0, f"{path} has no rows")
    for key in (STATE_KEY, ACTION_KEY):
        value = table[key][0].as_py()
        require(len(value) == 30, f"First {key} value in {path} is not 30D")
    return "passed"


def validate_dataset(dataset_path: Path) -> dict[str, Any]:
    dataset_path = dataset_path.expanduser().resolve()
    meta_path = dataset_path / "meta"
    info_path = meta_path / "info.json"
    episodes_path = meta_path / "episodes.jsonl"
    tasks_path = meta_path / "tasks.jsonl"

    require(dataset_path.is_dir(), f"Dataset directory does not exist: {dataset_path}")
    for path in (info_path, episodes_path, tasks_path):
        require(path.is_file(), f"Required metadata file does not exist: {path}")

    info = load_json(info_path)
    require(str(info.get("codebase_version", "")).startswith("v2"), "Dataset must be LeRobot v2")
    require(info.get("robot_type") == "bi_tianji_marvin", "robot_type must be bi_tianji_marvin")
    require(info.get("fps") == 30, "Expected dataset FPS to be 30")

    features = info.get("features", {})
    validate_vector_feature(features, STATE_KEY)
    validate_vector_feature(features, ACTION_KEY)
    for key in VIDEO_KEYS:
        require(key in features, f"Missing video feature: {key}")
        require(features[key].get("dtype") == "video", f"{key} must use video dtype")

    episodes = load_jsonl(episodes_path)
    tasks = load_jsonl(tasks_path)
    total_episodes = int(info.get("total_episodes", -1))
    require(total_episodes > 0, "total_episodes must be positive")
    require(len(episodes) == total_episodes, "episodes.jsonl count does not match total_episodes")
    require(tasks, "tasks.jsonl is empty")

    chunks_size = int(info.get("chunks_size", 1000))
    data_template = info["data_path"]
    first_parquet = dataset_path / data_template.format(episode_chunk=0, episode_index=0)
    require(first_parquet.is_file(), f"First parquet file does not exist: {first_parquet}")
    parquet_status = validate_first_parquet(first_parquet)

    missing_videos = []
    video_template = info["video_path"]
    for episode_index in range(total_episodes):
        episode_chunk = episode_index // chunks_size
        for video_key in VIDEO_KEYS:
            video_path = dataset_path / video_template.format(
                episode_chunk=episode_chunk,
                episode_index=episode_index,
                video_key=video_key,
            )
            if not video_path.is_file():
                missing_videos.append(video_path)
    if missing_videos:
        raise ValueError(
            f"Missing {len(missing_videos)} video files; first: {missing_videos[0]}"
        )

    return {
        "dataset_path": dataset_path,
        "total_episodes": total_episodes,
        "total_frames": int(info.get("total_frames", -1)),
        "total_videos": total_episodes * len(VIDEO_KEYS),
        "parquet_status": parquet_status,
        "task": tasks[0].get("task", ""),
    }


def atomic_write(path: Path, content: str, force: bool) -> str:
    if path.exists():
        current = path.read_text(encoding="utf-8")
        if current == content:
            return "already up to date"
        if not force:
            raise FileExistsError(f"Refusing to replace existing file without --force: {path}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = path.with_name(f"{path.name}.bak_{timestamp}")
        shutil.copy2(path, backup_path)
        backup_message = f"replaced; backup: {backup_path.name}"
    else:
        backup_message = "created"

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_file.write(content)
        temporary_path = Path(temporary_file.name)
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return backup_message


def main() -> None:
    args = parse_args()
    summary = validate_dataset(args.dataset_path)

    print("Dataset validation passed")
    print(f"  path: {summary['dataset_path']}")
    print(f"  episodes: {summary['total_episodes']}")
    print(f"  frames: {summary['total_frames']}")
    print(f"  videos: {summary['total_videos']}")
    print(f"  first parquet check: {summary['parquet_status']}")
    print(f"  task: {summary['task']}")

    if args.validate_only:
        print("Validation only; no files were written.")
        return

    dataset_path = summary["dataset_path"]
    modality_path = dataset_path / "meta" / "modality.json"
    modality_content = json.dumps(MODALITY_JSON, indent=2, ensure_ascii=True) + "\n"

    modality_result = atomic_write(modality_path, modality_content, args.force)
    print(f"modality.json: {modality_result}: {modality_path}")


if __name__ == "__main__":
    main()
