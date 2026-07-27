#!/usr/bin/env python3
"""Convert the Tianji Pico LeRobot v2.1 dataset from 30D to GR00T's 20D input.

The source dataset is never modified. Both state and action are converted from
``[xyz_mm, quat_xyzw, joints_7, trigger]`` per arm to
``[xyz_m, rot6d_columns, trigger]`` per arm. Videos are hard-linked by default.
"""

from __future__ import annotations

import argparse
import copy
import errno
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


STATE_KEY = "observation.state"
ACTION_KEY = "action"
VIDEO_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)

RAW_NAMES = [
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

STATE_20D_NAMES = [
    "left_ee_x",
    "left_ee_y",
    "left_ee_z",
    *(f"left_ee_r6d_{index}" for index in range(1, 7)),
    "left_trigger",
    "right_ee_x",
    "right_ee_y",
    "right_ee_z",
    *(f"right_ee_r6d_{index}" for index in range(1, 7)),
    "right_trigger",
]

# This is the exact order consumed by robot_client_tianji_openpi.py.
ACTION_20D_NAMES = [
    "left_x",
    "left_y",
    "left_z",
    *(f"left_r6d_{index}" for index in range(1, 7)),
    "pico_left_trigger",
    "right_x",
    "right_y",
    "right_z",
    *(f"right_r6d_{index}" for index in range(1, 7)),
    "pico_right_trigger",
]

MODALITY_JSON = {
    "state": {
        "tianji_ee": {"start": 0, "end": 20},
    },
    "action": {
        "tianji_ee": {"start": 0, "end": 20},
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
        description="Create a separate 20D GR00T dataset from Tianji Pico LeRobot data."
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Default: a sibling directory named <dataset>_groot_20d.",
    )
    parser.add_argument(
        "--video-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="Use hardlinks by default so videos consume no additional disk space.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the 30D source dataset without creating the output dataset.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output by first moving it to a timestamped backup.",
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


def _validate_vector_feature(features: dict[str, Any], key: str) -> None:
    require(key in features, f"Missing feature: {key}")
    feature = features[key]
    require(feature.get("dtype") == "float32", f"{key} must use float32")
    require(feature.get("shape") == [30], f"{key} must have shape [30]")
    require(
        feature.get("names") == RAW_NAMES,
        f"{key} names/order do not match the Tianji 30D schema",
    )


def validate_source(dataset_path: Path) -> dict[str, Any]:
    dataset_path = dataset_path.expanduser().resolve()
    meta_path = dataset_path / "meta"
    info_path = meta_path / "info.json"
    episodes_path = meta_path / "episodes.jsonl"
    tasks_path = meta_path / "tasks.jsonl"
    episode_stats_path = meta_path / "episodes_stats.jsonl"

    require(dataset_path.is_dir(), f"Dataset directory does not exist: {dataset_path}")
    for path in (info_path, episodes_path, tasks_path, episode_stats_path):
        require(path.is_file(), f"Required metadata file does not exist: {path}")

    info = load_json(info_path)
    require(str(info.get("codebase_version", "")).startswith("v2"), "Expected LeRobot v2")
    require(info.get("robot_type") == "bi_tianji_marvin", "Unexpected robot_type")
    require(info.get("fps") == 30, "Expected dataset FPS to be 30")

    features = info.get("features", {})
    _validate_vector_feature(features, STATE_KEY)
    _validate_vector_feature(features, ACTION_KEY)
    for key in VIDEO_KEYS:
        require(key in features, f"Missing video feature: {key}")
        require(features[key].get("dtype") == "video", f"{key} must use video dtype")

    episodes = load_jsonl(episodes_path)
    tasks = load_jsonl(tasks_path)
    episode_stats = load_jsonl(episode_stats_path)
    total_episodes = int(info.get("total_episodes", -1))
    require(total_episodes > 0, "total_episodes must be positive")
    require(len(episodes) == total_episodes, "episodes.jsonl count does not match info.json")
    require(len(episode_stats) == total_episodes, "episodes_stats.jsonl count is incorrect")
    require(tasks, "tasks.jsonl is empty")

    episode_indices = [int(record["episode_index"]) for record in episodes]
    require(
        episode_indices == list(range(total_episodes)),
        "episode indices must be contiguous and ordered from zero",
    )
    stats_by_episode = {int(record["episode_index"]): record for record in episode_stats}
    require(len(stats_by_episode) == total_episodes, "Duplicate episode statistics found")

    chunks_size = int(info.get("chunks_size", 1000))
    data_template = info["data_path"]
    video_template = info["video_path"]
    parquet_paths: list[Path] = []
    video_paths: list[Path] = []
    total_rows = 0
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        episode_chunk = episode_index // chunks_size
        parquet_path = dataset_path / data_template.format(
            episode_chunk=episode_chunk, episode_index=episode_index
        )
        require(parquet_path.is_file(), f"Missing parquet file: {parquet_path}")
        parquet = pq.ParquetFile(parquet_path)
        missing = {STATE_KEY, ACTION_KEY, "task_index"}.difference(parquet.schema_arrow.names)
        require(not missing, f"{parquet_path} is missing columns: {sorted(missing)}")
        require(
            parquet.metadata.num_rows == int(episode["length"]),
            f"Row count does not match episodes.jsonl: {parquet_path}",
        )
        total_rows += parquet.metadata.num_rows
        parquet_paths.append(parquet_path)

        for video_key in VIDEO_KEYS:
            video_path = dataset_path / video_template.format(
                episode_chunk=episode_chunk,
                episode_index=episode_index,
                video_key=video_key,
            )
            require(video_path.is_file(), f"Missing video file: {video_path}")
            video_paths.append(video_path)

    require(total_rows == int(info["total_frames"]), "Parquet rows do not match total_frames")
    require(len(video_paths) == int(info["total_videos"]), "Video count is inconsistent")
    return {
        "dataset_path": dataset_path,
        "info": info,
        "episodes": episodes,
        "tasks": tasks,
        "stats_by_episode": stats_by_episode,
        "parquet_paths": parquet_paths,
        "video_paths": video_paths,
    }


def convert_30d_to_20d(values: np.ndarray, context: str = "vector") -> np.ndarray:
    """Apply the same mm+quat to m+column-rot6d conversion as TianjiPolicy."""
    vectors = np.asarray(values, dtype=np.float32)
    require(vectors.ndim == 2 and vectors.shape[1] == 30, f"{context} must have shape (N, 30)")
    require(np.isfinite(vectors).all(), f"{context} contains non-finite values")

    converted_sides = []
    for side_name, start in (("left", 0), ("right", 15)):
        side = vectors[:, start : start + 15]
        quaternion = side[:, 3:7]
        norms = np.linalg.norm(quaternion, axis=1)
        require(np.isfinite(norms).all(), f"{context} has a non-finite {side_name} quaternion")
        require((norms > 1e-8).all(), f"{context} has a zero {side_name} quaternion")

        rotation = Rotation.from_quat(quaternion).as_matrix()
        rot6d = np.concatenate([rotation[:, :, 0], rotation[:, :, 1]], axis=1)
        converted_sides.append(
            np.concatenate(
                [side[:, 0:3] * 1e-3, rot6d.astype(np.float32), side[:, 14:15]],
                axis=1,
            )
        )

    return np.ascontiguousarray(np.concatenate(converted_sides, axis=1), dtype=np.float32)


def _column_to_numpy(table: pa.Table, key: str) -> np.ndarray:
    return np.asarray(table[key].to_pylist(), dtype=np.float32)


def _fixed_size_list(values: np.ndarray) -> pa.FixedSizeListArray:
    flat_values = pa.array(values.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat_values, 20)


def _update_huggingface_metadata(table: pa.Table) -> pa.Table:
    metadata = dict(table.schema.metadata or {})
    encoded = metadata.get(b"huggingface")
    if encoded is None:
        return table

    huggingface = json.loads(encoded.decode("utf-8"))
    features = huggingface.get("info", {}).get("features", {})
    for key in (STATE_KEY, ACTION_KEY):
        require(key in features, f"Parquet Hugging Face metadata is missing {key}")
        require(features[key].get("_type") == "Sequence", f"Unexpected metadata for {key}")
        features[key]["length"] = 20
    metadata[b"huggingface"] = json.dumps(huggingface, separators=(",", ":")).encode("utf-8")
    return table.replace_schema_metadata(metadata)


def _replace_vector_column(table: pa.Table, key: str, values: np.ndarray) -> pa.Table:
    index = table.schema.get_field_index(key)
    require(index >= 0, f"Missing parquet column: {key}")
    old_field = table.schema.field(index)
    array = _fixed_size_list(values)
    field = pa.field(key, array.type, nullable=old_field.nullable, metadata=old_field.metadata)
    return table.set_column(index, field, array)


def _vector_stats(values: np.ndarray) -> dict[str, Any]:
    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "count": [int(values.shape[0])],
    }


def _convert_parquet(source: Path, target: Path) -> tuple[np.ndarray, np.ndarray]:
    parquet = pq.ParquetFile(source)
    table = parquet.read()
    state = convert_30d_to_20d(_column_to_numpy(table, STATE_KEY), f"{source}:{STATE_KEY}")
    action = convert_30d_to_20d(_column_to_numpy(table, ACTION_KEY), f"{source}:{ACTION_KEY}")

    table = _replace_vector_column(table, STATE_KEY, state)
    table = _replace_vector_column(table, ACTION_KEY, action)
    table = _update_huggingface_metadata(table)
    target.parent.mkdir(parents=True, exist_ok=True)
    row_group_size = parquet.metadata.row_group(0).num_rows
    pq.write_table(table, target, compression="snappy", row_group_size=row_group_size)
    return state, action


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")


def _link_or_copy_video(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(source, target)
        except OSError as exc:
            if exc.errno == errno.EXDEV:
                raise OSError(
                    f"Cannot hardlink across filesystems: {source} -> {target}. "
                    "Re-run with --video-mode copy."
                ) from exc
            raise
        require(os.path.samefile(source, target), f"Hardlink verification failed: {target}")
    else:
        shutil.copy2(source, target)
    require(source.stat().st_size == target.stat().st_size, f"Video size mismatch: {target}")


def _validate_output(source_summary: dict[str, Any], output_path: Path, video_mode: str) -> None:
    info = load_json(output_path / "meta" / "info.json")
    require(info["features"][STATE_KEY]["shape"] == [20], "Output state metadata is not 20D")
    require(info["features"][ACTION_KEY]["shape"] == [20], "Output action metadata is not 20D")

    total_rows = 0
    for source_path in source_summary["parquet_paths"]:
        target_path = output_path / source_path.relative_to(source_summary["dataset_path"])
        parquet = pq.ParquetFile(target_path)
        require(
            parquet.schema_arrow.field(STATE_KEY).type.list_size == 20,
            f"Bad state schema: {target_path}",
        )
        require(
            parquet.schema_arrow.field(ACTION_KEY).type.list_size == 20,
            f"Bad action schema: {target_path}",
        )
        total_rows += parquet.metadata.num_rows
    require(total_rows == int(info["total_frames"]), "Output frame count is inconsistent")

    for source_path in source_summary["video_paths"]:
        target_path = output_path / source_path.relative_to(source_summary["dataset_path"])
        require(target_path.is_file(), f"Missing output video: {target_path}")
        require(
            source_path.stat().st_size == target_path.stat().st_size,
            f"Bad output video: {target_path}",
        )
        if video_mode == "hardlink":
            require(
                os.path.samefile(source_path, target_path),
                f"Video is not a hardlink: {target_path}",
            )


def create_dataset(
    source_summary: dict[str, Any], output_path: Path, video_mode: str, force: bool
) -> Path | None:
    source_path = source_summary["dataset_path"]
    output_path = output_path.expanduser().resolve()
    require(output_path != source_path, "Output path must differ from the source dataset")
    require(
        source_path not in output_path.parents,
        "Output path cannot be inside the source dataset",
    )
    require(output_path not in source_path.parents, "Output path cannot contain the source dataset")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        raise FileExistsError(f"Output already exists; pass --force to back it up: {output_path}")

    temporary_path = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent)
    )
    backup_path: Path | None = None
    try:
        info = copy.deepcopy(source_summary["info"])
        info["features"][STATE_KEY]["shape"] = [20]
        info["features"][STATE_KEY]["names"] = STATE_20D_NAMES
        info["features"][ACTION_KEY]["shape"] = [20]
        info["features"][ACTION_KEY]["names"] = ACTION_20D_NAMES
        _write_json(temporary_path / "meta" / "info.json", info)
        shutil.copy2(
            source_path / "meta" / "episodes.jsonl",
            temporary_path / "meta" / "episodes.jsonl",
        )
        shutil.copy2(source_path / "meta" / "tasks.jsonl", temporary_path / "meta" / "tasks.jsonl")
        _write_json(temporary_path / "meta" / "modality.json", MODALITY_JSON)

        converted_episode_stats = []
        total = len(source_summary["episodes"])
        for position, episode in enumerate(source_summary["episodes"], start=1):
            episode_index = int(episode["episode_index"])
            source_parquet = source_summary["parquet_paths"][episode_index]
            target_parquet = temporary_path / source_parquet.relative_to(source_path)
            state, action = _convert_parquet(source_parquet, target_parquet)

            episode_stats = copy.deepcopy(source_summary["stats_by_episode"][episode_index])
            episode_stats["stats"][STATE_KEY] = _vector_stats(state)
            episode_stats["stats"][ACTION_KEY] = _vector_stats(action)
            converted_episode_stats.append(episode_stats)
            print(f"Converted parquet {position}/{total}: episode {episode_index:06d}")
        _write_jsonl(temporary_path / "meta" / "episodes_stats.jsonl", converted_episode_stats)

        for position, source_video in enumerate(source_summary["video_paths"], start=1):
            target_video = temporary_path / source_video.relative_to(source_path)
            _link_or_copy_video(source_video, target_video, video_mode)
            if position % 15 == 0 or position == len(source_summary["video_paths"]):
                print(f"Prepared videos {position}/{len(source_summary['video_paths'])}")

        _validate_output(source_summary, temporary_path, video_mode)

        if output_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = output_path.with_name(f"{output_path.name}.bak_{timestamp}")
            output_path.rename(backup_path)
        try:
            temporary_path.rename(output_path)
        except BaseException:
            if backup_path is not None and not output_path.exists():
                backup_path.rename(output_path)
            raise
        return backup_path
    except BaseException:
        shutil.rmtree(temporary_path, ignore_errors=True)
        raise


def main() -> None:
    args = parse_args()
    summary = validate_source(args.dataset_path)
    source_path = summary["dataset_path"]
    output_path = args.output_path or source_path.with_name(f"{source_path.name}_groot_20d")

    print("Source validation passed")
    print(f"  path: {source_path}")
    print(f"  episodes: {len(summary['episodes'])}")
    print(f"  frames: {summary['info']['total_frames']}")
    print(f"  videos: {len(summary['video_paths'])}")
    print(f"  output: {output_path}")
    if args.validate_only:
        print("Validation only; no files were written.")
        return

    backup_path = create_dataset(summary, output_path, args.video_mode, args.force)
    print(f"Created 20D GR00T dataset: {output_path.expanduser().resolve()}")
    if backup_path is not None:
        print(f"Previous output backup: {backup_path}")
    print("Next: run gr00t/data/stats.py to create fresh GR00T statistics.")


if __name__ == "__main__":
    main()
