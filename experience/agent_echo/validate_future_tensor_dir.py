"""
validate_future_tensor_dir — Validate a FutureTensor storage directory.

Generated from validate_future_tensor_dir.viba:
    main := Optional[Stdout[$validation_error str]]
        <- ArgParse[$dir Directory[FutureTensor]]
        <- Import[experience/future_tensor/future_tensor.viba]
        <- { check shape schema, capacity shape, metadata and storage }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path


def _error(msg: str) -> str:
    return f"[ERROR] {msg}"


def _is_valid_uuid(s: str) -> bool:
    try:
        uuid.UUID(s)
        return True
    except ValueError:
        return False


def _validate_tensor_storage(tensor_dir: Path, expected_count: int | None) -> list[str]:
    """Validate a symbolic tensor storage directory. Returns list of errors."""
    errors: list[str] = []
    storage = tensor_dir / "storage"
    if not storage.is_dir():
        errors.append(_error(f"Missing storage directory: {storage}"))
        return errors
    files = []
    for root, _dirs, filenames in os.walk(str(storage)):
        for fn in filenames:
            if fn == "data":
                files.append(Path(root) / fn)
    if not files:
        errors.append(_error(f"No data files found in {storage}"))
    else:
        for f in files:
            if not f.is_file():
                errors.append(_error(f"Not a regular file: {f}"))
        if expected_count is not None and len(files) != expected_count:
            errors.append(_error(
                f"Expected {expected_count} data files, found {len(files)} in {storage}"))
    return errors


def _compute_element_count(shape: list[int]) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def _validate_segment_storage(dir_path: Path, segment: dict) -> list[str]:
    """Validate one logical-view segment's storage on disk."""
    tensor_path = Path(segment["symbolic_tensor_path"])
    shape_before = segment["shape_before"]
    shape_after = segment["shape_after"]
    count = _compute_element_count(shape_after) - _compute_element_count(shape_before)
    return _validate_tensor_storage(tensor_path, count)


def _validate_meta(dir_path: Path, meta: dict) -> list[str]:
    """Validate metadata and logical view consistency."""
    errors: list[str] = []
    if "ft_relative_to" not in meta:
        errors.append(_error("Missing ft_relative_to in ft_meta.json"))
    if "ft_tensor_uid" not in meta or not _is_valid_uuid(meta["ft_tensor_uid"]):
        errors.append(_error("Missing or invalid ft_tensor_uid in ft_meta.json"))
    if "ft_capacity_shape" not in meta:
        errors.append(_error("Missing ft_capacity_shape in ft_meta.json"))
    logical_view = meta.get("logical_view", [])
    if not logical_view:
        errors.append(_error("Missing logical_view in ft_meta.json"))
        return errors
    for i, seg in enumerate(logical_view):
        errors.extend(_validate_segment_storage(dir_path, seg))
    return errors


def validate_future_tensor_dir(dir_path: Path) -> list[str]:
    """Validate a FutureTensor directory.  Returns list of error messages."""
    errors: list[str] = []
    if not dir_path.is_dir():
        return [_error(f"Not a directory: {dir_path}")]
    meta_path = dir_path / "ft_meta.json"
    if not meta_path.is_file():
        return [_error(f"No ft_meta.json found in {dir_path}")]
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError as e:
        return [_error(f"Invalid ft_meta.json: {e}")]
    errors.extend(_validate_meta(dir_path, meta))
    return errors


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate a FutureTensor directory.")
    p.add_argument("--dir", type=Path, required=True)
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    errors = validate_future_tensor_dir(args.dir)
    if errors:
        for err in errors:
            print(err)
        sys.exit(1)
    print(f"Valid: {args.dir}")


if __name__ == "__main__":
    main()
