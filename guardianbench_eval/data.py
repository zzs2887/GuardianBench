from __future__ import annotations

import base64
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


@dataclass(frozen=True)
class ModelInput:
    instruction: str
    images: tuple[bytes, ...]


@dataclass(frozen=True)
class Sample:
    uid: str
    pair_key: str
    label: str
    model_input: ModelInput


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_samples(path: Path, limit_pairs: int | None = None) -> tuple[list[Sample], dict]:

    if limit_pairs is not None and (type(limit_pairs) is not int or limit_pairs < 1):
        raise ValueError("limit_pairs must be a positive integer")
    columns = ["uid", "pair_key", "label", "instruction", "images"]
    parquet = pq.ParquetFile(path)
    missing = set(columns) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"Missing public Parquet columns: {sorted(missing)}")
    samples, seen, pairs = [], set(), defaultdict(list)
    sizes = Counter()
    for batch in parquet.iter_batches(batch_size=64, columns=columns):
        for row in batch.to_pylist():
            uid = row["uid"]
            if isinstance(uid, bool) or not isinstance(uid, (str, int)) or str(uid).strip() == "":
                raise ValueError("Every row needs a nonempty string/integer uid")
            uid = str(uid)
            if uid in seen:
                raise ValueError(f"Duplicate uid: {uid}")
            seen.add(uid)
            pair_key, label, instruction = row["pair_key"], row["label"], row["instruction"]
            if not isinstance(pair_key, str) or not pair_key.strip():
                raise ValueError(f"uid {uid}: pair_key must be nonempty text")
            if label not in ("Safe", "Unsafe"):
                raise ValueError(f"uid {uid}: label must be Safe or Unsafe")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(f"uid {uid}: instruction must be nonempty text")
            encoded = row["images"]
            if not isinstance(encoded, list) or not encoded:
                raise ValueError(f"uid {uid}: images must be a nonempty list of base64 strings")
            images = []
            for value in encoded:
                try:
                    if not isinstance(value, str):
                        raise ValueError("expected base64 text")
                    raw = base64.b64decode(value, validate=True)
                    with Image.open(BytesIO(raw)) as image:
                        image.load()
                        if image.format not in {"PNG", "JPEG", "WEBP", "GIF"} or getattr(image, "n_frames", 1) != 1:
                            raise ValueError("expected a static PNG, JPEG, WEBP, or GIF")
                        sizes[f"{image.width}x{image.height}"] += 1
                    images.append(raw)
                except Exception as exc:
                    raise ValueError(f"uid {uid}: invalid embedded image ({type(exc).__name__})") from None
            sample = Sample(uid, pair_key, label, ModelInput(instruction, tuple(images)))
            samples.append(sample)
            pairs[pair_key].append(sample)
    if not samples:
        raise ValueError("The Parquet file contains no samples")
    for key, members in pairs.items():
        if len(members) != 2 or {s.label for s in members} != {"Safe", "Unsafe"}:
            raise ValueError(f"Pair {key!r} must contain exactly one Safe and one Unsafe row")
        if members[0].model_input.images != members[1].model_input.images:
            raise ValueError(f"Pair {key!r} must share the same embedded images")
    selected = set(list(pairs)[:limit_pairs]) if limit_pairs else set(pairs)
    chosen = [s for s in samples if s.pair_key in selected]
    return chosen, {
        "filename": path.name,
        "sha256": sha256_file(path),
        "total_rows": len(samples),
        "total_pairs": len(pairs),
        "selected_rows": len(chosen),
        "selected_pairs": len(selected),
        "labels": dict(Counter(s.label for s in chosen)),
        "image_sizes_full_file": dict(sizes),
        "selection": "all" if len(chosen) == len(samples) else "first_complete_pairs",
        "selected_uids": [s.uid for s in chosen],
    }
