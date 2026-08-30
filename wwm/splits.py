"""Deterministic, leakage-audited WWM dataset indexing.

The index is scenario-first: all windows from one CSI trajectory stay in the
same split.  Train/validation files are stratified by speed and ordered by a
stable SHA-1 key, so rebuilding an index on another machine gives the same
rows.  An independent test root is never used to choose a checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np


_SPEED_RE = re.compile(r"_(\d+(?:p\d+|\.\d+)?)kmh(?:_|$)", re.IGNORECASE)


@dataclass(frozen=True)
class Scenario:
    """One trajectory file and its metadata."""

    base: str
    split: str
    city: str
    speed_kmh: float
    samples: int
    h_path: str
    pos_path: str
    meta_path: str


def _read_json(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def infer_city(base: str, meta: Mapping[str, object]) -> str:
    scenario = meta.get("scenario", {})
    if isinstance(scenario, Mapping) and scenario.get("city_key"):
        return str(scenario["city_key"])
    # Dataset naming convention is <scenario_id>_<city>_...; retain the
    # complete known city token when it contains underscores.
    known = (
        "beijing_dashilar_qianmen", "chicago_loop", "hongkong_central",
        "shanghai_nanjing_road", "singapore_cbd", "tokyo_shinjuku",
        "wall_street", "munich", "etoile",
    )
    for city in known:
        if city in base.lower():
            return city
    parts = base.split("_")
    if len(parts) >= 2:
        return parts[1]
    raise ValueError(f"Cannot infer city from scenario name: {base}")


def infer_speed(base: str, meta: Mapping[str, object]) -> float:
    scenario = meta.get("scenario", {})
    if isinstance(scenario, Mapping) and scenario.get("speed_kmh") is not None:
        return float(scenario["speed_kmh"])
    match = _SPEED_RE.search(base)
    if match:
        return float(match.group(1).replace("p", "."))
    raise ValueError(f"Cannot infer speed_kmh from scenario name: {base}")


def discover_scenarios(
    dataset_root: Path,
    split: str,
    city: Optional[str] = None,
    max_files: Optional[int] = None,
) -> List[Scenario]:
    """Discover valid ``H/pos/meta`` triplets under ``dataset_root/split``."""

    root = Path(dataset_root).expanduser().resolve()
    h_dir = root / split / "H"
    if not h_dir.is_dir():
        raise FileNotFoundError(f"Missing H directory: {h_dir}")
    files = sorted(h_dir.glob("*_H.npy"), key=lambda path: path.name)
    if max_files and max_files > 0:
        files = files[: int(max_files)]
    scenarios: List[Scenario] = []
    for h_path in files:
        base = h_path.name[: -len("_H.npy")]
        pos_path = root / split / "pos" / f"{base}_pos.npy"
        meta_path = root / split / "meta" / f"{base}_meta.json"
        if not pos_path.is_file():
            raise FileNotFoundError(f"Missing position file for {base}: {pos_path}")
        meta = _read_json(meta_path)
        samples = int(np.load(h_path, mmap_mode="r").shape[0])
        record = Scenario(
            base=base,
            split=split,
            city=infer_city(base, meta),
            speed_kmh=infer_speed(base, meta),
            samples=samples,
            h_path=str(h_path),
            pos_path=str(pos_path),
            meta_path=str(meta_path),
        )
        if city and city != "any" and record.city != city:
            continue
        if record.samples > 0:
            scenarios.append(record)
    if not scenarios:
        raise RuntimeError(f"No scenarios found under {h_dir} (city={city!r})")
    return scenarios


def stable_validation_split(
    scenarios: Sequence[Scenario], validation_fraction: float = 0.1
) -> Tuple[List[Scenario], List[Scenario]]:
    """Split files by speed with a stable hash (never split windows)."""

    fraction = float(validation_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    train: List[Scenario] = []
    validation: List[Scenario] = []
    by_speed: Dict[float, List[Scenario]] = {}
    for item in scenarios:
        by_speed.setdefault(round(float(item.speed_kmh), 6), []).append(item)
    for speed, group in sorted(by_speed.items()):
        ordered = sorted(
            group,
            key=lambda item: hashlib.sha1(
                f"wwm-val:{speed:g}:{item.base}".encode("utf-8")
            ).hexdigest(),
        )
        count = min(len(ordered) - 1, max(1, int(round(len(ordered) * fraction))))
        validation.extend(ordered[:count])
        train.extend(ordered[count:])
    if not train or not validation:
        raise RuntimeError("Stable split produced an empty train or validation set")
    return (
        sorted(train, key=lambda item: item.base),
        sorted(validation, key=lambda item: item.base),
    )


def _quality_lookup(path: Optional[Path]) -> Dict[Tuple[str, int], Dict[str, float]]:
    if path is None:
        return {}
    data = np.load(str(path), allow_pickle=True)
    required = {"scenario_bases", "scenario_id", "sample_idx", "accepted", "sampler_weight"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"Quality index is missing fields: {sorted(missing)}")
    bases = data["scenario_bases"]
    result: Dict[Tuple[str, int], Dict[str, float]] = {}
    rms = data["context_rms"] if "context_rms" in data.files else None
    for row in range(len(data["sample_idx"])):
        key = (str(bases[int(data["scenario_id"][row])]), int(data["sample_idx"][row]))
        result[key] = {
            "accepted": float(bool(data["accepted"][row])),
            "sampler_weight": float(data["sampler_weight"][row]),
            "context_rms": float(rms[row]) if rms is not None else float("nan"),
        }
    return result


def iter_index_rows(
    scenarios: Sequence[Scenario],
    split_name: str,
    dataset_root: Path,
    quality_index: Optional[Path] = None,
    accepted_only: bool = False,
) -> Iterator[Dict[str, object]]:
    """Yield one row per sample with paths relative to ``dataset_root``."""

    quality = _quality_lookup(quality_index)
    root = Path(dataset_root).expanduser().resolve()
    for scenario in sorted(scenarios, key=lambda item: item.base):
        q_missing = 0
        for sample_idx in range(scenario.samples):
            entry = quality.get((scenario.base, sample_idx))
            if accepted_only and (entry is None or not bool(entry["accepted"])):
                q_missing += 1
                continue
            row: Dict[str, object] = {
                "split": split_name,
                "scenario_base": scenario.base,
                "city": scenario.city,
                "speed_kmh": scenario.speed_kmh,
                "sample_idx": sample_idx,
                "h_path": str(Path(scenario.h_path).resolve().relative_to(root)),
                "pos_path": str(Path(scenario.pos_path).resolve().relative_to(root)),
                "meta_path": str(Path(scenario.meta_path).resolve().relative_to(root)),
            }
            if entry is not None:
                row.update(entry)
            else:
                row.update({"accepted": "", "sampler_weight": "", "context_rms": ""})
            yield row


def audit_splits(splits: Mapping[str, Sequence[Scenario]], hash_contents: bool = False) -> Dict[str, object]:
    """Return overlap counts and optional byte-level H/pos hash audits."""

    bases = {name: {item.base for item in items} for name, items in splits.items()}
    overlap: Dict[str, int] = {}
    names = list(bases)
    for left_idx, left in enumerate(names):
        for right in names[left_idx + 1 :]:
            overlap[f"{left}_{right}"] = len(bases[left] & bases[right])
    report: Dict[str, object] = {
        "scenario_counts": {name: len(items) for name, items in splits.items()},
        "sample_counts": {name: sum(item.samples for item in items) for name, items in splits.items()},
        "base_name_overlap": overlap,
    }
    if any(overlap.values()):
        raise RuntimeError(f"Scenario leakage detected: {overlap}")
    if hash_contents:
        hashes: Dict[str, Dict[str, Set[str]]] = {}
        for name, items in splits.items():
            hashes[name] = {"h": set(), "pos": set()}
            for item in items:
                for kind, path in (("h", item.h_path), ("pos", item.pos_path)):
                    digest = hashlib.sha256()
                    with Path(path).open("rb") as handle:
                        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                            digest.update(chunk)
                    hashes[name][kind].add(digest.hexdigest())
        content_overlap: Dict[str, int] = {}
        for left_idx, left in enumerate(names):
            for right in names[left_idx + 1 :]:
                for kind in ("h", "pos"):
                    content_overlap[f"{left}_{right}_{kind}"] = len(
                        hashes[left][kind] & hashes[right][kind]
                    )
        report["content_hash_overlap"] = content_overlap
    return report


def build_protocol_index(
    dataset_root: Path,
    train_split: str = "train",
    test_root: Optional[Path] = None,
    test_split: Optional[str] = None,
    train_city: Optional[str] = None,
    test_city: Optional[str] = None,
    validation_fraction: float = 0.1,
    train_quality_index: Optional[Path] = None,
    test_quality_index: Optional[Path] = None,
    accepted_only: bool = False,
    audit_contents: bool = False,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Build rows and an audit manifest for the standard WWM protocol."""

    root = Path(dataset_root).expanduser().resolve()
    external_root = Path(test_root).expanduser().resolve() if test_root else root
    external_split = test_split or "test"
    all_train = discover_scenarios(root, train_split, city=train_city)
    train, validation = stable_validation_split(all_train, validation_fraction)
    test = discover_scenarios(external_root, external_split, city=test_city)
    split_map = {"train": train, "validation": validation, "test": test}
    audit = audit_splits(split_map, hash_contents=audit_contents)
    audit.update({"train_root": str(root), "test_root": str(external_root), "test_split": external_split})
    rows: List[Dict[str, object]] = []
    rows.extend(iter_index_rows(train, "train", root, train_quality_index, accepted_only))
    rows.extend(iter_index_rows(validation, "validation", root, train_quality_index, accepted_only))
    rows.extend(iter_index_rows(test, "test", external_root, test_quality_index, accepted_only))
    return rows, audit


def write_index(rows: Iterable[Mapping[str, object]], output_csv: Path) -> int:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = [
        "split", "scenario_base", "city", "speed_kmh", "sample_idx",
        "h_path", "pos_path", "meta_path", "accepted", "sampler_weight", "context_rms",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    return len(rows)


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--test-root", type=Path)
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--train-city")
    parser.add_argument("--test-city")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--train-quality-index", type=Path)
    parser.add_argument("--test-quality-index", type=Path)
    parser.add_argument("--accepted-only", action="store_true")
    parser.add_argument("--audit-contents", action="store_true", help="SHA-256 every H/pos file (slow)")
    args = parser.parse_args()
    rows, audit = build_protocol_index(
        args.dataset_root,
        train_split=args.train_split,
        test_root=args.test_root,
        test_split=args.test_split,
        train_city=args.train_city,
        test_city=args.test_city,
        validation_fraction=args.validation_fraction,
        train_quality_index=args.train_quality_index,
        test_quality_index=args.test_quality_index,
        accepted_only=args.accepted_only,
        audit_contents=args.audit_contents,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    count = write_index(rows, args.output_dir / "index.csv")
    manifest = {"format_version": 1, "rows": count, "audit": audit}
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
