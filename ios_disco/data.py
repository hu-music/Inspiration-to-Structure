import json
import os
import re
from collections import Counter
from typing import Any, Dict, Optional

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


ABC_BLOCK_RE = re.compile(
    r"<(?P<section>[a-zA-Z_-]+)>\s*\{\{(?P<abc>.*?)\}\}",
    re.DOTALL,
)


def load_dataset_any(path: str):
    if os.path.isdir(path):
        return load_from_disk(path)

    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return Dataset.from_list(rows)

    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            rows = obj
        elif isinstance(obj, dict):
            rows = obj.get("data") or obj.get("train") or list(obj.values())[0]
        else:
            raise ValueError("Unsupported JSON structure")
        return Dataset.from_list(rows)

    # Treat anything else as a HuggingFace Hub dataset id, for example
    # "cszhu09876/ios-disco-abc".
    return load_dataset(path)


def select_split(ds, split: Optional[str] = None):
    if isinstance(ds, DatasetDict):
        if split is None:
            split = "train" if "train" in ds else list(ds.keys())[0]
        return ds[split]
    return ds


def row_to_text(row: Dict[str, Any]) -> str:
    if "text" in row and isinstance(row["text"], str):
        return row["text"]

    for key in ("conversations", "messages"):
        if key in row:
            parts = []
            for msg in row[key]:
                parts.append(msg.get("content") or msg.get("value") or "")
            return "\n".join(parts)

    return json.dumps(row, ensure_ascii=False)


def inspect_dataset(path: str, split: Optional[str] = None, max_scan: Optional[int] = None):
    ds = select_split(load_dataset_any(path), split)
    n = len(ds) if max_scan is None else min(len(ds), max_scan)

    block_count_counter = Counter()
    section_counter = Counter()
    role_counter = Counter()
    bad_no_abc = 0

    for i in range(n):
        row = ds[i]
        if "conversations" in row:
            for msg in row["conversations"]:
                role_counter[msg.get("role") or msg.get("from")] += 1

        blocks = list(ABC_BLOCK_RE.finditer(row_to_text(row)))
        block_count_counter[len(blocks)] += 1
        if not blocks:
            bad_no_abc += 1
            continue
        section_counter.update(m.group("section").lower() for m in blocks)

    return {
        "num_rows_scanned": n,
        "columns": ds.column_names,
        "role_counts": role_counter,
        "block_count_per_sample": block_count_counter,
        "section_counts": section_counter,
        "intra_like_samples": block_count_counter[3],
        "inter_like_samples": block_count_counter[4],
        "samples_without_abc_blocks": bad_no_abc,
    }
