from __future__ import annotations
import json
import os
import zipfile
from datetime import datetime
from typing import Iterable, Any


def write_transcript_ndjson(path: str, step_records: Iterable[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in step_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_summary_json(path: str, summary: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def pack_evidence_bundle(
    out_zip: str,
    files: list[str],
    extra_meta: dict[str, Any] | None = None,
) -> None:
    """
    Pack given files plus a small manifest.json into a single ZIP.
    """
    os.makedirs(os.path.dirname(out_zip), exist_ok=True)
    manifest = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "files": [os.path.basename(p) for p in files],
    }
    if extra_meta:
        manifest.update(extra_meta)

    tmp_manifest = os.path.join(os.path.dirname(out_zip), "manifest.json")
    with open(tmp_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(tmp_manifest, arcname="manifest.json")
        for p in files:
            z.write(p, arcname=os.path.basename(p))