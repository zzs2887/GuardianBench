from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from . import __version__
from .config import ROOT, load_config
from .data import load_samples, sha256_file
from .metrics import extract_verdict, summarize


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def versions() -> dict:
    result = {"python": platform.python_version(), "guardianbench_eval": __version__}
    for name in ("pyarrow", "Pillow", "PyYAML", "openai", "torch", "transformers", "accelerate"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def evaluate(samples, backend, output: Path) -> dict:

    records = []
    interrupted = False
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        try:
            for index, sample in enumerate(samples, 1):
                start = perf_counter()
                record = {"uid": sample.uid, "pair_key": sample.pair_key, "label": sample.label}
                try:
                    prediction = backend.generate(sample.model_input)
                    if not isinstance(prediction.completion, str) or not prediction.completion.strip():
                        raise ValueError("Empty or nontext completion")
                    verdict = extract_verdict(prediction.completion)
                    record.update(status="ok", completion=prediction.completion, prediction=verdict,
                                  correct=verdict == sample.label, metadata=prediction.metadata)
                except Exception as exc:


                    error = {"type": type(exc).__name__}
                    if isinstance(getattr(exc, "error_type", None), str):
                        error["type"] = exc.error_type
                    code = getattr(exc, "status_code", None)
                    if isinstance(code, int):
                        error["http_status"] = code
                    record.update(status="error", completion=None, prediction=None,
                                  correct=False, error=error)
                record["elapsed_seconds"] = round(perf_counter() - start, 6)
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                records.append(record)
                print(f"[{index}/{len(samples)}] uid={sample.uid} {record['status']}", flush=True)
        except KeyboardInterrupt:
            interrupted = True
        finally:
            summary = summarize(samples, records)
            if interrupted:
                summary["status"] = "interrupted"
            write_json(output / "metrics.json", summary)
    return summary


def main(backend_name: str) -> int:
    parser = argparse.ArgumentParser(description=f"Evaluate GuardianBench via {backend_name} inference.")
    parser.add_argument("--config", type=Path, default=ROOT / f"configs/{backend_name}.yaml")
    parser.add_argument("--data", type=Path, help="Override the externally obtained test.parquet path")
    parser.add_argument("--output-dir", type=Path, help="New directory for this run (never overwritten)")
    parser.add_argument("--limit-pairs", type=int, help="Evaluate the first N complete pairs")
    parser.add_argument("--dry-run", action="store_true", help="Validate config/data/images without inference or output files")
    args = parser.parse_args()
    try:
        config = load_config(args.config, backend_name, data=args.data, output=args.output_dir, limit_pairs=args.limit_pairs)
        system_prompt = Path(config["system_prompt_path"]).read_text(encoding="utf-8").strip()
        if not system_prompt:
            raise ValueError("System prompt must not be empty")
        samples, dataset = load_samples(Path(config["data_path"]), config.get("limit_pairs"))

        from .backends import validate_backend_config
        config[backend_name] = validate_backend_config(backend_name, config[backend_name])
        if args.dry_run:
            print(json.dumps({"status": "validated_only", "backend": backend_name,
                              "dataset": {k: v for k, v in dataset.items() if k != "selected_uids"},
                              "prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest()}, indent=2))
            return 0
        output = Path(config["output_dir"])
        if output.exists():
            raise ValueError("Output directory already exists; choose a new --output-dir")
        from .backends import LocalBackend, OpenAIBackend
        backend = (LocalBackend if backend_name == "local" else OpenAIBackend)(config[backend_name], system_prompt)
        try:
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "run.json", {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "backend": backend_name, "config": config, "dataset": dataset,
                "system_prompt": system_prompt,
                "prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
                "versions": versions(),
                "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(ROOT.glob("guardianbench_eval/*.py"))},
            })
            summary = evaluate(samples, backend, output)
        finally:
            close = getattr(backend, "close", None)
            if close:
                close()
        print(json.dumps(summary, indent=2))
        print(f"Results: {output}")
        return 0 if summary["status"] == "complete" else 2
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"Configuration/data error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Evaluation setup failed ({type(exc).__name__}). Check dependencies, model path, endpoint, and key environment variable.", file=sys.stderr)
        return 2
