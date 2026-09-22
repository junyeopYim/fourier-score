"""Check public notebooks; optionally execute fresh CPU smoke runs.

Execution uses this Python interpreter through a temporary kernelspec, leaves
the source notebook untouched, and saves the executed copy and report under
saved/notebook_checks. GMM experiment data stays under saved/gmm_oracle.
"""

from __future__ import annotations

import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
import uuid

import nbformat


ROOT = Path(__file__).resolve().parents[1]
KOREAN = re.compile(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]")
RETIRED_METHOD = "fourier_gaussian_unscaled"
CHECK_METADATA = "fourier_score_check"


def source_fingerprint(notebook):
    """Fingerprint prose and code, independent of outputs and cell IDs."""
    cells = [(cell.cell_type, cell.source) for cell in notebook.cells]
    return hashlib.sha256(json.dumps(cells, ensure_ascii=False).encode()).hexdigest()


def validate_notebook(path, *, check_execution=True):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    errors = []
    counts = []
    nonempty_code = 0
    has_execution = False
    # Transform IPython syntax without executing it, then use Python's parser.
    from IPython.core.inputtransformer2 import TransformerManager

    transformer = TransformerManager()
    for number, cell in enumerate(notebook.cells, start=1):
        source = cell.source
        label = f"cell {number}"
        if KOREAN.search(source):
            errors.append(f"{label}: untranslated Korean source")
        if RETIRED_METHOD in source:
            errors.append(f"{label}: retired parameterization")
        if cell.cell_type == "markdown":
            # Code examples can discuss delimiters without being rendered math.
            prose = re.sub(r"```.*?```|~~~.*?~~~|`[^`]*`", "", source, flags=re.S)
            if re.search(r"\\[\[\]()\]]", prose):
                errors.append(f"{label}: use $...$ or $$...$$ math delimiters")
            if len(re.findall(r"(?<!\\)\$\$", prose)) % 2:
                errors.append(f"{label}: unbalanced $$ display math")
        if cell.cell_type != "code":
            continue
        if source.strip():
            nonempty_code += 1
        try:
            compile(
                transformer.transform_cell(source),
                f"{path}:cell_{number}",
                "exec",
                flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
            )
        except (SyntaxError, ValueError) as exc:
            errors.append(f"{label}: {exc}")
        if not check_execution:
            continue
        count = cell.get("execution_count")
        outputs = cell.get("outputs", [])
        has_execution |= count is not None or bool(outputs)
        if source.strip() and count is not None:
            counts.append(count)
        if outputs and count is None:
            errors.append(f"{label}: outputs retained without an execution count")
        for output in outputs:
            if output.output_type == "error":
                errors.append(f"{label}: saved error output: {output.get('ename', 'error')}")
            if (
                output.output_type == "execute_result"
                and output.get("execution_count") != count
            ):
                errors.append(f"{label}: output execution count differs from cell")
    if has_execution:
        if len(counts) != nonempty_code or any(a >= b for a, b in zip(counts, counts[1:])):
            errors.append("saved execution is partial or out of order; rerun from a fresh kernel")
        stamp = notebook.metadata.get(CHECK_METADATA, {})
        if stamp.get("source_sha256") != source_fingerprint(notebook):
            errors.append(
                "retained execution lacks a matching source fingerprint; clear outputs "
                "or use the checked copy produced by --execute"
            )
        if stamp.get("status") != "complete":
            errors.append("retained execution is not marked complete by --execute")
    return notebook, errors


def execute_smoke(path, notebook, output, timeout, run_tag):
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient

    notebook = copy.deepcopy(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    notebook.metadata.pop(CHECK_METADATA, None)
    environment = {
        **os.environ,
        "GMM_PRESET": "smoke",
        "GMM_DEVICE": "cpu",
        "GMM_RUN_TAG": run_tag,
        "CUDA_VISIBLE_DEVICES": "",
        "MPLBACKEND": "Agg",
    }
    stamp = {
        "source_sha256": source_fingerprint(notebook),
        "python_executable": sys.executable,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "run_tag": run_tag,
        "preset": "smoke",
        "device": "cpu",
        "status": "failed",
    }
    try:
        with TemporaryDirectory(prefix="fourier-notebook-kernel-") as temporary:
            kernel_name = "fourier-notebook-check"
            spec_dir = Path(temporary) / kernel_name
            spec_dir.mkdir()
            (spec_dir / "kernel.json").write_text(
                json.dumps({
                    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                    "display_name": "Current project Python",
                    "language": "python",
                }),
                encoding="utf-8",
            )
            client = NotebookClient(
                notebook, timeout=timeout, kernel_name=kernel_name,
                allow_errors=False, record_timing=True,
            )
            manager = client.create_kernel_manager()
            manager.kernel_spec_manager = KernelSpecManager(
                kernel_dirs=[temporary], ensure_native_kernel=False,
            )
            client.execute(cwd=str(ROOT), env=environment)
        stamp["status"] = "complete"
    finally:
        stamp["finished_utc"] = datetime.now(timezone.utc).isoformat()
        notebook.metadata[CHECK_METADATA] = stamp
        output.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, output)
    _, errors = validate_notebook(output)
    if errors:
        raise ValueError("Executed notebook failed validation: " + "; ".join(errors))
    return stamp


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", type=Path, help="Defaults to notebooks/*.ipynb")
    parser.add_argument("--execute", action="store_true", help="Run the CPU smoke preset in a fresh kernel")
    parser.add_argument("--timeout", type=int, default=600, help="Maximum seconds per code cell")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "saved/notebook_checks")
    args = parser.parse_args(argv)
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    paths = args.notebooks or sorted((ROOT / "notebooks").glob("*.ipynb"))
    if not paths:
        parser.error("No notebooks found")
    checked = []
    failed = False
    for path in paths:
        try:
            # --execute discards old outputs and proves a fresh, ordered run.
            notebook, errors = validate_notebook(path, check_execution=not args.execute)
        except Exception as exc:
            notebook, errors = None, [f"{type(exc).__name__}: {exc}"]
        checked.append((path, notebook))
        print(f"{'FAIL' if errors else 'PASS'} {path}", flush=True)
        for error in errors:
            print(f"  {error}", flush=True)
        failed |= bool(errors)
    if failed or not args.execute:
        return int(failed)
    check_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    directory = args.output_dir.resolve() / check_id
    directory.mkdir(parents=True, exist_ok=False)
    reports = []
    for index, (path, notebook) in enumerate(checked):
        destination = directory / f"{index + 1:02d}_{path.name}"
        run_tag = f"check_{check_id}_{index + 1}"
        print(f"EXEC {path} [CPU smoke; Python={sys.executable}]", flush=True)
        try:
            stamp = execute_smoke(path, notebook, destination, args.timeout, run_tag)
            reports.append({"notebook": str(path), "executed_copy": str(destination), **stamp})
            print(f"PASS execution: {destination}", flush=True)
        except Exception as exc:
            failed = True
            reports.append({"notebook": str(path), "executed_copy": str(destination), "status": "failed", "error": str(exc)})
            print(f"FAIL execution: {exc}\n  Partial notebook: {destination}", file=sys.stderr, flush=True)
    (directory / "report.json").write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {directory / 'report.json'}", flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
