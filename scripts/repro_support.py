"""Standard-library utilities for result export and numerical comparison."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterator


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def json_safe(value: Any) -> Any:
    """Convert unavailable nonfinite values to JSON null, recursively."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + '\n', encoding='utf-8')


def numeric_leaves(value: Any, prefix: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from numeric_leaves(item, prefix + (str(key),))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            yield from numeric_leaves(item, prefix + (str(idx),))
    elif value is None or isinstance(value, (float, int)) and not isinstance(value, bool):
        yield prefix, value


def export_metrics(results: dict, path: Path) -> None:
    """RFC-style quoted CSV; commas in variant names remain inside one field."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['experiment', 'metric_path', 'value'])
        for key, val in numeric_leaves(results):
            if val is None or isinstance(val, float) and not math.isfinite(val):
                val = ''
            writer.writerow([key[0], '/'.join(key[1:]), val])


def compare_results(reference: dict, actual: dict, *, rtol: float = 1e-4,
                    atol: float = 3e-7) -> dict:
    """Compare science outputs; record timing and iteration counts separately.

    A single-experiment run is compared only for that experiment. Missing
    expected scientific fields and unexpected nonfinite values are failures.
    Float32 stopping diagnostics are recorded but are not regression gates.
    """
    expected = dict(numeric_leaves({k: reference[k] for k in actual if k in reference}))
    observed = dict(numeric_leaves(actual))
    checked, passed, diagnostic = 0, 0, []
    failures, missing, changes = [], [], []
    for key, old in expected.items():
        label = '/'.join(key)
        if key not in observed:
            missing.append(label)
            continue
        new = observed[key]
        is_diagnostic = any('time' in k or 'iteration' in k or 'hidden_evaluation' in k
                            or k == 'fraction_negative' for k in key)
        old_unavailable = old is None or isinstance(old, float) and not math.isfinite(old)
        new_unavailable = new is None or isinstance(new, float) and not math.isfinite(new)
        if old_unavailable or new_unavailable:
            item = {'path': label, 'reference': json_safe(old), 'recomputed': json_safe(new)}
            if is_diagnostic:
                diagnostic.append(item)
            else:
                checked += 1
                if old_unavailable == new_unavailable:
                    passed += 1
                else:
                    failures.append(item)
            continue
        difference = abs(new - old)
        item = {'path': label, 'reference': old, 'recomputed': new,
                'absolute_difference': difference, 'tolerance': atol + rtol * abs(old)}
        if is_diagnostic:
            diagnostic.append(item)
            continue
        checked += 1
        if difference <= atol + rtol * abs(old):
            passed += 1
        else:
            failures.append(item)
        if difference:
            changes.append(item)
    return {
        'status': 'passed' if not failures and not missing else 'differences_detected',
        'experiments': list(actual), 'rtol': rtol, 'atol': atol,
        'scientific_values_checked': checked, 'scientific_values_passed': passed,
        'scientific_values_failed': len(failures), 'missing_reference_fields': missing,
        'failures': failures,
        'largest_absolute_differences': sorted(changes, key=lambda x: x['absolute_difference'], reverse=True)[:30],
        'diagnostic_values_excluded_from_gate': len(diagnostic),
        'diagnostics': diagnostic,
        'policy': 'Timing, iteration/hidden-evaluation counts, and fraction_negative are diagnostics. '
                  'RMSEs, certificate matrices, tubes, and calibration outputs are numerical regression targets. '
                  'Retraining comparisons are descriptive; checkpoint comparisons gate command success.'}


def verify_inputs(root: Path) -> dict:
    manifest = json.loads((root / 'REFERENCE_MANIFEST.json').read_text(encoding='utf-8'))
    failures = []
    for entry in manifest['files']:
        path = root / entry['path']
        if not path.is_file():
            failures.append({'path': entry['path'], 'reason': 'missing'})
        elif sha256(path) != entry['sha256']:
            failures.append({'path': entry['path'], 'reason': 'SHA-256 mismatch'})
    return {'status': 'passed' if not failures else 'failed',
            'files_checked': len(manifest['files']), 'failures': failures}
