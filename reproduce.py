#!/usr/bin/env python3
"""Safe entry point for the implicit deep inverse-control reproducibility package."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
from repro_support import compare_results, verify_inputs, write_json
from export_tables import export_tables


def fresh_output(text: str | None, command: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    path = Path(text).expanduser().resolve() if text else ROOT / 'runs' / f'{command}_{stamp}'
    # Always require a new directory. Existing files are never removed.
    path.mkdir(parents=True, exist_ok=False)
    return path


def run_logged(command: list[str], workspace: Path) -> int:
    env = os.environ.copy()
    env.update(MPLBACKEND='Agg', PYTHONHASHSEED='0', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1',
               MPLCONFIGDIR=str(workspace / '.mplconfig'))
    with (workspace / 'run.log').open('w', encoding='utf-8') as log:
        with subprocess.Popen(command, cwd=workspace, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding='utf-8', errors='replace', bufsize=1) as proc:
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                return proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                raise


def stage(workspace: Path, command: str) -> None:
    for dirname in ['code', 'scripts']:
        shutil.copytree(ROOT / dirname, workspace / dirname,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (workspace / 'results').mkdir()
    if command in ['checkpoints', 'audit']:
        for path in (ROOT / 'results').glob('*_models.pt'):
            shutil.copyfile(path, workspace / 'results' / path.name)
    if command == 'audit':
        shutil.copyfile(ROOT / 'results' / 'all_results.json', workspace / 'results' / 'all_results.json')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    subs.add_parser('check', help='Verify SHA-256 of frozen scientific inputs; standard library only.')
    subs.add_parser('test', help='Run unit and algebraic-equation tests.')
    for name in ['checkpoints', 'train', 'audit', 'tables']:
        sub = subs.add_parser(name, help={
            'checkpoints': 'Regenerate experiments from saved weights without training.',
            'train': 'Train from scratch and regenerate experiments in a new workspace.',
            'audit': 'Run independent equation and checkpoint certificate checks.',
            'tables': 'Export manuscript tables from JSON without ML dependencies.'}[name])
        sub.add_argument('--output', help='New output directory; existing directories are refused.')
        if name in ['checkpoints', 'train']:
            sub.add_argument('--experiment', choices=['all', 'static', 'narx', 'servo'], default='all')
        if name == 'tables':
            sub.add_argument('--results', type=Path, default=ROOT / 'results/all_results.json')
    args = parser.parse_args(argv)
    if args.command == 'check':
        result = verify_inputs(ROOT)
        print(json.dumps(result, indent=2))
        return 0 if result['status'] == 'passed' else 1
    if args.command == 'test':
        env = os.environ.copy()
        env.update(MPLBACKEND='Agg', PYTHONHASHSEED='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        return subprocess.call([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], cwd=ROOT, env=env)
    if args.command == 'tables':
        data = json.loads(args.results.read_text(encoding='utf-8'))
        out = fresh_output(args.output, args.command)
        export_tables(data, out)
        print(f'Tables and machine-readable exports: {out}')
        return 0
    integrity = verify_inputs(ROOT)
    if integrity['status'] != 'passed':
        print(json.dumps(integrity, indent=2), file=sys.stderr)
        print('Frozen inputs differ from the manifest. Review changes before running.', file=sys.stderr)
        return 1
    workspace = fresh_output(args.output, args.command)
    stage(workspace, args.command)
    write_json(workspace / 'run_manifest.json', {
        'status': 'running', 'mode': args.command,
        'experiment': getattr(args, 'experiment', None),
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'frozen_inputs': integrity, 'reference_manifest': json.loads((ROOT/'REFERENCE_MANIFEST.json').read_text())})
    if args.command == 'audit':
        command = [sys.executable, str(workspace / 'code/audit_light_revision.py')]
    else:
        command = [sys.executable, str(workspace / 'scripts/run_experiments.py'),
                   '--mode', args.command, '--experiment', args.experiment]
    print(f'Workspace: {workspace}', flush=True)
    start = time.perf_counter()
    exit_code = run_logged(command, workspace)
    comparison = None
    if exit_code == 0 and args.command in ['checkpoints', 'train']:
        reference = json.loads((ROOT / 'results/all_results.json').read_text(encoding='utf-8'))
        actual = json.loads((workspace / 'results/all_results.json').read_text(encoding='utf-8'))
        export_tables(actual, workspace / 'tables')
        comparison = compare_results(reference, actual)
        write_json(workspace / 'comparison.json', comparison)
        print(f"Reference comparison: {comparison['status']} "
              f"({comparison['scientific_values_passed']}/{comparison['scientific_values_checked']} values)")
        if args.command == 'checkpoints' and comparison['status'] != 'passed':
            exit_code = 1
    final_integrity = verify_inputs(ROOT)
    if final_integrity['status'] != 'passed':
        exit_code = 1
    manifest = json.loads((workspace / 'run_manifest.json').read_text())
    status = 'passed' if exit_code == 0 else 'failed'
    if exit_code == 0 and comparison is not None and comparison['status'] != 'passed':
        status = 'completed_with_reference_differences'
    manifest.update(status=status, execution_status='completed' if exit_code == 0 else 'failed', exit_code=exit_code,
                    elapsed_seconds=time.perf_counter()-start,
                    finished_utc=datetime.now(timezone.utc).isoformat(),
                    frozen_inputs_after_run=final_integrity,
                    comparison_status=None if comparison is None else comparison['status'])
    write_json(workspace / 'run_manifest.json', manifest)
    print(f"Run {manifest['status']}. Outputs: {workspace}")
    return exit_code


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except FileExistsError as exc:
        print(f'Output already exists: {exc.filename}. Choose a new --output directory.', file=sys.stderr)
        raise SystemExit(2)
    except (OSError, ValueError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(2)
    except KeyboardInterrupt:
        print('\nInterrupted; partial outputs were retained.', file=sys.stderr)
        raise SystemExit(130)
