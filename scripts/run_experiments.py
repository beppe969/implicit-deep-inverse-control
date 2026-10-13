"""Worker for isolated experiment runs. Invoke through reproduce.py.

The released numerical functions are imported directly. In checkpoint mode,
only the optimizer calls are replaced: saved weights are restored and the
same minibatch RNG draws are consumed. Data generation and evaluation run in
full, including the diagnostic cyclic-forward Newton comparison.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import torch
import matplotlib
import experiments as ex
from repro_support import export_metrics, write_json


class CheckpointRestorer:
    """Restore models with strict state-dict checks and exact minibatch replay."""

    def __init__(self, experiment: str):
        self.experiment = experiment
        self.weights = torch.load(ex.RES / f'{experiment}_models.pt',
                                  weights_only=True, map_location='cpu')
        self.original_forward = ex.fit_forward
        self.original_direct = ex.fit_mlp
        self.events: list[dict] = []

    def _replay(self, fn, model, args, kwargs) -> None:
        bound = inspect.signature(fn).bind(model, *args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments
        data = a['chi'] if fn is self.original_forward else a['X']
        count = len(data)
        for _ in range(a['steps']):
            torch.randint(0, count, (min(a['batch'], count),))

    def forward(self, model, *args, **kwargs) -> None:
        if self.experiment == 'static':
            key = 'model'
        elif self.experiment == 'servo':
            key = 'model' if model.input_features else 'control_affine'
        else:
            weight = kwargs.get('sensitivity_weight', 0.)
            if model.kappa_f > 0:
                name = 'implicit forward (A!=0)'
            elif not model.input_features:
                name = ('control-affine (B_u=0), lambda_s=3e-3' if weight == 3e-3
                        else 'control-affine (B_u=0)')
            elif weight == 0:
                name = 'lambda_s=0'
            elif weight == 1e-3:
                name = 'lambda_s=1e-3'
            elif weight == 1e-4:
                name = 'main'
            else:
                raise ValueError(f'Unrecognized NARX sensitivity weight: {weight}')
            key = 'variant:' + name
        model.load_state_dict(self.weights[key], strict=True)
        self._replay(self.original_forward, model, args, kwargs)
        model.eval()
        self.events.append({'role': 'forward', 'state_dict': key,
                            'minibatch_draws_replayed': kwargs.get('steps', 1800)})
        print(f'      restored {key}; minibatch RNG replayed', flush=True)

    def direct(self, model, *args, **kwargs) -> None:
        model.load_state_dict(self.weights['inverse'], strict=True)
        self._replay(self.original_direct, model, args, kwargs)
        model.eval()
        self.events.append({'role': 'direct inverse', 'state_dict': 'inverse',
                            'minibatch_draws_replayed': kwargs.get('steps', 1800)})

    def __enter__(self):
        ex.fit_forward, ex.fit_mlp = self.forward, self.direct
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        ex.fit_forward, ex.fit_mlp = self.original_forward, self.original_direct


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=['checkpoints', 'train'])
    parser.add_argument('--experiment', default='all', choices=['all', 'static', 'narx', 'servo'])
    args = parser.parse_args()
    which = ['static', 'narx', 'servo'] if args.experiment == 'all' else [args.experiment]
    result, timings, restored = {}, {}, {}
    for name in which:
        print(f'[{args.mode}] {name}', flush=True)
        t0 = time.perf_counter()
        if args.mode == 'checkpoints':
            with CheckpointRestorer(name) as loader:
                result[name] = getattr(ex, f'run_{name}')()
            restored[name] = loader.events
        else:
            result[name] = getattr(ex, f'run_{name}')()
        timings[name] = time.perf_counter() - t0
        write_json(ex.RES / 'all_results.json', result)
        ex.write_summary(result)
        export_metrics(result, ex.RES / 'metrics.csv')
        print(f'[{name}] completed in {timings[name]:.2f} s', flush=True)
    write_json(ROOT / 'environment.json', {
        'python': platform.python_version(), 'platform': platform.platform(),
        'machine': platform.machine(), 'processor': platform.processor(),
        'torch': torch.__version__, 'numpy': np.__version__,
        'matplotlib': matplotlib.__version__, 'torch_threads': torch.get_num_threads(),
        'device': 'cpu', 'mode': args.mode, 'experiments': which,
        'seconds_by_experiment': timings, 'restored_checkpoints': restored,
        'cuda_used': False})


if __name__ == '__main__':
    main()
