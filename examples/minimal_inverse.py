"""Load the main NARX checkpoint, invert a requested output, and check the cycle.

Run from the repository root: python examples/minimal_inverse.py
"""
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
os.environ.setdefault('MPLBACKEND', 'Agg')
import torch
import experiments as ex


def main() -> None:
    ck = torch.load(ROOT / 'results/narx_models.pt', weights_only=True, map_location='cpu')
    state = ck['model']
    model = ex.AlgebraicImplicitMap(10, 2, 48, state['D_u'], state['D_chi'], kappa_i=.68)
    model.load_state_dict(state, strict=True)
    model.eval()
    history = torch.zeros(1, 10)
    requested_output = torch.tensor([[.1, -.1]])
    command, info = model.inverse(history, requested_output, tol=2e-7, return_info=True)
    with torch.no_grad():
        predicted_output = model(history, command)
    print(json.dumps({'history': history.tolist(), 'requested_output': requested_output.tolist(),
                      'computed_input': command.tolist(), 'predicted_output': predicted_output.tolist(),
                      'max_cycle_error': float(torch.max(torch.abs(predicted_output-requested_output))),
                      'hidden_residual': info['residual'].tolist(),
                      'iterations': info['iterations'].tolist()}, indent=2))


if __name__ == '__main__':
    main()
