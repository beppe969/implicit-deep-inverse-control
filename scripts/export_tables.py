"""Export the four manuscript tables from a results JSON file.

No third-party dependencies are required. Tables use the manuscript's displayed
rounding; metrics.csv retains all scalar numeric results at full precision.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from repro_support import export_metrics, write_json


def fmt(value, digits: int = 4, scale: float = 1.) -> str:
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return '--'
    return f'{value * scale:.{digits}f}'


def tex_escape(text: str) -> str:
    replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%',
                    '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
                    '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(replacements.get(char, char) for char in text)


def write_table(folder: Path, name: str, title: str, headers: list[str], rows: list[list[str]], notes: str) -> str:
    with (folder / (name + '.csv')).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)
    lines = ['## ' + title, '', '| ' + ' | '.join(headers) + ' |',
             '| ' + ' | '.join('---' for _ in headers) + ' |']
    lines += ['| ' + ' | '.join(row) + ' |' for row in rows]
    lines += ['', notes, '']
    text = '\n'.join(lines)
    (folder / (name + '.md')).write_text(text, encoding='utf-8')
    tex = ['% Generated from all_results.json. Requires the booktabs package.',
           '% ' + title, '% ' + notes.replace('\n', ' '),
           r'\begin{tabular}{l' + 'r' * (len(headers)-1) + '}', r'\toprule',
           ' & '.join(tex_escape(h) for h in headers) + r' \\', r'\midrule']
    tex += [' & '.join(tex_escape(cell) for cell in row) + r' \\' for row in rows]
    tex += [r'\bottomrule', r'\end{tabular}', '']
    (folder / (name + '.tex')).write_text('\n'.join(tex), encoding='utf-8')
    return text


def export_tables(results: dict, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    combined = ['# Numerical tables', '', 'Values are rounded as in the manuscript. '
                'See metrics.csv for full-precision scalar results.', '']
    if 'static' in results:
        s = results['static']
        rows = [
            ['Inverse RMSE', fmt(s['inverse_rmse_algebraic']), fmt(s['inverse_rmse_newton']), fmt(s['inverse_rmse_direct'])],
            ['True-map cycle RMSE', fmt(s['true_cycle_rmse_algebraic']), fmt(s['true_cycle_rmse_algebraic']), fmt(s['true_cycle_rmse_direct'])],
            ['Noisy-target inverse RMSE', fmt(s['noisy_inverse_rmse_algebraic']), fmt(s['noisy_inverse_rmse_algebraic']), fmt(s['noisy_inverse_rmse_direct'])],
            ['Hidden-layer evaluations (mean)', fmt(s['mean_iterations'], 2), fmt(s['newton_mean_hidden_evaluations'], 2), '--'],
            ['Iterations (mean / max)', f"{s['mean_iterations']:.2f} / {s['max_iterations']:.0f}",
             f"{s['newton_mean_iterations']:.2f} / {s['newton_max_iterations']:.0f}", '--']]
        notes = ('Newton true-map cycle and noisy-target cells follow the manuscript\'s same-root convention: '
                 'the archived JSON stores these two metrics for the algebraic inverse only. '
                 'The independently stored Newton input-recovery RMSE verifies the clean-target root comparison. '
                 'Iteration counts can vary at the float32 stopping threshold.')
        combined.append(write_table(folder, 'table_I_static', 'Table I. Static MIMO inversion',
                                    ['Metric', 'Algebraic', 'Newton', 'Direct'], rows, notes))
    if 'narx' in results:
        n = results['narx']
        names = [('main', 'Main (lambda_s = 1e-4)'), ('lambda_s=0', 'lambda_s = 0'),
                 ('lambda_s=1e-3', 'lambda_s = 1e-3'),
                 ('control-affine (B_u=0)', 'B_u = 0, lambda_s = 1e-4'),
                 ('control-affine (B_u=0), lambda_s=3e-3', 'B_u = 0, lambda_s = 3e-3'),
                 ('implicit forward (A!=0)', 'A != 0, lambda_s = 1e-4')]
        rows = []
        for key, label in names:
            v = n['variants'][key]
            tube = v.get('u_tube')
            largest = max(tube) if tube and all(x is not None and math.isfinite(x) for x in tube) else None
            rows.append([label, fmt(v['forward_rmse'], 2, 1000), fmt(v['inverse_rmse_algebraic'], 2, 1000),
                         fmt(v['inverse_contraction_inf'], 2), fmt(v['rho_H'], 3), fmt(largest, 2),
                         fmt(v['closed_loop']['rmse_mean'], 2, 1000)])
        for label, key, field in [('Direct inverse', 'direct inverse', 'inverse_rmse_direct'),
                                  ('Linear ARX inverse', 'linear ARX inverse', 'inverse_rmse_linear')]:
            rows.append([label, '--', fmt(n[field], 2, 1000), '--', '--', '--',
                         fmt(n['monte_carlo'][key]['rmse_mean'], 2, 1000)])
        combined.append(write_table(folder, 'table_II_narx_models', 'Table II. NARX models and certificates',
                    ['Model', 'Forward RMSE x 1e3', 'Inverse RMSE x 1e3', 'Inverse contraction', 'rho(H)',
                     'Max. input-tube radius', 'CL RMSE x 1e3'], rows,
                    'Only the main model uses fresh matched closed-loop calibration. '
                    'Other tube values are conditional calculations from preliminary open-loop envelopes. '
                    'An unavailable tube is shown as --.'))
        rows = []
        for method in ['implicit inverse', 'online Newton', 'direct inverse', 'linear ARX inverse']:
            m = n['monte_carlo'][method]
            rows.append([method, fmt(m['rmse_mean'], 5), fmt(m['rmse_p95'], 5),
                         fmt(m['max_error_mean']), fmt(m['max_error_p95']),
                         fmt(m.get('hidden_evaluations_mean'), 1)])
        combined.append(write_table(folder, 'table_III_narx_tracking', 'Table III. Tracking over 120 uncertain NARX plants',
                    ['Controller', 'Mean RMSE', '95th-percentile RMSE', 'Mean max. error', '95th-percentile max. error',
                     'Hidden evaluations'], rows,
                    'Measured feedback is noisy; tracking statistics use noise-free plant outputs. '
                    'The theorem\'s output tube applies to measured output errors.'))
    if 'servo' in results:
        rows = []
        for method in ['implicit inverse', 'control-affine inverse', 'direct inverse', 'nominal computed torque']:
            m = results['servo']['monte_carlo'][method]
            rows.append([method, fmt(m['position_rmse_mean']), fmt(m['position_rmse_p95']), fmt(m['max_position_error_mean'])])
        combined.append(write_table(folder, 'table_IV_servo', 'Table IV. Servo tracking over 80 uncertain plants',
                     ['Controller', 'Mean RMSE', '95th-percentile RMSE', 'Mean max. error'], rows,
                     'Position errors are in radians. Samples are aligned at (k+1)h. '
                     'This state-informed example is an empirical tracking study.'))
    (folder / 'TABLES.md').write_text('\n'.join(combined), encoding='utf-8')
    export_metrics(results, folder / 'metrics.csv')
    write_json(folder / 'results_portable.json', results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory to preserve existing exports.')
    export_tables(json.loads(args.results.read_text(encoding='utf-8')), args.output)
