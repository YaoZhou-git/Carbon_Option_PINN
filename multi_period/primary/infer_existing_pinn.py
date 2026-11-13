"""Read-only N4 PINN inference; no training and no new reference integration.

Run from the package root with: python primary/infer_existing_pinn.py.
The architecture below copies factor_pinn.py's forward map so this script does
not import a training entry point. Only the named existing checkpoint is loaded.
The saved reference values/curves are read as data; no quadrature or QMC is run.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE
OUT = PACKAGE / 'outputs' / 'primary'
CODE = PACKAGE / 'core'
CHECKPOINT = PACKAGE / 'models' / 'baseline' / 'N4_factor_pinn.pt'
SOURCE = PACKAGE / 'reference'
SCAN = SOURCE / 'data' / 'future_quote_scans.csv'
RISK = SOURCE / 'data' / 'risk_decomposition.csv'
MATCHED = SOURCE / 'data' / 'matched_initial_variance.csv'
torch.set_num_threads(4)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def numpy_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Cannot serialize {type(value).__name__}')


def phi(x):
    return torch.exp(-x.square() / 2) / np.sqrt(2 * np.pi)


def Phi(x):
    return .5 * (1 + torch.erf(x / np.sqrt(2)))


class FactorPINN(nn.Module):
    """Identical architecture/forward to the supplied trained N4 model."""
    def __init__(self, cfg, n):
        super().__init__()
        self.cfg, self.n = cfg, n
        weights = np.array([np.prod(cfg.kappa[:j]) for j in range(n)])
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
        corr = np.full((n, n), cfg.rho)
        np.fill_diagonal(corr, 1)
        self.register_buffer('R', torch.tensor(corr, dtype=torch.float32))
        for name in ['T', 'alpha', 'beta']:
            self.register_buffer(name, torch.tensor(getattr(cfg, name)[:n], dtype=torch.float32))
        layers = [nn.Linear(n + 1, 64), nn.Tanh()]
        for _ in range(2):
            layers.extend([nn.Linear(64, 64), nn.Tanh()])
        layers.append(nn.Linear(64, 1))
        self.net = nn.Sequential(*layers)

    def diffusion(self, state):
        y, t = state[:, :self.n], state[:, self.n:]
        xi = np.sqrt(2) * torch.erfinv(2 * y.clamp(1e-6, 1 - 1e-6) - 1)
        return phi(xi) * torch.sqrt(self.beta * (self.T - t) ** (-self.alpha))

    def forward(self, state):
        y, t = state[:, :self.n], state[:, self.n:]
        h = self.cfg.tau - t
        d = (y @ self.weights)[:, None] - self.cfg.K / self.cfg.pi
        sig = self.diffusion(state) * self.weights
        v = (sig * (sig @ self.R)).sum(1, keepdim=True) * h
        sd = torch.sqrt(v.clamp_min(0) + 1e-10)
        z = d / sd
        baseline = d * Phi(z) + sd * phi(z)
        inp = torch.cat([2 * y - 1, 2 * t / self.cfg.tau - 1], 1)
        trial = torch.exp(-self.cfg.r * h) * (baseline + h * .1 * self.net(inp))
        return torch.where(h == 0, d.relu(), trial)


def normalized_factors(quotes, cfg):
    values = np.asarray(quotes, dtype=float).copy()
    values[:, :-1] -= np.asarray(cfg.kappa) * values[:, 1:]
    if not np.all((values > 0) & (values < cfg.pi)):
        raise ValueError('Invalid factors: no clipping is permitted.')
    return values / cfg.pi


def price_and_gradients(net, quotes, cfg):
    """Autograd in bounded factors plus exact quote-chain conversion.

    C = pi * u(y,t), y_i=(A_i-kappa_i A_(i+1))/pi.
    C_Xi = u_yi; C_A1=C_X1;
    C_Aj=C_Xj-kappa_(j-1)*C_X(j-1), j=2,...,N.
    """
    y = normalized_factors(quotes, cfg)
    states = torch.tensor(np.c_[y, np.zeros(len(y))], dtype=torch.float32,
                          requires_grad=True)
    prices = cfg.pi * net(states)
    dC_dy = torch.autograd.grad(prices.sum(), states)[0][:, :4]
    # Keep network/autograd float32; use float64 for reported differences and
    # the deterministic coordinate conversion to avoid NumPy scalar rounding.
    factor_delta = dC_dy.detach().numpy().astype(np.float64) / cfg.pi
    jacobian = np.eye(4)
    for j in range(3):
        jacobian[j, j + 1] = -cfg.kappa[j]
    quote_delta = factor_delta @ jacobian
    return prices.detach().numpy().ravel().astype(np.float64), factor_delta, quote_delta, y


def existing_checkpoint_inventory():
    """Filename/config existence check, never deserializes legacy models."""
    paths = sorted(str(p.relative_to(ROOT)) for p in (ROOT / 'models').rglob('*')
                   if p.is_file() and p.suffix in {'.pt', '.pth'})
    relevant = [p for p in paths if 'N4' in Path(p).name or 'factor_pinn' in Path(p).name]
    configs = []
    for path in relevant:
        candidate = ROOT / path
        config = candidate.parent.parent / 'comparison_config.json'
        if config.exists():
            data = json.loads(config.read_text(encoding='utf-8-sig'))
            configs.append({'checkpoint': path, 'config_path': str(config.relative_to(ROOT)),
                            'beta': data.get('beta'), 'T': data.get('T'),
                            'alpha': data.get('alpha'), 'kappa': data.get('kappa'),
                            'rho': data.get('rho')})
    tokens = ['1.9774', '3.4588', '1.58193', '2.76704']
    hits = [p for p in paths if any(t in p for t in tokens)]
    return {'method': 'Packaged model filename inventory; no additional model loads',
            'checkpoint_files_count': len(paths), 'relevant_paths': relevant,
            'adjacent_N4_configs': configs, 'matched_beta_filename_hits': hits,
            'conclusion': 'Inventory is limited to the packaged baseline and matched models; historical workspace inventory is retained only in saved provenance.'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    before_hash = sha256(CHECKPOINT)
    load_start = time.perf_counter()
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    cfg = SimpleNamespace(**ckpt['parameters'])
    assert ckpt['N'] == 4 and cfg.tau == 3
    net = FactorPINN(cfg, 4)
    net.load_state_dict(ckpt['state_dict'], strict=True)
    net.eval()
    load_seconds = time.perf_counter() - load_start
    curves = np.array([[60., 45., 30., 20.], [60., 60., 60., 60.], [60., 65., 70., 75.]])
    names = ['downward', 'flat', 'upward']
    sobol_rounded = [16.3434, 12.4709, 10.6120]
    scans = read_csv(SCAN)
    scan_quotes = np.tile(curves[0], (len(scans), 1))
    for i, row in enumerate(scans):
        scan_quotes[i, int(row['period']) - 1] = float(row['quote'])
    all_quotes = np.r_[curves, scan_quotes]
    y = normalized_factors(all_quotes, cfg)
    states = torch.tensor(np.c_[y, np.zeros(len(y))], dtype=torch.float32)
    # One small warmup and one single existing-point batch measurement only.
    with torch.no_grad():
        net(states[:3])
        timer = time.perf_counter()
        timed_prices = (cfg.pi * net(states)).numpy().ravel()
        inference_seconds = time.perf_counter() - timer
    timer = time.perf_counter()
    prices, factors, deltas, y = price_and_gradients(net, all_quotes, cfg)
    autograd_seconds = time.perf_counter() - timer
    assert np.allclose(prices, timed_prices, atol=2e-5)
    # Independent direct autograd through quote->factor map verifies the chain.
    quote_input = torch.tensor(curves[:1], dtype=torch.float32, requires_grad=True)
    kappas = torch.tensor(cfg.kappa, dtype=torch.float32)
    factor_input = torch.cat([quote_input[:, :-1] - kappas * quote_input[:, 1:],
                              quote_input[:, -1:]], dim=1) / cfg.pi
    direct_state = torch.cat([factor_input, torch.zeros((1, 1))], dim=1)
    direct_delta = torch.autograd.grad((cfg.pi * net(direct_state)).sum(), quote_input)[0].detach().numpy()[0]
    chain_error = float(np.max(np.abs(direct_delta - deltas[0])))
    assert chain_error < 1e-6
    curve_rows = []
    for i, name in enumerate(names):
        row = {'curve': name, **{f'A{j+1}': curves[i, j] for j in range(4)},
               'PINN_C': prices[i], 'saved_Sobol_C_rounded': sobol_rounded[i],
               'price_error_vs_rounded_reference': prices[i] - sobol_rounded[i],
               'relative_price_error_percent': 100 * (prices[i] / sobol_rounded[i] - 1)}
        row.update({f'y{j+1}': y[i, j] for j in range(4)})
        row.update({f'dC_dX{j+1}': factors[i, j] for j in range(4)})
        row.update({f'dC_dA{j+1}': deltas[i, j] for j in range(4)})
        curve_rows.append(row)
    write_csv('three_curve_prices_and_gradients.csv', curve_rows)
    saved_risk = read_csv(RISK)
    delta_rows = []
    for j, saved in enumerate(saved_risk):
        actual_ref = float(saved['delta_A'])
        factor_positive = factors[0, j]
        offset = 0. if j == 0 else -cfg.kappa[j-1] * factors[0, j-1]
        delta_rows.append({'period': j+1, 'PINN_dC_dX': factor_positive,
                           'PINN_adjacent_offset': offset, 'PINN_dC_dA': deltas[0, j],
                           'saved_Sobol_dC_dA': actual_ref,
                           'error_vs_saved_Sobol': deltas[0, j] - actual_ref,
                           'relative_error_percent': 100 * abs(deltas[0, j] - actual_ref) / abs(actual_ref),
                           'saved_Sobol_delta_halfwidth95': float(saved['delta_halfwidth95']),
                           'saved_Sobol_dC_dX': float(saved['delta_X']),
                           'saved_Sobol_adjacent_offset': float(saved['adjacent_offset'])})
    write_csv('downward_delta_and_factor_decomposition.csv', delta_rows)
    scan_rows = []
    for i, source_row in enumerate(scans):
        k = i + 3
        row = {**source_row, **{f'A{j+1}': scan_quotes[i, j] for j in range(4)},
               'PINN_C': prices[k], 'PINN_minus_saved_C': prices[k] - float(source_row['C'])}
        row.update({f'y{j+1}': y[k, j] for j in range(4)})
        row.update({f'PINN_dC_dX{j+1}': factors[k, j] for j in range(4)})
        row.update({f'PINN_dC_dA{j+1}': deltas[k, j] for j in range(4)})
        scan_rows.append(row)
    write_csv('future_quote_scans_pinn_exact_saved_points.csv', scan_rows)
    secants, scan_summary = [], []
    for period in [2, 3, 4]:
        rows = [row for row in scan_rows if int(row['period']) == period]
        for left, right in zip(rows[:-1], rows[1:]):
            step = float(right['quote']) - float(left['quote'])
            ref = (float(right['C']) - float(left['C'])) / step
            pred = (right['PINN_C'] - left['PINN_C']) / step
            secants.append({'period': period, 'quote_left': left['quote'], 'quote_right': right['quote'],
                            'saved_reference_secant': ref, 'PINN_secant': pred,
                            'secant_error': pred - ref, 'same_sign': bool(ref * pred > 0),
                            'reference_abs_lt_001': bool(abs(ref) < .01)})
        sub = [row for row in secants if row['period'] == period]
        errors = np.array([row['PINN_minus_saved_C'] for row in rows])
        scan_summary.append({'period': period, 'point_count': len(rows),
                             'price_MAE': float(np.mean(abs(errors))),
                             'price_max_abs_error': float(np.max(abs(errors))),
                             'secant_max_abs_error': max(abs(row['secant_error']) for row in sub),
                             'secant_sign_disagreement_count': sum(not row['same_sign'] for row in sub),
                             'small_reference_secant_count_abs_lt_001': sum(row['reference_abs_lt_001'] for row in sub),
                             'PINN_autograd_delta_min': min(row[f'PINN_dC_dA{period}'] for row in rows),
                             'PINN_autograd_delta_max': max(row[f'PINN_dC_dA{period}'] for row in rows)})
    write_csv('future_quote_scan_secants.csv', secants)
    write_csv('future_quote_scan_summary.csv', scan_summary)
    errors = np.array([row['error_vs_saved_Sobol'] for row in delta_rows])
    signs_ok = bool(np.all(deltas[0] * np.array([float(row['delta_A']) for row in saved_risk]) > 0))
    last_two_relative = [row['relative_error_percent'] / 100 for row in delta_rows[-2:]]
    acceptable = bool(np.max(abs(errors)) < .002 and signs_ok and max(last_two_relative) < .05)
    metadata = {'run_started_UTC': started, 'run_finished_UTC': datetime.now(timezone.utc).isoformat(),
                'purpose': 'Inference-only audit of existing N4; no training, no new reference integration.',
                'checkpoint': str(CHECKPOINT), 'checkpoint_sha256': before_hash,
                'checkpoint_sha256_after': sha256(CHECKPOINT),
                'checkpoint_metadata': {k: v for k, v in ckpt.items() if k != 'state_dict'},
                'source_files_sha256': {str(p): sha256(p) for p in [CODE/'factor_pinn.py', CODE/'factor_model.py', SCAN, RISK, MATCHED]},
                'environment': {'python': sys.version, 'executable': sys.executable,
                                'torch': torch.__version__, 'numpy': np.__version__,
                                'platform': platform.platform(), 'device': 'cpu', 'dtype': 'float32', 'torch_threads': 4},
                'timing': {'checkpoint_load_and_model_setup_seconds': load_seconds,
                           'warmup_points': 3, 'timed_forward_batch_points': len(all_quotes),
                           'single_forward_batch_seconds': inference_seconds,
                           'single_forward_per_point_seconds': inference_seconds / len(all_quotes),
                           'single_autograd_batch_seconds': autograd_seconds,
                           'interpretation': 'Single small CPU batch after one warmup; not a throughput benchmark.'},
                'coordinate_chain': 'C_Xi = C_yi/pi; C_A1=C_X1; C_Aj=C_Xj-kappa_(j-1)*C_X(j-1).',
                'chain_vs_direct_quote_autograd_max_abs_difference': chain_error,
                'coverage': {'all_evaluated_y_min': float(y.min()), 'all_evaluated_y_max': float(y.max()),
                             'all_evaluated_inside_random_factor_interior': bool(np.all((y >= .002) & (y <= .998))),
                             'three_curves_y': y[:3].tolist(),
                             'training_random_y_range': [.002, .998],
                             'training_face_values': [.0001, .9999],
                             'training_t_range': '[0, .999*tau); exact t=0 is not guaranteed sampled',
                             'interpretation': 'Within declared factor-domain support, not proof of realized point density or accuracy.'},
                'baseline_delta_acceptance': {'max_abs_error_threshold': .002,
                                               'all_four_signs_agree': signs_ok,
                                               'last_two_relative_error_threshold': .05,
                                               'last_two_relative_errors': last_two_relative,
                                               'max_abs_error': float(np.max(abs(errors))),
                                               'passes_requested_criterion': acceptable},
                'scan_summary': scan_summary,
                'beta_checkpoint_existence': existing_checkpoint_inventory(),
                'no_training': True, 'no_new_quadrature_or_Sobol': True}
    assert metadata['checkpoint_sha256'] == metadata['checkpoint_sha256_after']
    (OUT/'inference_results.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=numpy_scalar), encoding='utf-8')
    print(json.dumps({'prices': curve_rows, 'downward_deltas': delta_rows,
                      'scan_summary': scan_summary, 'timing': metadata['timing'],
                      'baseline_pass': acceptable}, ensure_ascii=False, indent=2, default=numpy_scalar))


if __name__ == '__main__':
    main()
