"""Separate, recorded-flight replication; frozen synthetic inference is untouched."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from rgpwm.data import WindowBatch, corrupt_observations
from rgpwm.training import TrainConfig, train_model, seed_everything, windows_to_loader
from external_validity.data import load_records, fit_normalizer
from external_validity.physics import KinematicThrustPrior

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/external/nanodrone'
OUT = ROOT / 'results/external_nanodrone'
METHODS = ('cfr', 'unfiltered', 'recurrent')
SEEDS = (131, 149, 167, 181, 199)
REGIMES = {'recorded': (0., 0.), 'noise': (.1, 0.),
           'missing': (0., .3), 'combined': (.1, .3)}
FIELDS = ('context_values', 'context_mask', 'context_delta', 'context_actions',
          'future_actions', 'targets')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def provenance():
    paths = [*(ROOT / 'code/external_validity' / name for name in
               ('__init__.py', 'data.py', 'physics.py', 'run.py')),
             ROOT / 'code/rgpwm/training.py', ROOT / 'code/rgpwm/data.py',
             ROOT / 'code/rgpwm/metrics.py',
             ROOT / 'plans/26_external_validity_revision.md',
             DATA / 'download_manifest.json',
             DATA / 'source_metadata/README.md', DATA / 'source_metadata/repository_tree.json',
             *sorted(DATA.glob('**/*.csv'))]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}


def make_windows(records, norm, horizon=25, stride=10, noise=0., missing=0., seed=601):
    """u[t] advances x[t] to x[t+1]; context actions arrive at each context state."""
    arrays = {key: [] for key in FIELDS}
    ids, origins = [], []
    for record in records:
        states = (record.states - norm.state_mean) / norm.state_std
        actions = (record.actions - norm.action_mean) / norm.action_std
        # Stable record-specific seed; same perturbation draw across methods/seeds.
        record_seed = int(hashlib.sha256(f'{seed}:{record.record_id}'.encode()).hexdigest()[:8], 16)
        obs = corrupt_observations(states, noise, missing, record_seed)
        for anchor in range(8, len(states) - horizon + 1, stride):
            start = anchor - 8
            ca = np.zeros((8, 4))
            ca[1:] = actions[start:anchor-1]
            arrays['context_values'].append(obs.values[start:anchor])
            arrays['context_mask'].append(obs.mask[start:anchor])
            arrays['context_delta'].append(obs.delta[start:anchor])
            arrays['context_actions'].append(ca)
            arrays['future_actions'].append(actions[anchor-1:anchor+horizon-1])
            arrays['targets'].append(states[anchor:anchor+horizon])
            ids.append(record.record_id)
            origins.append(anchor-1)
    if not ids:
        raise ValueError('no complete forecast windows')
    windows = WindowBatch(**{k: np.asarray(v, dtype=np.float32) for k, v in arrays.items()},
                          trajectory_ids=np.asarray(ids))
    return windows, np.asarray(origins)


def mixed_windows(records, norm):
    clean, _ = make_windows(records, norm)
    noisy, _ = make_windows(records, norm, noise=.05, missing=.1)
    return WindowBatch(**{k: np.concatenate([getattr(clean, k), getattr(noisy, k)]) for k in FIELDS},
                       trajectory_ids=np.concatenate([clean.trajectory_ids, noisy.trajectory_ids]))


class ForecastModel(nn.Module):
    """Identical learned capacity; CFR alone filters the forecast origin."""
    def __init__(self, method, norm, prior, width=64):
        super().__init__()
        if method not in (*METHODS, 'nominal'):
            raise ValueError(method)
        self.method = method
        self.prior = copy.deepcopy(prior)
        for key in ('state_mean', 'state_std', 'action_mean', 'action_std'):
            self.register_buffer(key, torch.tensor(getattr(norm, key), dtype=torch.float32))
        self.encoder = nn.GRU(40, width, batch_first=True)
        self.cell = nn.GRUCell(16, width)
        self.head = nn.Sequential(nn.Linear(width+16, width), nn.SiLU(),
                                  nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 24))

    def step(self, state, action):
        physical = self.prior(state * self.state_std + self.state_mean,
                              action * self.action_std + self.action_mean)
        return (physical - self.state_mean) / self.state_std

    def forward(self, values, mask, delta, context_actions, future_actions):
        _, hidden = self.encoder(torch.cat((values, mask, torch.log1p(delta), context_actions), -1))
        hidden = hidden[-1]
        state = values[:, -1]
        if self.method == 'cfr':
            state = values[:, 0]
            for t in range(1, values.shape[1]):
                prediction = self.step(state, context_actions[:, t])
                state = prediction + .5 * mask[:, t] * (values[:, t] - prediction)
        means, variances = [], []
        for action in future_actions.unbind(1):
            base = state if self.method == 'recurrent' else self.step(state, action)
            if self.method == 'nominal':
                state = base
                variance = torch.ones_like(state)
            else:
                correction, raw_var = self.head(torch.cat((state, action, hidden), -1)).chunk(2, -1)
                state = base + .25 * torch.tanh(correction)
                variance = torch.exp(raw_var.clamp(-8., 4.))
                hidden = self.cell(torch.cat((state, action), -1), hidden)
            means.append(state)
            variances.append(variance)
        return {'mean': torch.stack(means, 1), 'variance': torch.stack(variances, 1)}


def train_one(method, seed, lr, norm, prior, train, val, epochs, destination):
    seed_everything(seed)
    model = ForecastModel(method, norm, prior)
    config = TrainConfig(epochs=epochs, batch_size=256, learning_rate=lr,
                         patience=8, reliability_weight=0.)
    result = train_model(model, train, val, config, torch.device('cuda'), seed,
                         wall_clock_deadline=time.perf_counter() + 3600)
    if result.stopped_for_budget:
        save_json(destination.with_suffix('.budget_failure.json'), asdict(result))
        raise RuntimeError('training time guard fired; incomplete run retained')
    payload = {'method': method, 'seed': seed, 'learning_rate': lr,
               'parameter_count': sum(p.numel() for p in model.parameters()),
               'config': asdict(config), 'result': asdict(result)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), destination.with_suffix('.pt'))
    payload['checkpoint_sha256'] = digest(destination.with_suffix('.pt'))
    save_json(destination.with_suffix('.json'), payload)
    print(json.dumps({'method': method, 'seed': seed, 'lr': lr,
                      'epochs': result.epochs_completed, 'best_epoch': result.best_epoch,
                      'seconds': round(result.training_seconds, 2),
                      'validation_nll': result.best_validation_nll}), flush=True)
    return payload


def preparation():
    records = load_records(DATA)
    norm = fit_normalizer(records['train'])
    prior = KinematicThrustPrior.fit(records['train'])
    train = mixed_windows(records['train'], norm)
    val = mixed_windows(records['val'], norm)
    return records, norm, prior, train, val


def develop():
    manifest = OUT / 'training_manifest.json'
    if manifest.exists():
        raise RuntimeError('training manifest exists; use exact-hash resume command')
    records, norm, prior, train, val = preparation()
    payload = {'complete': False, 'sources': provenance(), 'selection': {},
               'norm': {k: v.tolist() for k,v in asdict(norm).items()},
               'prior': prior.fit_metadata,
               'records': {s: [{'id': r.record_id, 'samples': len(r.states)} for r in rs]
                           for s,rs in records.items()},
               'training_windows': len(train.targets), 'validation_windows': len(val.targets),
               'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)}
    save_json(manifest, payload)
    finish_training(payload, records, norm, prior, train, val)


def finish_training(payload, records, norm, prior, train, val):
    if payload.get('complete') is True:
        raise RuntimeError('completed training is immutable; proceed to evaluation')
    if payload['sources'] != provenance():
        raise RuntimeError('source/data change since external training began')
    for method in METHODS:
        trials = []
        for lr in (.001, .003):
            path = OUT / 'tuning' / f'{method}_17_lr{int(lr * 1_000_000)}'
            if path.with_suffix('.json').exists():
                trial = json.loads(path.with_suffix('.json').read_text())
                expected_config = asdict(TrainConfig(epochs=40, batch_size=256,
                                        learning_rate=lr, patience=8, reliability_weight=0.))
                if (trial['method'] != method or trial['seed'] != 17 or
                        trial['learning_rate'] != lr or
                        trial['config'] != expected_config or
                        digest(path.with_suffix('.pt')) != trial['checkpoint_sha256']):
                    raise RuntimeError('checkpoint changed')
            else:
                trial = train_one(method, 17, lr, norm, prior, train, val, 40, path)
            trials.append(trial)
        winner = min(trials, key=lambda t: t['result']['best_validation_nll'])
        payload['selection'][method] = winner['learning_rate']
    save_json(OUT / 'training_manifest.json', payload)
    for seed in SEEDS:
        for method in METHODS:
            path = OUT / 'final' / f'{method}_{seed}'
            if path.with_suffix('.json').exists():
                record = json.loads(path.with_suffix('.json').read_text())
                expected_config = asdict(TrainConfig(epochs=40, batch_size=256,
                                        learning_rate=payload['selection'][method],
                                        patience=8, reliability_weight=0.))
                if (record['method'] != method or record['seed'] != seed or
                        record['config'] != expected_config or
                        record['learning_rate'] != payload['selection'][method] or
                        digest(path.with_suffix('.pt')) != record['checkpoint_sha256']):
                    raise RuntimeError('completed training differs from selection')
                continue
            train_one(method, seed, payload['selection'][method], norm, prior, train, val, 40, path)
    payload['complete'] = True
    payload['training_outputs'] = {p.relative_to(ROOT).as_posix(): digest(p)
                                   for p in sorted(OUT.glob('*/*')) if p.suffix in {'.pt', '.json'}}
    save_json(OUT / 'training_manifest.json', payload)
    print('All development and final training complete; test evaluation remains separate.', flush=True)


@torch.no_grad()
def evaluate():
    manifest = json.loads((OUT / 'training_manifest.json').read_text())
    if manifest['complete'] is not True or manifest['sources'] != provenance():
        raise RuntimeError('incomplete or changed training')
    for path, sha in manifest['training_outputs'].items():
        if digest(ROOT / path) != sha:
            raise RuntimeError(f'training output changed: {path}')
    if (OUT / 'evaluation_manifest.json').exists():
        raise RuntimeError('evaluation already exists')
    records, norm, prior, _, _ = preparation()
    all_rows, window_rows = [], []
    (OUT / 'predictions').mkdir(exist_ok=True)
    for regime, (noise, missing) in REGIMES.items():
        windows, origins = make_windows(records['test'], norm, horizon=50, stride=50,
                                        noise=noise, missing=missing)
        for method in (*METHODS, 'nominal'):
            for seed in (SEEDS if method != 'nominal' else (0,)):
                model = ForecastModel(method, norm, prior)
                if method != 'nominal':
                    model.load_state_dict(torch.load(OUT / 'final' / f'{method}_{seed}.pt',
                                                    weights_only=True, map_location='cpu'))
                model.cuda().eval()
                predictions = []
                for batch in windows_to_loader(windows, 512, False, 0, 0):
                    pred = model(*(v.cuda() for v in batch[:5]))['mean'].cpu().numpy()
                    if not np.isfinite(pred).all():
                        raise FloatingPointError(f'nonfinite prediction {method} {seed} {regime}')
                    predictions.append(pred)
                pred = np.concatenate(predictions)
                np.savez_compressed(OUT / 'predictions' / f'{method}_{seed}_{regime}.npz',
                                    prediction=pred, target=windows.targets,
                                    record=windows.trajectory_ids, origin=origins)
                for horizon in (25, 50):
                    error = pred[:, :horizon] - windows.targets[:, :horizon]
                    sq = error.astype(np.float64)**2
                    pos = (error[..., :3] * norm.state_std[:3])**2
                    vel = (error[..., 3:6] * norm.state_std[3:6])**2
                    for flight in np.unique(windows.trajectory_ids):
                        select = windows.trajectory_ids == flight
                        all_rows.append({'method': method, 'seed': seed, 'regime': regime,
                                         'horizon': horizon, 'flight': flight, 'windows': int(select.sum()),
                                         'nrmse': float(np.sqrt(sq[select].mean())),
                                         'position_rmse_m': float(np.sqrt(pos[select].sum(-1).mean())),
                                         'velocity_rmse_m_s': float(np.sqrt(vel[select].sum(-1).mean()))})
                    for i, flight in enumerate(windows.trajectory_ids):
                        window_rows.append({'method': method,'seed': seed,'regime': regime,
                                            'horizon': horizon,'flight': flight,'origin': int(origins[i]),
                                            'mse': float(sq[i].mean())})
    pd.DataFrame(all_rows).to_csv(OUT / 'flight_metrics.csv', index=False)
    pd.DataFrame(window_rows).to_csv(OUT / 'window_metrics.csv', index=False)
    files = [OUT / 'flight_metrics.csv', OUT / 'window_metrics.csv',
             *sorted((OUT / 'predictions').glob('*.npz'))]
    save_json(OUT / 'evaluation_manifest.json', {'complete': True,
               'training_manifest_sha256': digest(OUT / 'training_manifest.json'),
               'sources': provenance(), 'files': {p.relative_to(ROOT).as_posix(): digest(p) for p in files},
               'flight_rows': len(all_rows), 'window_rows': len(window_rows)})
    print('External evaluation complete.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('pilot', 'train', 'resume', 'evaluate'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for declared experiment budget')
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == 'pilot':
        _, norm, prior, train, val = preparation()
        result = train_one('cfr', 17, .001, norm, prior, train, val, 3, OUT / 'pilot' / 'cfr_17')
        seconds = result['result']['training_seconds']
        print(json.dumps({'projected_21_run_max_seconds': seconds / 3 * 40 * 21,
                          'training_windows': len(train.targets)}), flush=True)
    elif args.stage == 'train':
        develop()
    elif args.stage == 'resume':
        payload = json.loads((OUT / 'training_manifest.json').read_text())
        finish_training(payload, *preparation())
    else:
        evaluate()


if __name__ == '__main__':
    main()
