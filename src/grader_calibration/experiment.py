"""Seed bundles, the pilot sample-size rule, bundle storage and the table of primitive endpoints."""

from __future__ import annotations

import csv
import gzip
import io
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import NormalDist

import numpy as np

from . import simulator

FIELDS = ("endpoint", "method", "controller", "subtypes", "prior")


def _keys():
    _, policies, observers = simulator.layout()
    keys = {("delayed_success", *k) for k in policies}
    for J, controller, prior, observer, _ in observers:
        keys |= {(e, observer, controller, J, prior) for e in ("shadow_error_mse_final", "shadow_error_bias_final")}
    return sorted(keys)


KEYS = _keys()


def bundle_seed(base, i):
    """Bundle ``i`` of a phase is simulated from the 64-bit seed ``base * 2^32 + i``."""
    return (base << 32) | i


def bundle_path(directory, i):
    return Path(directory) / f"bundle_{i:06d}.npz"


def _simulate(task):
    path, seed, hidden = task
    arrays = simulator.run_bundle(seed, hidden)
    partial = path.with_suffix(".partial.npz")
    np.savez_compressed(partial, **arrays)
    partial.rename(path)
    return path.name


def run_phase(config, phase, bundles, directory, workers=4):
    """Simulate the bundles of a phase that are not stored yet. The first final bundles keep hidden arrays."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    base = config["seeds"][phase]
    tasks = [
        (bundle_path(directory, i), bundle_seed(base, i), phase == "final" and i < config["hidden_bundles"])
        for i in range(bundles)
        if not bundle_path(directory, i).exists()
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, name in enumerate(pool.map(_simulate, tasks), 1):
            print(f"{phase}: {name} ({done}/{len(tasks)})", flush=True)


def stored_bundles(directory, bundles):
    """Arrays of the stored bundles ``0 .. bundles - 1``, one bundle at a time."""
    for i in range(bundles):
        with np.load(bundle_path(directory, i)) as arrays:
            yield arrays


def endpoint_matrix(bundles):
    """Endpoints of a sequence of bundles as an array of shape (bundles, len(KEYS))."""
    rows = []
    for arrays in bundles:
        values = simulator.endpoints(arrays)
        rows.append([values[k] for k in KEYS])
    return np.array(rows)


def write_endpoints(path, matrices):
    with gzip.GzipFile(path, "wb", mtime=0) as raw, io.TextIOWrapper(raw, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("phase", "bundle", *FIELDS, "value"))
        for phase, matrix in matrices.items():
            for i, row in enumerate(matrix):
                for key, value in zip(KEYS, row, strict=True):
                    writer.writerow((phase, i, *key, repr(float(value))))


def read_endpoints(path):
    """``{phase: matrix}`` from the endpoint table, columns in the order of ``KEYS``."""
    column = {k: j for j, k in enumerate(KEYS)}
    rows = {}
    with gzip.open(path, "rt", newline="") as f:
        for r in csv.DictReader(f):
            key = (r["endpoint"], r["method"], r["controller"], int(r["subtypes"]), r["prior"])
            rows.setdefault(r["phase"], []).append((int(r["bundle"]), column[key], float(r["value"])))
    matrices = {}
    for phase, entries in rows.items():
        matrix = np.full((max(e[0] for e in entries) + 1, len(KEYS)), np.nan)
        for i, j, value in entries:
            matrix[i, j] = value
        if np.isnan(matrix).any():
            raise ValueError(f"incomplete endpoint table for {phase}")
        matrices[phase] = matrix
    return matrices


def contrasts(matrix, registry):
    """Registered contrasts per bundle: fixed linear combinations of the bundle's endpoints."""
    column = {k: j for j, k in enumerate(KEYS)}
    out = np.empty((len(matrix), len(registry)))
    for j, entry in enumerate(registry):
        terms = []
        for *key, coefficient in entry["terms"]:
            numerator, denominator = map(int, coefficient.split("/"))
            terms.append((column[tuple(key)], numerator, denominator))
        for i, row in enumerate(matrix):
            out[i, j] = math.fsum(row[c] * num / den for c, num, den in terms)
    return out


def critical_values(config):
    """Two-sided Bonferroni critical value of each family."""
    return {
        family: NormalDist().inv_cdf(1 - config["family_alpha"][family] / (2 * decisions))
        for family, decisions in config["family_decisions"].items()
    }


def sample_size(pilot, config):
    """N = round_to * ceil(max(minimum, max_c (z_c s_c / h_c)^2) / round_to) from the pilot SDs."""
    registry = config["registry"]
    values = contrasts(pilot, registry)
    z = critical_values(config)
    sd = {entry["id"]: float(values[:, j].std(ddof=1)) for j, entry in enumerate(registry)}
    required = {e["id"]: (z[e["family"]] * sd[e["id"]] / e["target_half_width"]) ** 2 for e in registry}
    spec = config["sample_size"]
    n = spec["round_to"] * math.ceil(max(spec["minimum"], max(required.values())) / spec["round_to"])
    return dict(
        pilot_bundles=len(pilot),
        sd=sd,
        required_bundles=required,
        determining_contrast=max(required, key=required.get),
        final_bundles=n,
    )
