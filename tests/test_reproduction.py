import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from grader_calibration import analysis, experiment, simulator

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config" / "study.json").read_text())
RESULTS = ROOT / "results"
DATA = ROOT / "data"
FINAL_BUNDLES = 512


def assert_matches(actual, expected, rel_tol=1e-12):
    """Equal structure and values; floats may differ in the last digits across Python builds."""
    actual = json.loads(json.dumps(actual))
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_matches(actual[key], expected[key], rel_tol)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            assert_matches(a, b, rel_tol)
    elif isinstance(expected, float):
        assert math.isclose(actual, expected, rel_tol=rel_tol), (actual, expected)
    else:
        assert actual == expected


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


@pytest.fixture(scope="module")
def endpoints():
    return experiment.read_endpoints(RESULTS / "endpoints.csv.gz")


def test_simulator_reproduces_recorded_bundle(endpoints):
    recorded = json.loads((Path(__file__).parent / "data" / "final_bundle_000000.json").read_text())
    arrays = simulator.run_bundle(experiment.bundle_seed(CONFIG["seeds"]["final"], 0), hidden=True)
    assert sorted(arrays) == sorted(recorded)
    for key, array in arrays.items():
        assert [str(array.dtype), list(array.shape), digest(array)] == recorded[key], key
    values = simulator.endpoints(arrays)
    np.testing.assert_array_equal([values[k] for k in experiment.KEYS], endpoints["final"][0])


def test_simulator_reproduces_recorded_endpoints(endpoints):
    for phase, i in (("pilot", 0), ("final", 511)):
        values = simulator.endpoints(simulator.run_bundle(experiment.bundle_seed(CONFIG["seeds"][phase], i)))
        np.testing.assert_array_equal([values[k] for k in experiment.KEYS], endpoints[phase][i])


def test_final_size_follows_from_pilot(endpoints):
    sizing = json.loads((RESULTS / "sample_size.json").read_text())
    assert_matches(experiment.sample_size(endpoints["pilot"], CONFIG), sizing)
    assert sizing["final_bundles"] == len(endpoints["final"]) == FINAL_BUNDLES


def test_analysis_reproduces_from_endpoints(endpoints):
    assert_matches(analysis.analyze(endpoints["final"], CONFIG), json.loads((RESULTS / "analysis.json").read_text()))


def test_reported_primary_results():
    registered = json.loads((RESULTS / "analysis.json").read_text())["registered"]
    learning = registered["learning_SW_minus_SN_matched"]
    mse = registered["shadow_MSE_SW_minus_SN_matched"]
    assert [round(100 * learning[k], 4) for k in ("mean", "lower", "upper")] == [-0.028, -0.0548, -0.0012]
    assert [round(100 * learning["equivalence"][k], 4) for k in ("lower", "upper")] == [-0.0518, -0.0042]
    assert learning["equivalence"]["equivalent"] is True
    assert [round(mse[k], 6) for k in ("mean", "lower", "upper")] == [-0.000194, -0.000248, -0.00014]
    assert all(row["target_met"] for row in registered.values())


@pytest.mark.skipif(not (DATA / "final").exists(), reason="bundles not generated")
def test_endpoints_and_shadow_errors_reproduce_from_data(endpoints):
    for phase, n in (("pilot", CONFIG["pilot_bundles"]), ("final", FINAL_BUNDLES)):
        matrix = experiment.endpoint_matrix(experiment.stored_bundles(DATA / phase, n))
        np.testing.assert_array_equal(matrix, endpoints[phase])
    shadow = analysis.shadow_errors(
        experiment.stored_bundles(DATA / "final", FINAL_BUNDLES),
        experiment.stored_bundles(DATA / "final", CONFIG["hidden_bundles"]),
    )
    assert_matches(shadow, json.loads((RESULTS / "shadow_errors.json").read_text()))
