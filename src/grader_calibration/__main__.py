"""Command-line entry point: ``python -m grader_calibration <command>``.

pilot      simulate the pilot bundles and derive the final number of bundles
final      simulate the final bundles
analyze    endpoint table, registered/descriptive/post hoc contrasts and shadow-error diagnostics
figures    Figures 1 and 2 from results/analysis.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import analysis, experiment

ROOT = Path.cwd()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="grader_calibration")
    parser.add_argument("command", choices=["pilot", "final", "analyze", "figures"])
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "study.json")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--figures", type=Path, default=ROOT / "figures")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    data, results = args.data, args.results

    if args.command == "pilot":
        n = config["pilot_bundles"]
        experiment.run_phase(config, "pilot", n, data / "pilot", args.workers)
        pilot = experiment.endpoint_matrix(experiment.stored_bundles(data / "pilot", n))
        _write(results / "sample_size.json", experiment.sample_size(pilot, config))
    elif args.command == "final":
        n = json.loads((results / "sample_size.json").read_text())["final_bundles"]
        experiment.run_phase(config, "final", n, data / "final", args.workers)
    elif args.command == "analyze":
        n = json.loads((results / "sample_size.json").read_text())["final_bundles"]
        matrices = {
            "pilot": experiment.endpoint_matrix(experiment.stored_bundles(data / "pilot", config["pilot_bundles"])),
            "final": experiment.endpoint_matrix(experiment.stored_bundles(data / "final", n)),
        }
        experiment.write_endpoints(results / "endpoints.csv.gz", matrices)
        _write(results / "analysis.json", analysis.analyze(matrices["final"], config))
        shadow = analysis.shadow_errors(
            experiment.stored_bundles(data / "final", n),
            experiment.stored_bundles(data / "final", config["hidden_bundles"]),
        )
        _write(results / "shadow_errors.json", shadow)
    elif args.command == "figures":
        from . import figures

        figures.make_all(results, args.figures)


if __name__ == "__main__":
    main()
