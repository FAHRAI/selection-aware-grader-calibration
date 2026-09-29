# Selection-aware grader calibration under a fixed audit budget

Code, configuration and results for the paper

> O. Kholodniak, O. Prokhorov. *Selection-aware grader calibration and learner outcomes under a fixed audit budget:
> a simulation study.* Manuscript under review, 2026.

## The study in brief

An adaptive tutor teaches six cohorts of ten learners three binary skills. An automated grader flips responses
with an error rate that depends on the grading context (skill and subtype; 6 or 60 contexts). In every round one of
the ten graded answers is verified, chosen with probability that grows with the disagreement between the grade and
the tutor's prediction. The tutor estimates the error rates from the verified answers.

- **Estimators.** SN counts audited errors without correction; SW weights each audited answer by `1 / (10 pi)`, the
  inverse of its audit probability; UW uses a uniform audit. Reference policies freeze the error rate (SF, NA),
  receive the true rates (SX) or grade without error (PG). Controllers G, C80 and C95 differ in whether skills are
  certified and dropped once the belief reaches 0.8 or 0.95.
- **Two experiments.** In the *shadow* experiment the estimators replay the audit on the same fixed teaching
  trajectory and their final error-rate MSE is compared. In the *closed loop* each policy's estimate drives its own
  teaching and the expected delayed learner success is compared.
- **Design.** A seed bundle contains all 60 learner runs and 48 shadow observers of one set of random inputs, so all
  contrasts are paired within a bundle. A pilot of 256 bundles fixed the final sample of 512 bundles.
- **Result.** Weighting reduced the final error-rate MSE by 0.000194 (98.333% interval [−0.000248, −0.000140]),
  whereas expected delayed success changed by −0.0280 percentage points ([−0.0548, −0.0012]); the 96.667%
  equivalence interval [−0.0518, −0.0042] lies inside the ±0.1 pp margin.

The protocol, simulation code, contrast registry and seeds were fixed and fingerprinted with SHA-256 before the
pilot and the final data were generated. The timestamped records are available from the authors on request.

## Repository layout

| Path | Contents |
|---|---|
| `src/grader_calibration/simulator.py` | Learners, grader, audit, controllers, closed-loop policies and shadow observers of one bundle |
| `src/grader_calibration/estimators.py` | Count and likelihood estimators of the error rates, quadrature for the matched prior |
| `src/grader_calibration/experiment.py` | Bundle seeds and storage, the endpoint table, registered contrasts and the pilot sample-size rule |
| `src/grader_calibration/analysis.py` | Registered families A–C, descriptive and post hoc contrasts, shadow-error diagnostics |
| `src/grader_calibration/figures.py` | Figures 1 and 2 of the paper |
| `config/study.json` | Seeds, pilot size, sample-size rule, familywise error rates and the contrast registry |
| `results/` | Outputs reported in the paper |
| `tests/` | Reproduction checks against recorded bundles and released results |

## Reproducing the results

The recorded run used Python 3.12 and NumPy 2.3.5 on macOS (arm64) with single-threaded BLAS, which the package sets
on import. Other platforms may differ in the last digits.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[figures,test]"
pytest
python -m grader_calibration figures
```

`pytest` re-simulates recorded bundles, checks every array of final bundle 0 against its SHA-256 digest and
recomputes the sample size and all contrasts from the endpoint table. Once the bundles have been generated in
`data/` (see [Data](#data)), it also recomputes the endpoints of all 768 bundles and the shadow-error diagnostics.
The commands

```bash
python -m grader_calibration pilot     # pilot bundles and results/sample_size.json
python -m grader_calibration final     # final bundles; existing bundles are kept
python -m grader_calibration analyze   # endpoints.csv.gz, analysis.json and shadow_errors.json
```

reuse stored bundles and simulate only what is missing; on an empty `data/` directory they rerun the study
(about five minutes with four workers, 5.1 GB).

## Results

| File | Contents |
|---|---|
| `sample_size.json` | Pilot standard deviations of the registered contrasts and the final number of bundles |
| `endpoints.csv.gz` | 180 primitive endpoints (delayed success, shadow MSE and signed error) of every pilot and final bundle |
| `analysis.json` | Registered contrasts with the equivalence test, descriptive levels and paired differences, post hoc contrasts |
| `shadow_errors.json` | Contextwise decomposition of the SN and SW errors and the forecast diagnostic on the NA trajectories |

## Data

`results/endpoints.csv.gz` holds every primitive endpoint of the 256 pilot and 512 final bundles. The bundles
themselves (every learner trajectory, audit and estimator update; final bundles 0–15 also keep the hidden responses
and mastery states) are deterministic in the seeds of `config/study.json` and are regenerated by the `pilot` and
`final` commands; the regenerated bundles are identical, bit for bit, to those of the recorded run.

## Citation and license

See `CITATION.cff`. Code: MIT. Results and data: CC BY 4.0.
