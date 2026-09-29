"""Registered contrasts, descriptive summaries, post hoc contrasts and the shadow-error diagnostics.

All intervals are normal Monte Carlo intervals over independent bundles. Registered contrasts use the
Bonferroni critical value of their family; descriptive and post hoc intervals are pointwise 95%.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

from .experiment import FIELDS, KEYS, contrasts, critical_values
from .simulator import CONTROLLERS, SUBTYPES, layout

Z95 = NormalDist().inv_cdf(0.975)
COLUMN = {k: j for j, k in enumerate(KEYS)}


def summarize(values, z=None):
    values = np.asarray(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    se = sd / math.sqrt(len(values))
    row = dict(mean=mean, sd=sd, se=se, n=len(values))
    if z is not None:
        row.update(lower=mean - z * se, upper=mean + z * se)
    return row


def registered(final, config):
    """Families A-C with difference decisions and, for the primary learning contrast, the TOST equivalence test."""
    values = contrasts(final, config["registry"])
    z = critical_values(config)
    out = {}
    for j, entry in enumerate(config["registry"]):
        family = entry["family"]
        row = dict(family=family, unit=entry["unit"], **summarize(values[:, j], z[family]))
        row["half_width"] = z[family] * row["se"]
        row["target_half_width"] = entry["target_half_width"]
        row["target_met"] = row["half_width"] <= entry["target_half_width"]
        row["reject_zero"] = abs(row["mean"]) > row["half_width"]
        if "equivalence" in entry["decisions"]:
            alpha = config["family_alpha"][family] / config["family_decisions"][family]
            margin = config["equivalence"]["margin"]
            ze = NormalDist().inv_cdf(1 - alpha)
            lower, upper = row["mean"] - ze * row["se"], row["mean"] + ze * row["se"]
            row["equivalence"] = dict(
                margin=margin, lower=lower, upper=upper, equivalent=lower > -margin and upper < margin
            )
        out[entry["id"]] = row
    return out


def _learning_difference(final, first, second, controller, J, prior):
    a = final[:, COLUMN[("delayed_success", first, controller, J, prior)]]
    b = final[:, COLUMN[("delayed_success", second, controller, J, prior)]]
    return a - b


def descriptive(final, priors):
    """Levels of every primitive endpoint and paired learning differences in every cell."""
    levels = [dict(zip(FIELDS, key, strict=True), **summarize(final[:, j])) for j, key in enumerate(KEYS)]
    paired = []
    for prior in priors:
        for controller in CONTROLLERS:
            for J in SUBTYPES:
                for first, second in (("SW", "SN"), ("SW", "UW"), ("PG", "NA")):
                    values = _learning_difference(final, first, second, controller, J, prior)
                    paired.append(
                        dict(
                            contrast=f"{first} - {second}",
                            controller=controller,
                            subtypes=J,
                            prior=prior,
                            **summarize(values, Z95),
                        )
                    )
    return dict(levels=levels, paired_learning=paired)


def post_hoc(final):
    """Matched-prior contrasts added after the registered analysis, averaged equally over the design cells."""

    def cell_mean(endpoint, first, second, subtypes):
        cells = [(c, J) for c in CONTROLLERS for J in subtypes]
        a = [COLUMN[(endpoint, first, c, J, "matched")] for c, J in cells]
        b = [COLUMN[(endpoint, second, c, J, "matched")] for c, J in cells]
        return np.array([float(np.mean([row[i] - row[j] for i, j in zip(a, b, strict=True)])) for row in final])

    vectors = {}
    for first, second in (("SX", "SF"), ("SX", "SW")):
        vectors[f"learning_{first}_minus_{second}"] = cell_mean("delayed_success", first, second, SUBTYPES)
    for first, second in (("SW", "SN"), ("SW", "UW"), ("SL", "SW")):
        for label, subtypes in (("all", SUBTYPES), ("J2", (2,)), ("J20", (20,))):
            vectors[f"MSE_{first}_minus_{second}_{label}"] = cell_mean(
                "shadow_error_mse_final", first, second, subtypes
            )
    vectors["MSE_SW_minus_SN_J2_minus_J20"] = vectors["MSE_SW_minus_SN_J2"] - vectors["MSE_SW_minus_SN_J20"]
    return {k: summarize(v, Z95) for k, v in vectors.items()}


def analyze(final, config):
    return dict(
        final_bundles=len(final),
        registered=registered(final, config),
        descriptive=descriptive(final, ("matched", "historical_weak")),
        post_hoc=post_hoc(final),
    )


def error_decomposition(bundles):
    """Matched-prior SN and SW shadow errors: mean(error^2) = mean(error)^2 + var(error) per context.

    Moments are taken across bundles for each fixed context and combined with the fixed context weights.
    True rates vary across bundles, so the variance term is not the estimator variance at a fixed rate.
    """
    _, _, observers = layout()
    selected = [
        (o, driver, (observer, controller, J))
        for o, (J, controller, prior, observer, driver) in enumerate(observers)
        if prior == "matched" and observer in ("SN", "SW")
    ]
    errors, weights = {}, {}
    for arrays in bundles:
        estimates = arrays["shadow_final_estimates"]
        truth = arrays["world_true_context_rates"]
        context_weights = arrays["world_context_weights"]
        for o, driver, key in selected:
            n = 3 * key[2]
            errors.setdefault(key, []).append((estimates[o] - truth[driver])[:n])
            weights[key] = context_weights[driver, :n]
    cells = {}
    for key, rows in errors.items():
        e = np.stack(rows)
        w = weights[key]
        m = e.mean(axis=0)
        cells["/".join(map(str, key))] = dict(
            mse=float(np.sum(w * np.mean(e * e, axis=0))),
            squared_context_mean_error=float(np.sum(w * m * m)),
            variance_of_context_error=float(np.sum(w * np.var(e, axis=0, ddof=0))),
            signed_mean_error=float(np.sum(w * m)),
        )
    pooled = {}
    for label, subtypes in (("all", SUBTYPES), ("J2", (2,)), ("J20", (20,))):
        pooled[label] = {}
        for method in ("SN", "SW"):
            rows = [cells[f"{method}/{c}/{J}"] for c in CONTROLLERS for J in subtypes]
            pooled[label][method] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        pooled[label]["SN_minus_SW"] = {k: pooled[label]["SN"][k] - pooled[label]["SW"][k] for k in pooled[label]["SN"]}
    return dict(cells=cells, pooled=pooled)


def forecast_diagnostic(bundles):
    """Pre-grade forecast p = 0.2 + 0.7 b against the hidden response on the NA trajectories.

    Ratios of sums over all responses of bundles that keep hidden arrays, per context and per cell.
    """
    runs, _, _ = layout()
    drivers = [(u, J, controller) for u, (J, controller, _, method) in enumerate(runs) if method == "NA"]
    pooled, n_bundles = {}, 0
    for arrays in bundles:
        n_bundles += 1
        skills = arrays["world_skills"]
        belief = arrays["world_belief_pre"]
        context = arrays["world_context"]
        hidden_y = arrays["hidden_y"]
        for u, J, controller in drivers:
            p = 0.2 + 0.7 * np.take_along_axis(belief[u], skills[u][..., None], axis=2)[..., 0]
            ctx = context[u].ravel()
            counts = np.bincount(ctx, minlength=3 * J)
            sums_p = np.bincount(ctx, weights=p.ravel(), minlength=3 * J)
            sums_y = np.bincount(ctx, weights=hidden_y[u].ravel(), minlength=3 * J)
            for c in range(3 * J):
                total = pooled.setdefault((controller, J, c), dict(count=0, sum_p=0.0, sum_y=0))
                total["count"] += int(counts[c])
                total["sum_p"] += float(sums_p[c])
                total["sum_y"] += int(sums_y[c])
    contexts = []
    for (controller, J, c), total in sorted(pooled.items()):
        n = total["count"]
        contexts.append(
            dict(
                controller=controller,
                subtypes=J,
                context=c,
                count=n,
                mean_p=total["sum_p"] / n if n else None,
                mean_y=total["sum_y"] / n if n else None,
            )
        )
    cells = []
    for controller in CONTROLLERS:
        for J in SUBTYPES:
            rows = [pooled[(controller, J, c)] for c in range(3 * J)]
            n = sum(r["count"] for r in rows)
            sum_p = math.fsum(r["sum_p"] for r in rows)
            sum_y = sum(r["sum_y"] for r in rows)
            cells.append(
                dict(
                    controller=controller,
                    subtypes=J,
                    responses=n,
                    mean_p=sum_p / n,
                    mean_y=sum_y / n,
                    mean_p_minus_mean_y=(sum_p - sum_y) / n,
                )
            )
    return dict(bundles=n_bundles, cells=cells, contexts=contexts)


def shadow_errors(final_bundles, hidden_bundles):
    return dict(
        error_decomposition=error_decomposition(final_bundles),
        forecast_diagnostic=forecast_diagnostic(hidden_bundles),
    )
