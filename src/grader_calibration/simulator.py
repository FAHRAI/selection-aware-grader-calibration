"""Adaptive tutor with a noisy automated grader and a targeted audit of one answer per round.

Six cohorts of ten learners practise three binary skills for 40 rounds each. In every round the
controller picks a skill for each learner, each skill is split into ``J`` grading subtypes, and the
grader flips the learner's response with the context's error rate. After all ten grades are visible,
one answer is verified; the tutor updates its mastery belief with the verified label for that answer
and with the grade, discounted by the current error-rate estimate, for the others.

Audited policies (all verify one answer per round)
    UW  uniform audit, count estimator
    SN  targeted audit, unweighted count estimator
    SW  targeted audit, count estimator weighted by 1 / (10 pi)
    SF  targeted audit, error rate frozen at the prior mean
    SX  targeted audit, true error rates supplied
Reference policies (no audit)
    NA  error rate frozen at the prior mean
    PG  error-free grades

Controllers: G always teaches the skill with the lowest belief; C80 and C95 additionally certify a skill
once its belief reaches 0.8 or 0.95 and stop teaching it.

Shadow observers (UW, SN, SW and the likelihood estimator SL) replay the audit on the fixed trajectory
of the NA policy, so their estimates never influence teaching.
"""

from __future__ import annotations

import numpy as np

from .estimators import PRIORS, CountEstimator, LikelihoodEstimator

COHORTS, PUPILS, STEPS, SKILLS = 6, 10, 40, 3
ROUNDS = COHORTS * STEPS
SUBTYPES = (2, 20)
MAX_CONTEXTS = SKILLS * max(SUBTYPES)
CONTROLLERS = ("G", "C80", "C95")
THRESHOLDS = {"C80": 0.8, "C95": 0.95}
METHODS = ("UW", "SN", "SW", "SF", "SX", "NA", "PG")
COUNTING = ("UW", "SN", "SW")
OBSERVERS = ("UW", "SN", "SW", "SL")
FROZEN_ERROR = 0.14
INITIAL_MASTERY = 0.2
LEARNING = 0.075
FORGETTING = 0.002
RETENTION = 0.998
DELAYED_RETENTION = 0.98**7
STREAMS = dict(rates=1, initial=2, response=3, flip=4, learn=5, forget=6, subtype=7, tie=8, audit=9)
PADDED = ("final_estimates", "true_context_rates", "context_weights", "final_F", "final_N")


def environment(seed):
    """Exogenous uniforms of one bundle; each stream uses ``SeedSequence([seed, stream])``."""
    rng = {k: np.random.default_rng(np.random.SeedSequence([int(seed), v])) for k, v in STREAMS.items()}
    shapes = dict(
        initial=(COHORTS, PUPILS, SKILLS),
        response=(COHORTS, STEPS, PUPILS),
        flip=(COHORTS, STEPS, PUPILS),
        learn=(COHORTS, STEPS, PUPILS),
        forget=(COHORTS, STEPS, PUPILS, SKILLS),
        subtype=(COHORTS, STEPS, PUPILS),
        tie=(COHORTS, STEPS, PUPILS, SKILLS),
        audit=(COHORTS, STEPS),
    )
    env = {k: rng[k].random(shape) for k, shape in shapes.items()}
    env["tie"] *= 1e-8
    env["rates"] = 0.02 + 0.36 * rng["rates"].beta(1.5, 3.0, 60)
    return env


def context_distribution(J):
    """Subtype probabilities and the weight of each of the 3J contexts."""
    probs = np.array([0.6, 0.4]) if J == 2 else 1 / np.arange(1, J + 1)
    probs = probs / probs.sum()
    return probs, np.tile(probs / 3, 3)


def context_rates(env, J):
    return env["rates"][np.concatenate([np.arange(J) + 20 * k for k in range(3)])]


def posterior(b, z, e):
    """Mastery belief after grade ``z`` when grades are flipped with probability ``e``."""
    l1 = 0.9 * (1 - e) + 0.1 * e
    l0 = 0.2 * (1 - e) + 0.8 * e
    l1 = np.where(z, l1, 1 - l1)
    l0 = np.where(z, l0, 1 - l0)
    return b * l1 / np.maximum(b * l1 + (1 - b) * l0, 1e-15)


def select_skills(belief, certified, controller, t, tie):
    score = belief + tie
    if controller == "G":
        return np.argmin(score, axis=1)
    skill = np.argmin(np.where(certified, np.inf, score), axis=1)
    skill[np.all(certified, axis=1)] = t % SKILLS
    return skill


def audit_probabilities(p, z, uniform=False):
    """Targeted audit: probability 0.02 + 0.8 s_i / sum(s), where s_i is the grade's disagreement with p_i."""
    if uniform:
        return np.full(PUPILS, 0.1)
    score = np.where(z, 1 - p, p)
    total = float(score.sum())
    return np.full(PUPILS, 0.1) if total == 0 else 0.02 + 0.8 * score / total


def choose_audit(pi, u):
    return min(int(np.searchsorted(np.cumsum(pi), u, side="right")), PUPILS - 1)


def teach(env, J, controller, prior, method):
    """One closed-loop trajectory over six cohorts. Keys starting with ``hidden_`` are never seen by the tutor."""
    probs, weights = context_distribution(J)
    rates = context_rates(env, J)
    estimator = CountEstimator(prior, 3 * J) if method in COUNTING else None
    fixed = rates if method == "SX" else np.full(3 * J, 0.0 if method == "PG" else FROZEN_ERROR)
    trace = dict(
        belief_pre=np.empty((ROUNDS, PUPILS, SKILLS)),
        belief_post=np.empty((ROUNDS, PUPILS, SKILLS)),
        skills=np.empty((ROUNDS, PUPILS), np.int8),
        context=np.empty((ROUNDS, PUPILS), np.int16),
        z=np.empty((ROUNDS, PUPILS), np.int8),
        pi=np.zeros((ROUNDS, PUPILS)),
        selected=np.full(ROUNDS, -1, np.int8),
        verified_y=np.full(ROUNDS, -1, np.int8),
        e_used=np.empty((ROUNDS, PUPILS)),
        e_before=np.empty((ROUNDS, PUPILS)),
        F_increment=np.zeros(ROUNDS),
        N_increment=np.zeros(ROUNDS),
        certified_post=np.empty((ROUNDS, PUPILS, SKILLS), bool),
        hidden_y=np.empty((ROUNDS, PUPILS), np.int8),
        hidden_mastery_post=np.empty((ROUNDS, PUPILS, SKILLS), bool),
        cohort_mastery_counts=np.empty(COHORTS, np.int16),
        premature_certifications=np.zeros(COHORTS, np.int16),
        new_certifications=np.zeros(COHORTS, np.int16),
        action_switch_count=np.zeros(COHORTS, np.int16),
    )
    pupil = np.arange(PUPILS)
    for co in range(COHORTS):
        truth = env["initial"][co] < INITIAL_MASTERY
        belief = np.full((PUPILS, SKILLS), INITIAL_MASTERY)
        certified = np.zeros((PUPILS, SKILLS), bool)
        previous = None
        for t in range(STEPS):
            r = STEPS * co + t
            trace["belief_pre"][r] = belief
            skill = select_skills(belief, certified, controller, t, env["tie"][co, t])
            if previous is not None:
                trace["action_switch_count"][co] += np.count_nonzero(skill != previous)
            previous = skill.copy()
            subtype = np.searchsorted(np.cumsum(probs), env["subtype"][co, t], side="right")
            ctx = J * skill + subtype
            b = belief[pupil, skill]
            p = 0.2 + 0.7 * b
            e = (estimator.means if estimator is not None else fixed)[ctx]
            mastered = truth[pupil, skill]
            y = env["response"][co, t] < (0.2 + 0.7 * mastered)
            true_e = np.zeros(PUPILS) if method == "PG" else rates[ctx]
            z = np.logical_xor(y, env["flip"][co, t] < true_e)
            observed = z.copy()
            used = e.copy()
            if method not in ("NA", "PG"):
                pi = audit_probabilities(p, z, method == "UW")
                selected = choose_audit(pi, env["audit"][co, t])
                verified = int(y[selected])
                observed[selected] = verified
                used[selected] = 0.0
                trace["pi"][r] = pi
                trace["selected"][r] = selected
                trace["verified_y"][r] = verified
            post = posterior(b, observed, used)
            belief[pupil, skill] = post + (1 - post) * LEARNING
            truth[pupil, skill] = mastered | (~mastered & (env["learn"][co, t] < LEARNING))
            truth &= env["forget"][co, t] >= FORGETTING
            belief *= RETENTION
            if controller in THRESHOLDS:
                new = (belief >= THRESHOLDS[controller]) & ~certified
                trace["new_certifications"][co] += np.count_nonzero(new)
                trace["premature_certifications"][co] += np.count_nonzero(new & ~truth)
                certified |= new
            if estimator is not None:
                error = int(z[selected] != observed[selected])
                weight = 1 / (10 * pi[selected]) if method in ("UW", "SW") else 1.0
                estimator.update(int(ctx[selected]), error, weight)
                trace["F_increment"][r] = error * weight
                trace["N_increment"][r] = weight
            trace["belief_post"][r] = belief
            trace["skills"][r] = skill
            trace["context"][r] = ctx
            trace["z"][r] = z
            trace["e_before"][r] = e
            trace["e_used"][r] = used
            trace["certified_post"][r] = certified
            trace["hidden_y"][r] = y
            trace["hidden_mastery_post"][r] = truth
        trace["cohort_mastery_counts"][co] = truth.sum()
    trace["final_estimates"] = estimator.means.copy() if estimator is not None else fixed.copy()
    trace["true_context_rates"] = rates.copy()
    trace["context_weights"] = weights
    trace["final_F"] = estimator.F.copy() if estimator is not None else np.zeros(3 * J)
    trace["final_N"] = estimator.N.copy() if estimator is not None else np.zeros(3 * J)
    return trace


def shadow(driver, env, J, prior, observer):
    """Replay the audit on a fixed NA trajectory and update one estimator."""
    estimator = LikelihoodEstimator(prior, 3 * J) if observer == "SL" else CountEstimator(prior, 3 * J)
    out = dict(
        selected=np.empty(ROUNDS, np.int8),
        verified_y=np.empty(ROUNDS, np.int8),
        pi=np.empty((ROUNDS, PUPILS)),
        F_increment=np.zeros(ROUNDS),
        N_increment=np.zeros(ROUNDS),
    )
    audit = env["audit"].ravel()
    for r in range(ROUNDS):
        ctx, z = driver["context"][r], driver["z"][r]
        p = 0.2 + 0.7 * driver["belief_pre"][r, np.arange(PUPILS), driver["skills"][r]]
        pi = audit_probabilities(p, z, observer == "UW")
        selected = choose_audit(pi, audit[r])
        verified = int(driver["hidden_y"][r, selected])
        if observer == "SL":
            estimator.update_round(ctx, z, p, selected, verified)
        else:
            error = int(z[selected] != verified)
            weight = 1 / (10 * pi[selected]) if observer in ("UW", "SW") else 1.0
            estimator.update(int(ctx[selected]), error, weight)
            out["F_increment"][r] = weight * error
            out["N_increment"][r] = weight
        out["selected"][r] = selected
        out["verified_y"][r] = verified
        out["pi"][r] = pi
    counts = observer != "SL"
    out["final_estimates"] = estimator.means.copy()
    out["final_F"] = estimator.F.copy() if counts else np.zeros(3 * J)
    out["final_N"] = estimator.N.copy() if counts else np.zeros(3 * J)
    return out


def layout():
    """Storage order of a bundle.

    Returns the 60 unique learner runs ``(subtypes, controller, prior, method)``, a map from every
    policy ``(method, controller, subtypes, prior)`` to its run, and the 48 shadow observers
    ``(subtypes, controller, prior, observer, driver run)``. Runs whose estimate does not depend on the
    prior (SF, SX, NA, PG) are simulated once and shared by both prior labels.
    """
    runs, policies, observers = [], {}, []
    for J in SUBTYPES:
        for controller in CONTROLLERS:
            index = {}
            for prior in PRIORS:
                for method in METHODS:
                    key = (method, prior if method in COUNTING else None)
                    if key not in index:
                        index[key] = len(runs)
                        runs.append((J, controller, key[1], method))
                    policies[(method, controller, J, prior)] = index[key]
                for observer in OBSERVERS:
                    observers.append((J, controller, prior, observer, index[("NA", None)]))
    return runs, policies, observers


def _stack(rows):
    out = {}
    for key in rows[0]:
        if key in PADDED:
            out[key] = np.zeros((len(rows), MAX_CONTEXTS))
            for i, row in enumerate(rows):
                out[key][i, : len(row[key])] = row[key]
        else:
            out[key] = np.stack([row[key] for row in rows])
    return out


def run_bundle(seed, hidden=False):
    """Arrays of one bundle: ``world_*`` for learner runs, ``shadow_*`` for observers, optionally ``hidden_*``."""
    env = environment(seed)
    runs, _, observers = layout()
    worlds = [teach(env, J, controller, prior, method) for J, controller, prior, method in runs]
    shadows = [shadow(worlds[driver], env, J, prior, observer) for J, _, prior, observer, driver in observers]
    public = [{k: v for k, v in w.items() if not k.startswith("hidden_")} for w in worlds]
    arrays = {"world_" + k: v for k, v in _stack(public).items()}
    arrays.update({"shadow_" + k: v for k, v in _stack(shadows).items()})
    if hidden:
        for key in ("hidden_y", "hidden_mastery_post"):
            arrays[key] = np.stack([w[key] for w in worlds])
    return arrays


def endpoints(arrays):
    """The 180 primitive endpoints of a bundle, keyed by ``(endpoint, method, controller, subtypes, prior)``.

    delayed_success: expected success on a fresh task seven days after each cohort, averaged over cohorts.
    shadow_error_mse_final / shadow_error_bias_final: context-weighted squared and signed error of the
    observer's final estimate.
    """
    _, policies, observers = layout()
    out = {}
    counts = arrays["world_cohort_mastery_counts"]
    for (method, controller, J, prior), run in policies.items():
        value = np.mean(0.2 + 0.7 * (counts[run] / (PUPILS * SKILLS)) * DELAYED_RETENTION)
        out[("delayed_success", method, controller, J, prior)] = float(value)
    for o, (J, controller, prior, observer, driver) in enumerate(observers):
        n = 3 * J
        error = arrays["shadow_final_estimates"][o, :n] - arrays["world_true_context_rates"][driver, :n]
        weights = arrays["world_context_weights"][driver, :n]
        out[("shadow_error_mse_final", observer, controller, J, prior)] = float(np.sum(weights * error**2))
        out[("shadow_error_bias_final", observer, controller, J, prior)] = float(np.sum(weights * error))
    return out
