"""Statistical rigor helpers for the LLM benchmark — pure stdlib.

A small eval (N~40) cannot be read as a point estimate: a 90% vs 87.5% headline
is ~1 question of difference and almost certainly inside the noise band. These
helpers attach uncertainty to accuracy (percentile bootstrap CI) and test
whether a *paired* difference between two models on the same questions is real
(exact McNemar / binomial sign test).

Pure stdlib (random, math) on purpose, so the benchmark keeps running anywhere
with just Python + httpx — no numpy/scipy dependency.
"""

from __future__ import annotations

import math
import random

DEFAULT_RESAMPLES = 10000


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _percentile(sorted_xs: list[float], p: float) -> float:
    """Linear-interpolation percentile (numpy 'linear' method). p in [0, 100]."""
    if not sorted_xs:
        raise ValueError("cannot take a percentile of an empty sequence")
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    k = (len(sorted_xs) - 1) * (p / 100.0)
    lo_idx = math.floor(k)
    hi_idx = math.ceil(k)
    if lo_idx == hi_idx:
        return sorted_xs[int(k)]
    return sorted_xs[lo_idx] * (hi_idx - k) + sorted_xs[hi_idx] * (k - lo_idx)


def bootstrap_ci(
    outcomes: list[float],
    confidence: float = 0.95,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean of per-question outcomes.

    ``outcomes`` is one number per question (e.g. 1.0 correct / 0.0 wrong).
    Returns ``(point, lo, hi)`` — the observed mean and the CI bounds.

    Note: this is a plain *percentile* bootstrap. For a proportion at small N it
    slightly undercovers vs Wilson or BCa intervals — it is an honest
    uncertainty band, not a coverage guarantee. The significance verdict is
    delegated to the exact McNemar test (:func:`mcnemar_exact`), not to this CI.
    """
    if not outcomes:
        raise ValueError("outcomes must be non-empty")

    n = len(outcomes)
    point = _mean(outcomes)
    rng = random.Random(seed)

    means = []
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            total += outcomes[rng.randrange(n)]
        means.append(total / n)
    means.sort()

    alpha = 1.0 - confidence
    lo = _percentile(means, alpha / 2 * 100)
    hi = _percentile(means, (1 - alpha / 2) * 100)
    return (point, lo, hi)


def paired_bootstrap_delta_ci(
    a: list[float],
    b: list[float],
    confidence: float = 0.95,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Paired bootstrap CI for the delta ``mean(b) - mean(a)``.

    ``a`` and ``b`` are aligned per-question outcomes for two models on the SAME
    questions; resampling the shared index preserves the pairing. Returns
    ``(delta, lo, hi)``.
    """
    if len(a) != len(b):
        raise ValueError("a and b must have equal length (paired)")
    if not a:
        raise ValueError("inputs must be non-empty")

    n = len(a)
    delta = _mean(b) - _mean(a)
    rng = random.Random(seed)

    deltas = []
    for _ in range(n_resamples):
        sum_a = 0.0
        sum_b = 0.0
        for _ in range(n):
            i = rng.randrange(n)
            sum_a += a[i]
            sum_b += b[i]
        deltas.append((sum_b - sum_a) / n)
    deltas.sort()

    alpha = 1.0 - confidence
    lo = _percentile(deltas, alpha / 2 * 100)
    hi = _percentile(deltas, (1 - alpha / 2) * 100)
    return (delta, lo, hi)


def mcnemar_exact(a: list[int], b: list[int]) -> dict:
    """Exact McNemar (binomial sign) test on paired binary outcomes.

    Only the discordant pairs carry information: ``a_only`` = A correct & B
    wrong, ``b_only`` = A wrong & B correct. Two-sided exact p-value from the
    binomial with p=0.5. Robust at the small discordant counts typical of a
    40-question eval, where the chi-square approximation is unreliable.
    """
    if len(a) != len(b):
        raise ValueError("a and b must have equal length (paired)")

    a_only = sum(1 for x, y in zip(a, b) if x and not y)
    b_only = sum(1 for x, y in zip(a, b) if y and not x)
    n = a_only + b_only

    if n == 0:
        p_value = 1.0
    else:
        k = min(a_only, b_only)
        tail = sum(math.comb(n, i) for i in range(k + 1))
        p_value = min(1.0, 2.0 * tail * (0.5**n))

    return {"a_only": a_only, "b_only": b_only, "n_discordant": n, "p_value": p_value}


def compare_models(
    a_by_qid: dict[str, float],
    b_by_qid: dict[str, float],
    confidence: float = 0.95,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 42,
) -> dict:
    """Compare two models paired on their common question ids (delta = B - A).

    Returns accuracies, the bootstrap delta CI, and the exact McNemar p-value
    with a ``significant`` flag (p < 1 - confidence).
    """
    common = sorted(set(a_by_qid) & set(b_by_qid))
    result: dict = {"n_common": len(common)}

    if not common:
        result.update(
            acc_a=None, acc_b=None, delta=None,
            delta_ci=None, p_value=None, significant=False,
        )
        return result

    a = [float(a_by_qid[q]) for q in common]
    b = [float(b_by_qid[q]) for q in common]

    delta, lo, hi = paired_bootstrap_delta_ci(a, b, confidence, n_resamples, seed)
    # Outcomes are strictly 1.0/0.0 (strict-CORRECT accuracy). The rounding keeps
    # McNemar binary; if outcomes ever carry partial credit in (0,1), the CI
    # (raw floats) and this test would diverge — binarize deliberately upstream then.
    mc = mcnemar_exact([int(round(x)) for x in a], [int(round(x)) for x in b])
    alpha = 1.0 - confidence

    result.update(
        acc_a=_mean(a),
        acc_b=_mean(b),
        delta=delta,
        delta_ci=(lo, hi),
        p_value=mc["p_value"],
        significant=mc["p_value"] < alpha,
    )
    return result
