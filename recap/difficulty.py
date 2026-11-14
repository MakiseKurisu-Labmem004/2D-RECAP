"""Convert requested difficulty proportions into exact scene counts."""

from fractions import Fraction
import math
from numbers import Integral


DIFFICULTIES = ('easy', 'medium', 'hard')


def difficulty_quotas(count, ratios):
    if not isinstance(count, Integral) or count < 0:
        raise ValueError('Scene count must be a nonnegative integer')
    if set(ratios) != set(DIFFICULTIES):
        raise ValueError('difficulty_ratio must contain exactly easy, medium, and hard')
    values = [float(ratios[name]) for name in DIFFICULTIES]
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError('difficulty_ratio values must be finite and nonnegative')
    if not math.isclose(math.fsum(values), 1., rel_tol=0., abs_tol=1e-8):
        raise ValueError('difficulty_ratio values must sum to 1')
    # Decimal fractions give equal remainders equal priority despite float rounding.
    weights = {name: Fraction(str(ratios[name])) for name in DIFFICULTIES}
    total = sum(weights.values())
    targets = {name: count * weight / total for name, weight in weights.items()}
    quotas = {name: math.floor(target) for name, target in targets.items()}
    order = sorted(DIFFICULTIES, key=lambda name: targets[name] - quotas[name], reverse=True)
    for name in order[:count - sum(quotas.values())]:
        quotas[name] += 1
    return quotas
