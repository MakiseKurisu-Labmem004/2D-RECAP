import numpy as np
import pytest

from recap.difficulty import difficulty_quotas


RATIOS = dict(easy=.60, medium=.30, hard=.10)


@pytest.mark.parametrize('count, expected', [
    (200, [120, 60, 20]), (50, [30, 15, 5]), (30, [18, 9, 3]),
    (10, [6, 3, 1]), (4, [3, 1, 0]), (2, [1, 1, 0]),
    (1, [1, 0, 0]), (0, [0, 0, 0]),
])
def test_exact_difficulty_quotas(count, expected):
    quotas = difficulty_quotas(count, RATIOS)
    assert list(quotas.values()) == expected
    assert sum(quotas.values()) == count


def test_zero_ratio_and_equal_remainders():
    assert difficulty_quotas(3, dict(easy=.5, medium=.5, hard=0)) == dict(easy=2, medium=1, hard=0)


@pytest.mark.parametrize('ratios', [
    dict(easy=.6, medium=.3),
    dict(easy=.6, medium=.3, hard=.1, extra=0),
    dict(easy=.6, medium=.3, hard=.2),
    dict(easy=0, medium=0, hard=0),
    dict(easy=-.1, medium=.9, hard=.2),
    dict(easy=np.nan, medium=.9, hard=.1),
    dict(easy=np.inf, medium=.9, hard=.1),
])
def test_invalid_difficulty_ratios(ratios):
    with pytest.raises(ValueError):
        difficulty_quotas(10, ratios)


@pytest.mark.parametrize('count', [-1, 1.5])
def test_invalid_scene_count(count):
    with pytest.raises(ValueError, match='Scene count'):
        difficulty_quotas(count, RATIOS)
