"""Exact integration of a causal C3 actuator-command filter."""

import math
import numpy as np


class C3CommandFilter:
    def __init__(self, position, tau):
        if not np.isfinite(tau) or tau <= 0:
            raise ValueError('control_filter_tau must be positive and finite')
        position = np.asarray(position, dtype=float)
        if position.shape != (2,) or not np.isfinite(position).all():
            raise ValueError('Filter initial position must be finite with shape (2,)')
        self.tau = tau
        self.state = np.tile(position, (4, 1))

    def advance(self, target, dt):
        # Four cascaded first-order filters: the output and derivatives 1..3
        # remain continuous when the input target changes. No Euler instability.
        target = np.asarray(target, dtype=float)
        if target.shape != (2,) or not np.isfinite(target).all() or not np.isfinite(dt) or dt < 0:
            raise ValueError('Filter target must be finite with shape (2,), and dt finite and nonnegative')
        z = self.state - target
        x = dt/self.tau
        if x > 1000:
            self.state[:] = target
            return target.copy()
        self.state = target + np.exp(-x) * np.array([
            sum(x**k/math.factorial(k)*z[i-k] for k in range(i+1)) for i in range(4)])
        return self.state[-1].copy()

    def derivatives(self):
        x1, x2, x3, x4 = self.state
        return (x4.copy(), (x3-x4)/self.tau, (x2-2*x3+x4)/self.tau**2,
                (x1-3*x2+3*x3-x4)/self.tau**3)
