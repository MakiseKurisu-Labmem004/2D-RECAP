"""Global C3 minimum-snap joint splines with conservative curve clearance checks."""

import math
import numpy as np

from .geometry import clearance


class UnsafeSplineError(ValueError):
    pass


def derivative(coefficients, order=1):
    result = np.asarray(coefficients)
    for _ in range(order):
        result = result[1:] * np.arange(1, len(result))[:, None]
    return result


def evaluate(coefficients, u):
    return np.polynomial.polynomial.polyval(u, coefficients)


def power_to_bezier(coefficients):
    degree = len(coefficients)-1
    return np.array([sum(coefficients[k] * math.comb(i, k) / math.comb(degree, k)
                         for k in range(i+1)) for i in range(degree+1)])


def split_bezier(control):
    row = np.asarray(control)
    left, right = [row[0]], [row[-1]]
    while len(row) > 1:
        row = (row[:-1] + row[1:]) / 2
        left.append(row[0]); right.append(row[-1])
    return np.array(left), np.array(right[::-1])


def derivative_bound(control, depth=4):
    if depth == 0:
        return np.max(abs(control))
    left, right = split_bezier(control)
    return max(derivative_bound(left, depth-1), derivative_bound(right, depth-1))


def curve_clear(control, scene, cfg, depth=0):
    left, right = split_bezier(control)
    middle = left[-1]
    limit, margin = cfg['env']['joint_limit'], cfg['env']['margin']
    if np.any(abs(middle) > limit+1e-10):
        return False
    gap = clearance(middle, scene, cfg['env']['link_radius']).min()
    if gap <= margin: return False
    radius = np.max(abs(control-middle), axis=0)
    # Any point on either 0.5 m link moves by at most |dq1| + 0.5|dq2|.
    if np.all(abs(control) <= limit+1e-10) and gap > margin + radius[0] + .5*radius[1]:
        return True
    if depth >= 16:
        return False
    return curve_clear(left, scene, cfg, depth+1) and curve_clear(right, scene, cfg, depth+1)


def curve_length(control, tolerance=1e-5, depth=0):
    lower = np.linalg.norm(control[-1]-control[0])
    upper = np.linalg.norm(np.diff(control, axis=0), axis=1).sum()
    if upper-lower <= tolerance or depth >= 14:
        return float((lower+upper)/2)
    left, right = split_bezier(control)
    return curve_length(left, tolerance/2, depth+1) + curve_length(right, tolerance/2, depth+1)


def hermite_matrix():
    matrix = np.zeros((8, 8))
    for end in (0, 1):
        for order in range(4):
            for power in range(order, 8):
                matrix[end*4+order, power] = math.factorial(power)/math.factorial(power-order) * end**(power-order)
    return np.linalg.inv(matrix)


HERMITE = hermite_matrix()
SNAP = np.zeros((8, 8))
for i in range(4, 8):
    for j in range(4, 8):
        SNAP[i, j] = math.factorial(i)/math.factorial(i-4) * math.factorial(j)/math.factorial(j-4) / (i+j-7)
SEPTIC = np.array([0., 0., 0., 0., 35., -84., 70., -20.])[:, None]
SEPTIC_BOUNDS = [derivative_bound(power_to_bezier(derivative(SEPTIC, order))) for order in (1, 2, 3)]


def minimum_snap(path, spans):
    count = max(0, len(path)-2)*3
    hessian = np.zeros((count, count))
    gradient = np.zeros((count, 2))
    pieces = []
    for index, span in enumerate(spans):
        fixed = np.zeros((8, 2)); fixed[0] = path[index]; fixed[4] = path[index+1]
        free = np.zeros((8, count))
        for side, knot in ((0, index), (1, index+1)):
            if 0 < knot < len(path)-1:
                for order in range(1, 4):
                    free[side*4+order, (knot-1)*3+order-1] = span**order
        base, mapping = HERMITE @ fixed, HERMITE @ free
        weight = SNAP / span**7
        hessian += mapping.T @ weight @ mapping
        gradient += mapping.T @ weight @ base
        pieces.append((base, mapping))
    if count:
        scale = np.sqrt(np.maximum(np.diag(hessian), 1e-20))
        normalized = hessian / scale[:, None] / scale[None, :]
        derivatives = np.linalg.solve(normalized + np.eye(count)*1e-12, -gradient/scale[:, None]) / scale[:, None]
    else:
        derivatives = np.empty((0, 2))
    return pieces, derivatives


class JointSpline:
    def __init__(self, path, cfg, scene):
        path = np.asarray(path, dtype=float)
        if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2 or not np.isfinite(path).all():
            raise ValueError('Spline path must contain at least two finite joint poses')
        keep = np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-10]
        path = path[keep]
        if len(path) == 1: path = np.vstack((path, path))
        planner = cfg['planner']
        limits = (planner.get('expert_max_velocity', .35), planner.get('expert_max_acceleration', 2.),
                  planner.get('expert_max_jerk', 10.))
        if not all(np.isfinite(value) and value > 0 for value in limits):
            raise ValueError('Expert velocity, acceleration and jerk must be positive and finite')
        dt = cfg['env']['timestep']*cfg['env']['substeps']
        lengths = np.max(abs(np.diff(path, axis=0)), axis=1)
        # Allocate acceleration/jerk time to short segments too; distance-only
        # allocation would let a tiny corner force the entire path to crawl.
        spans = np.maximum(dt, np.max(np.array([(bound*lengths/limit)**(1/order)
                           for order, (bound, limit) in enumerate(zip(SEPTIC_BOUNDS, limits), 1)]), axis=0))
        spans /= spans.sum()
        pieces, derivatives = minimum_snap(path, spans)
        self.blend = None
        for blend in (1., .5, .25, .125, .0625, 0.):
            coefficients = np.array([base + blend * mapping @ derivatives for base, mapping in pieces])
            controls = [power_to_bezier(c) for c in coefficients]
            if all(curve_clear(control, scene, cfg) for control in controls):
                self.blend = blend
                break
        if self.blend is None:
            raise UnsafeSplineError('No C3 spline satisfies joint limits and full-link clearance')
        self.coefficients = coefficients
        self.spans = spans
        self.knots = np.r_[0., np.cumsum(spans)]
        self.path = path
        self.length = sum(curve_length(control) for control in controls)
        duration = dt
        for order, limit in enumerate(limits, 1):
            bound = max(derivative_bound(power_to_bezier(derivative(c, order))) / span**order
                        for c, span in zip(coefficients, spans))
            duration = max(duration, (bound/limit)**(1/order))
        self.duration = np.ceil(duration/dt)*dt

    def reference(self, time, order=0):
        s = float(np.clip(time/self.duration, 0., 1.))
        index = min(np.searchsorted(self.knots, s, side='right')-1, len(self.spans)-1)
        index = max(0, index)
        u = np.clip((s-self.knots[index])/self.spans[index], 0., 1.)
        return evaluate(derivative(self.coefficients[index], order), u) / (self.duration*self.spans[index])**order
