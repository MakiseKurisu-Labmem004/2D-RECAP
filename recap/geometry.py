"""Planar arm geometry shared by simulation and trajectory visualization."""

import numpy as np


def points(q):
    elbow = .5 * np.array([np.cos(q[0]), np.sin(q[0])])
    return np.array([[0., 0.], elbow, elbow + .5 * np.array([np.cos(sum(q)), np.sin(sum(q))])])


def clearance(q, scene, radius):
    p = points(q)
    distances = []
    for a, b in zip(p[:-1], p[1:]):
        u = np.clip(np.dot(np.asarray(scene['obstacle'])-a, b-a)/np.dot(b-a, b-a), 0, 1)
        distances.append(np.linalg.norm(a+u*(b-a)-scene['obstacle'])-scene['radius']-radius)
    return np.asarray(distances)
