"""Planar arm geometry shared by simulation and trajectory visualization."""

import numpy as np


def points(q):
    elbow = .5 * np.array([np.cos(q[0]), np.sin(q[0])])
    return np.array([[0., 0.], elbow, elbow + .5 * np.array([np.cos(sum(q)), np.sin(sum(q))])])
