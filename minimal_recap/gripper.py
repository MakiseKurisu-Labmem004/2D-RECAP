"""A physical-substep opening reference governor; never edits simulation state."""
import math


class GripperMotion:
    """Limits total-gap reference velocity/acceleration, not contact velocities.

    The braking response time prevents reaching a travel limit at full speed.
    The reference may lag a reversing command; it is part of resumable state.
    """
    def __init__(self, config, opening, maximum, dt):
        self.enabled = config is not None
        self.maximum, self.dt = maximum, dt
        self.target = self.reference = float(opening)
        self.velocity = 0.
        if self.enabled:
            self.speed = float(config['max_speed'])
            self.acceleration = float(config['max_acceleration'])
            self.response = float(config['response_time'])
            if not all(math.isfinite(x) and x > 0 for x in [self.speed, self.acceleration, self.response]):
                raise ValueError('Gripper motion limits must be positive and finite')
            if self.response < self.speed/self.acceleration + dt:
                raise ValueError('gripper response_time must be >= max_speed/max_acceleration + timestep')

    def command(self, opening):
        self.target = min(self.maximum, max(0., float(opening)))
        if not self.enabled:
            self.reference, self.velocity = self.target, 0.

    def advance(self):
        if self.enabled:
            desired = max(-self.speed, min(self.speed, (self.target-self.reference)/self.response))
            dv = self.acceleration*self.dt
            self.velocity += max(-dv, min(dv, desired-self.velocity))
            self.reference += self.velocity*self.dt
            # Properly initialized references brake before the physical limits.
            # Do not silently clip a bad/restored controller state.
            if not -1e-12 <= self.reference <= self.maximum+1e-12:
                raise RuntimeError('Gripper reference left its travel range')
        return self.reference

    def snapshot(self):
        return dict(target=self.target, reference=self.reference, velocity=self.velocity)

    def restore(self, state):
        self.target = float(state['target'])
        self.reference = float(state['reference'])
        self.velocity = float(state['velocity'])
