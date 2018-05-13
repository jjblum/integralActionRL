import numpy as np

import Policy

"""
A simple physics simulation of a 1-D oscillator with a gravity-like environmental force.
This force is not a universal constant like gravity, nor is it random - it is meant to be a long-term bias.
If the learner learns a policy with one bias, then it changes to another, can the learner cope?
Perhaps we can learn integral effects to overcome bias so that a single learner can deal with changing bias.
"""


def simpleOscillatorODE(starting_state, starting_time, oscillator):
    x = starting_state[0]
    v = starting_state[1]
    m, k, c, g, f = oscillator.getPhysics()
    qdot = np.zeros(starting_state.shape)
    qdot[0] = v
    qdot[1] = 1/m*(-k*x - c*v + g + f)
    return qdot


class SimpleOscillator:
    def __init__(self, m=1, k=1, c=0, g=0, goal=1, control_hz=10, max_force=100, policy="pid", name="pid"):
        self._t = 0.0  # time
        self._t_last_action_update = 0.0  # time of last control update
        self._m = m  # mass
        self._k = k  # spring constant
        self._c = c  # damping constant
        self._g = g  # bias force (e.g. gravity)
        self._f = 0  # current action force
        self._state = [0, 0]  # state [x, xdot]
        self._goal = [goal, 0]  # the goal state
        self._control_hz = control_hz
        self._name = name
        self._max_force = max_force  # maximum possible action
        if policy == "pid":
            self._policy = Policy.Policy_PID(1, 0.1, 0.2, self._goal, self._t, "pid")

    def getPhysics(self):
        return self._m, self._k, self._c, self._g, self._f

    def setGoal(self, goal):
        self._goal = [goal, 0]
        self._policy.setGoal(self._goal)

    def getGoal(self):
        return self._goal[0]

    def getAction(self, t):
        """
        Given current state, return policy's recommended action
        """
        self._t = t
        if (t - self._t_last_action_update) >= 1/self._control_hz or t == 0:
            self._f = 0.0*self._f + 1.0*self._max_force*self._policy.getAction(self._state, t)  # policy returns value between -1 and 1, a relative effort
            self._t_last_action_update = t

    def getLastAction(self):
        return self._f

    def getState(self):
        return self._state

    def setState(self, state):
        self._state = state

    def getName(self):
        return self._name

