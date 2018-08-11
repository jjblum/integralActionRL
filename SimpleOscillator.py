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
    def __init__(self, m=1, k=1, c=0, g=0, goal=1, control_hz=10, max_force=100, policy="pid", name="pid", initial_state=(0, 0)):
        self._t = 0.0  # time
        self._t_last_action_update = 0.0  # time of last control update
        self._m = m  # mass
        self._k = k  # spring constant
        self._c = c  # damping constant
        self._g = g  # bias force (e.g. gravity)
        self._f = 0  # current action force
        self._state = list(initial_state)  # state [x, xdot]
        self._goal = [goal, 0]  # the goal state
        self._control_hz = control_hz
        self._name = name
        self._max_force = max_force  # maximum possible action
        self._initial_state = list(initial_state)
        self._previous_state = list(initial_state)
        self._previous_action = 0
        if policy == "pid":
            self._policy = Policy.Policy_PID(0.01, 0.001, 0.01, self._goal, self._t, "pid")
        elif policy == "random_pid":
            self._policy = Policy.Policy_RandomlyGeneratedPID(self._goal, self._t, "random_pid")

    def getPhysics(self):
        return self._m, self._k, self._c, self._g, self._f

    def setGoal(self, goal):
        self._goal = [goal, 0]
        self._policy.setGoal(self._goal)

    def getGoal(self):
        return self._goal[0]

    def getAction(self, t):
        """
        Given current state, return policy's recommended action.
        Returns true if an action was evaluated (i.e. control Hz), false if action was not evaluated
        """
        self._t = t
        # print("oscillator {} t = {}".format(self._name, self._t))
        if (t - self._t_last_action_update) >= 1/self._control_hz or t == 0:
            # policy returns value between -1 and 1, a relative effort, scale by max force
            self._f = self._max_force*self._policy.getAction(self._state, t)
            self._t_last_action_update = t
            if t == 0:
                self.setPrevious()
            if t != 0:
                return True
        return False

    def getPreviousAction(self):
        return self._previous_action

    def getState(self):
        return self._state

    def getPreviousState(self):
        return self._previous_state

    def setState(self, state):
        self._state = state

    def setPrevious(self):
        self._previous_state = self.getState()
        self._previous_action = self._f

    def getName(self):
        return self._name

