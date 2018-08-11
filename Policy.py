import numpy as np


class UniversalPID:
    def __init__(self, P, I, D, t, name):
        self._P = P
        self._I = I
        self._D = D
        self._t = t
        self._tOld = t
        self._errorDerivative = 0.0
        self._errorAccumulation = 0.0
        self._errorOld = 0.0
        self._name = name

    def signal(self, error, t):
        dt = t - self._t
        self._t = t
        self._errorDerivative = 0.0
        if dt > 0:
            self._errorDerivative = (error - self._errorOld)/dt
        self._errorAccumulation += dt*error
        self._errorOld = error
        return self._P*error + self._I*self._errorAccumulation + self._D*self._errorDerivative


class Policy_PID:
    def __init__(self, p, i, d, goal, t, name):
        self._name = name
        self._goal = goal
        self._pid = UniversalPID(p, i, d, t, name)

    def setGoal(self, goal):
        self._goal = goal

    def getAction(self, state, t):
        error = np.array(self._goal) - np.array(state)
        signal = self._pid.signal(error[0], t)
        if signal > 1:
            signal = 1
        if signal < -1:
            signal = -1
        return signal


class Policy_NoAction:
    def __init__(self, name="NoAction"):
        self._name = name

    def setGoal(self, goal):
        return

    def getAction(self, state, t):
        return 0


class Policy_RandomlyGeneratedPID:
    def __init__(self, goal, t, name):
        self._name = name
        self._goal = goal
        p = np.random.rand()*0.01
        i = np.random.rand()*0.002
        d = np.random.rand()*0.01
        self._pid = UniversalPID(p, i, d, t, name)

    def setGoal(self, goal):
        self._goal = goal

    def getAction(self, state, t):
        error = np.array(self._goal) - np.array(state)
        signal = self._pid.signal(error[0], t)
        if signal > 1:
            signal = 1
        if signal < -1:
            signal = -1
        return signal