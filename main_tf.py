import tensorflow as tf
import numpy as np
import scipy.integrate as spi
# from multiprocessing import Process
# from time import sleep


import SimpleOscillator

STATE_DIMENSIONS = 2  # position, velocity
ACTION_DIMENSIONS = 1  # force
EXPERIENCE_DIMENSIONS = STATE_DIMENSIONS*2 + ACTION_DIMENSIONS + 1  # state 0, action, state 1, reward

ODE_TIME_STEP = 0.1  # seconds
ODE_DEADLINE = 100  # seconds before ODE is prematurely terminated


def generateReward(state, goal, t):
    is_terminal = False
    reward = 0
    if np.abs(state[0] - goal) < 1 and np.abs(state[1]) < 1:
        print("ODE oscillator reached terminal state in {:.2f} seconds".format(t))
        is_terminal = True
        reward = 1000
    return reward, is_terminal


def main(sess=None):
    global EXPERIENCE_DIMENSIONS
    if sess is None:
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

    # create the oscillator that will generate experiences
    osci = SimpleOscillator.SimpleOscillator(k=0.1, c=0.1, goal=100, g=0, max_force=100, control_hz=5)

    # create the dynamic size array for experiences
    experiences = list()

    # run a single instance of the oscillator, storing state transitions in the experiences
    print("Running a single instance of the ODE oscillator")
    osci_time = 0  # ODE simulation elapsed time in seconds
    terminal = False
    osci.setPrevious()
    while not terminal and osci_time < ODE_DEADLINE:
        action_updated = osci.getAction(osci_time)
        if action_updated:
            # collate the experience
            reward, terminal = generateReward(osci.getState(), osci.getGoal(), osci_time)
            experience = (osci.getPreviousState(), osci.getPreviousAction(), osci.getState(), reward)
            print("Generated experience: {}".format(experience))
            osci.setPrevious()  # set previous state and action to current state and action
            experiences.append(experience)

        # ODE simulator forward in time by ODE_TIME_STEP seconds -- ASSUMES THIS IS LESS THAN INTERVAL BETWEEN CONTROL ACTIONS!!!
        times = np.linspace(osci_time, osci_time + ODE_TIME_STEP, 10)
        states = spi.odeint(SimpleOscillator.simpleOscillatorODE, osci.getState(), times, (osci,))
        osci.setState(states[-1])
        osci_time += ODE_TIME_STEP
    if not terminal:
        print("ODE oscillator did not reach terminal state in {} seconds, terminating".format(ODE_DEADLINE))

    return


if __name__ == "__main__":
    main()