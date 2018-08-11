import tensorflow as tf
import numpy as np
import scipy.integrate as spi
# from multiprocessing import Process
# from time import sleep


import SimpleOscillator


ODE_INSTANCES_BEFORE_LEARNING_STARTS = 10


STATE_DIMENSIONS = 2  # position, velocity
ACTION_DIMENSIONS = 1  # force
EXPERIENCE_DIMENSIONS = STATE_DIMENSIONS*2 + ACTION_DIMENSIONS + 1  # state 0, action, state 1, reward

ODE_TIME_STEP = 0.1  # seconds
ODE_DEADLINE = 100  # seconds before ODE is prematurely terminated

OSCILLATOR_K = 0.1
OSCILLATOR_C = 0.0001
OSCILLATOR_G = 0


def generateReward(state, goal, t):
    is_terminal = False
    reward = 0
    if np.abs(state[0] - goal) < 1 and np.abs(state[1]) < 1:
        print("ODE oscillator reached terminal state in {:.2f} seconds".format(t))
        is_terminal = True
        reward = 1000
    return reward, is_terminal


def singleODEInstance():
    # create the oscillator that will generate experiences
    osci = SimpleOscillator.SimpleOscillator(k=OSCILLATOR_K, c=OSCILLATOR_C, goal=100, g=OSCILLATOR_G, max_force=100, control_hz=5, policy="random_pid")

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
            # print("Generated experience: {}".format(experience))
            osci.setPrevious()  # set previous state and action to current state and action
            experiences.append(experience)

            print_visual = "-"*200
            index = int(osci.getState()[0])
            print_visual = print_visual[:index] + "X" + print_visual[index + 1:]
            print(print_visual)



        # ODE simulator forward in time by ODE_TIME_STEP seconds -- ASSUMES THIS IS LESS THAN INTERVAL BETWEEN CONTROL ACTIONS!!!
        times = np.linspace(osci_time, osci_time + ODE_TIME_STEP, 10)
        states = spi.odeint(SimpleOscillator.simpleOscillatorODE, osci.getState(), times, (osci,))
        osci.setState(states[-1])
        osci_time += ODE_TIME_STEP
    if not terminal:
        print("ODE oscillator did not reach terminal state in {} seconds, terminating".format(ODE_DEADLINE))
    return experiences


def randomExperienceReplay(experiences, size_of_subset=None):
    if size_of_subset is None:
        size_of_subset = int(len(experiences)/10)
        if size_of_subset < 1:
            size_of_subset = 1
    indices = list(np.random.randint(low=0, high=len(experiences), size=size_of_subset))
    return [experiences[i] for i in indices]


def main(sess=None):
    global EXPERIENCE_DIMENSIONS
    if sess is None:
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

    experiences = list()
    for _ in range(ODE_INSTANCES_BEFORE_LEARNING_STARTS):
        experiences.extend(singleODEInstance())

    # alternate learning the dynamics model via experience replay and generating more experiences
    experiences_subset = randomExperienceReplay(experiences)


    return


if __name__ == "__main__":
    main()