import keras
import tensorflow as tf
import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
# from multiprocessing import Process
# from time import sleep


import SimpleOscillator


VERBOSE_PRINTOUT_FOR_OSCILLATORS = False

ODE_INSTANCES_BEFORE_LEARNING_STARTS = 100

STATE_DIMENSIONS = 2  # position, velocity
ACTION_DIMENSIONS = 1  # force
EXPERIENCE_DIMENSIONS = STATE_DIMENSIONS*2 + ACTION_DIMENSIONS + 1  # state 0, action, state 1, reward

DYN_INPUT_DIMENSIONS = STATE_DIMENSIONS + ACTION_DIMENSIONS
DYN_OUTPUT_DIMENSIONS = STATE_DIMENSIONS
DYN_HIDDEN_LAYERS_SIZES = [20, 20]
DYN_DROPOUT_RATE = 0.1

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
    if np.abs(state[0]) > 200 or np.abs(state[1] > 200):
        print("ODE was very unstable, terminating after {:.2f} seconds".format(t))
        is_terminal = True
        reward = -1000
    return reward, is_terminal


def singleODEInstance():
    # create the oscillator that will generate experiences
    osci = SimpleOscillator.SimpleOscillator(k=OSCILLATOR_K, c=OSCILLATOR_C, goal=50, g=OSCILLATOR_G, max_force=100, control_hz=5, policy="random_lazy_pid")

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
            old_state = osci.getPreviousState()
            action = osci.getPreviousAction()
            new_state = list(osci.getState())
            experience = [old_state[0], old_state[1], action, new_state[0], new_state[1], reward]
            # print("Generated experience: {}".format(experience))
            osci.setPrevious()  # set previous state and action to current state and action
            experiences.append(experience)

            if VERBOSE_PRINTOUT_FOR_OSCILLATORS:
                print_visual = "-"*200
                index = int(osci.getState()[0])+100
                print_visual = print_visual[:index] + "X" + print_visual[index + 1:]
                print(print_visual)

        # ODE simulator forward in time by ODE_TIME_STEP seconds -- ASSUMES THIS IS LESS TIME THAN INTERVAL BETWEEN CONTROL ACTIONS!!!
        times = np.linspace(osci_time, osci_time + ODE_TIME_STEP, 10)
        states = spi.odeint(SimpleOscillator.simpleOscillatorODE, osci.getState(), times, (osci,))
        osci.setState(states[-1])
        osci_time += ODE_TIME_STEP
    if not terminal:
        print("ODE oscillator did not reach terminal state in {} seconds, terminating".format(ODE_DEADLINE))
    return experiences


def singleNNInstance(dyn_model):
    osci = SimpleOscillator.SimpleOscillator(k=OSCILLATOR_K, c=OSCILLATOR_C, goal=50, g=OSCILLATOR_G, max_force=100, control_hz=5, policy="pid")
    position_time_history = list()
    print("Running a single instance of the dynamics NN oscillator")
    osci_time = 0  # simulation elapsed time in seconds
    terminal = False
    osci.setPrevious()
    while not terminal and osci_time < ODE_DEADLINE:
        osci.getAction(osci_time)  # we don't check if action is updated, because by definition it MUST be b/c of how we use the NN

        reward, terminal = generateReward(osci.getState(), osci.getGoal(), osci_time)

        old_state = osci.getPreviousState()
        action = osci.getPreviousAction()

        position_time_history.append([osci_time, old_state[0]])

        dyn_input = np.reshape(np.array([old_state[0], old_state[1], action]), (1, DYN_INPUT_DIMENSIONS))

        osci.setPrevious()

        dyn_output = dyn_model.predict(dyn_input)

        osci.setState(dyn_output[0])

        osci_time += 1/5  ###################################### TODO: time steps must coincide with the timestep used to generate the NN inputs!!!!! i.e. 1/control_hz

        #print_visual = "-" * 200
        #index = int(osci.getState()[0]) + 100
        #print_visual = print_visual[:index] + "X" + print_visual[index + 1:]
        #print(print_visual)

    position_time_history.append([osci_time, old_state[0]])
    position_time_history = np.array(position_time_history)
    plt.plot(position_time_history[:, 0], position_time_history[:, 1])
    plt.show()



def randomExperienceReplay(experiences, size_of_subset=None):
    if size_of_subset is None:
        size_of_subset = int(len(experiences)/10)
        if size_of_subset < 1:
            size_of_subset = 1
    indices = list(np.random.randint(low=0, high=len(experiences), size=size_of_subset))
    return [experiences[i] for i in indices]


def randomDynamicsReplay(experiences, size_of_subset=None):
    experiences_subset = np.array(randomExperienceReplay(experiences, size_of_subset=size_of_subset))
    dyn_batch_inputs = experiences_subset[:, 0:DYN_INPUT_DIMENSIONS]
    dyn_batch_outputs = experiences_subset[:, DYN_INPUT_DIMENSIONS:-1]
    return dyn_batch_inputs, dyn_batch_outputs


def main(sess=None):
    if sess is None:
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

    experiences = list()
    for _ in range(ODE_INSTANCES_BEFORE_LEARNING_STARTS):
        experiences.extend(singleODEInstance())


    # TODO: do we need to normalize (mean = 0, std.dev. = 1) the experiences?

    # TODO: setup a simple Relu network representing state+action->new state transitions
    dyn_model = keras.Sequential()

    n = 1
    dyn_model.add(keras.layers.Dense(DYN_HIDDEN_LAYERS_SIZES[0], activation="relu", name="dyn_hidden_1", input_shape=(DYN_INPUT_DIMENSIONS,)))
    dyn_model.add(keras.layers.Dropout(DYN_DROPOUT_RATE, name="dyn_dropout_1"))

    for dense_layer_size in DYN_HIDDEN_LAYERS_SIZES[1:]:
        n += 1
        dyn_model.add(keras.layers.Dense(dense_layer_size, activation="relu", name="dyn_hidden_" + str(n)))
        dyn_model.add(keras.layers.Dropout(DYN_DROPOUT_RATE, name="dyn_dropout_" + str(n)))

    dyn_model.add(keras.layers.Dense(DYN_OUTPUT_DIMENSIONS, activation="linear", name="dyn_output"))
    dyn_model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam())

    # TODO: minibatch learn the experiences subset
    # alternate learning the dynamics model via experience replay and generating more experiences
    dyn_batch_inputs, dyn_batch_outputs = randomDynamicsReplay(experiences, len(experiences))
    dyn_model.fit(dyn_batch_inputs, dyn_batch_outputs, epochs=10, verbose=True)

    # TODO: after X minibatches have been learned...
    # TODO:     1) generate dropout samples of the NN
    # TODO:     2) perform a physics rollout, replacing the ODE integration with the NN dropout samples (timestep = control Hz timestep, or ODE timestep???)
    # TODO:     3) visually compare the ODE integration vs. population of NN dropout sample rollouts - MUST USE FIXED POLICY FOR ALL ROLLOUTS!!!

    singleNNInstance(dyn_model)

    return


if __name__ == "__main__":
    main()