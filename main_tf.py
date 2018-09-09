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
DYN_HIDDEN_LAYERS_SIZES = [100]
DYN_DROPOUT_RATE = 0.25
ACTIVATION = "relu"
EPOCHS = 10
NORMALIZE_EXPERIENCES = False

ODE_TIME_STEP = 0.1  # seconds
ODE_DEADLINE = 150  # seconds before ODE is prematurely terminated

OSCILLATOR_M = 1
OSCILLATOR_K = 0.1
OSCILLATOR_C = 0.001
OSCILLATOR_G = 20


def generateReward(oscillator, t):
    is_terminal = False
    reward = 0

    goal = oscillator.getGoal()
    state = oscillator.getState()
    m, k, c, g, f = oscillator.getPhysics()

    kinetic_energy = 0.5*m*np.power(state[1], 2)
    potential_energy = 0.5*k*np.power(goal, 2) + OSCILLATOR_M*np.abs(g)
    total_energy = kinetic_energy + potential_energy
    terminal_total_energy = 0.5*k*np.power(goal, 2) + m*np.abs(g)  # potential energy only

    if np.abs(total_energy - terminal_total_energy) < 0.1 and np.abs(state[0] - goal) < 0.1:
        print("oscillator reached terminal state [{:.2f}, {:.2f}] in {:.2f} seconds".format(state[0], state[1], t))
        is_terminal = True
        reward = 1000

    if np.abs(state[0]) > 200 or np.abs(state[1]) > 200:
        print("oscillator was very unstable, terminating after {:.2f} seconds".format(t))
        is_terminal = True
        reward = -1000

    return reward, is_terminal


def singleODEInstance(goal=None):
    if goal is None:
        goal = 100*np.random.rand()
    # create the oscillator that will generate experiences
    osci = SimpleOscillator.SimpleOscillator(m=OSCILLATOR_M, k=OSCILLATOR_K, c=OSCILLATOR_C, goal=goal, g=OSCILLATOR_G, max_force=100, control_hz=5, policy="random_pid")

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
            reward, terminal = generateReward(osci, osci_time)
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


def singleNNInstance(dyn_model, dyn_inputs_mean=None, dyn_inputs_stddev=None, dyn_outputs_mean=None, dyn_outputs_stddev=None):
    osci = SimpleOscillator.SimpleOscillator(m=OSCILLATOR_M, k=OSCILLATOR_K, c=OSCILLATOR_C, goal=25, g=OSCILLATOR_G, max_force=100, control_hz=5, policy="pid")
    state_time_history = list()
    action_time_history = list()
    print("Running a single instance of the dynamics NN oscillator")
    osci_time = 0
    terminal = False
    osci.setPrevious()
    while not terminal and osci_time < ODE_DEADLINE:
        action_updated = osci.getAction(osci_time, forced_to_take_action=True)

        if not (osci_time == 0 or action_updated):
            print("t = {} didn't take an action!".format(osci_time))

        goal = osci.getGoal()
        reward, terminal = generateReward(osci, osci_time)

        old_state = osci.getPreviousState()
        action = osci.getPreviousAction()

        state_time_history.append([osci_time, old_state[0], old_state[1]])
        action_time_history.append([osci_time, action])

        dyn_input = np.reshape(np.array([old_state[0], old_state[1], action]), (1, DYN_INPUT_DIMENSIONS))

        if NORMALIZE_EXPERIENCES:
            dyn_input -= np.full(dyn_input.shape, dyn_inputs_mean)
            dyn_input /= np.full(dyn_input.shape, dyn_inputs_stddev)

        osci.setPrevious()

        dyn_output = dyn_model.predict(dyn_input)

        if NORMALIZE_EXPERIENCES:
            dyn_output *= np.full(dyn_output.shape, dyn_outputs_stddev)
            dyn_output += np.full(dyn_output.shape, dyn_outputs_mean)

        new_state = dyn_output[0]
        osci.setState(new_state)

        osci_time += 1/5  ###################################### time steps must coincide with the timestep used to generate the NN inputs!!!!! i.e. 1/control_hz

    #state_time_history.append([osci_time, old_state[0], old_state[1]])
    #action_time_history.append([osci_time, action])
    state_time_history = np.array(state_time_history)
    action_time_history = np.array(action_time_history)

    final_position = state_time_history[-1, 1]
    final_velocity = state_time_history[-1, 2]
    final_action = action_time_history[-1, 1]
    m, k, c, g, f = osci.getPhysics()
    kinetic_energy = 0.5*m*np.power(final_velocity, 2)
    potential_energy = 0.5*k*np.power(osci.getGoal(), 2) + OSCILLATOR_M*np.abs(g)
    total_energy = kinetic_energy + potential_energy
    terminal_total_energy = 0.5*k*np.power(osci.getGoal(), 2) + m*np.abs(g)  # potential energy only
    print("\tFinal position: {:.2f}\n\tFinal velocity: {:.2f}\n\tFinal action: {:.2f}, Final energy gap: {:.2f}".format(
        final_position, final_velocity, final_action, terminal_total_energy - total_energy))
    mean_position = np.mean(state_time_history[:, 1])
    mean_velocity = np.mean(state_time_history[:, 2])
    mean_action = np.mean(action_time_history[:, 1])
    print("\tMean position: {:.2f}\n\tMean velocity: {:.2f}\n\tMean action: {:.2f}".format(mean_position, mean_velocity, mean_action))

    fig, ax1 = plt.subplots()

    ax1.plot(state_time_history[:, 0], state_time_history[:, 1], 'r')
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('dyn NN position', color="r")
    ax1.tick_params('y', colors='r')

    ax2 = ax1.twinx()
    ax2.plot(state_time_history[:, 0], state_time_history[:, 2], 'b')
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('dyn NN velocity', color="b")
    ax2.tick_params('y', colors='b')

    ax3 = ax1.twinx()
    ax3.plot(action_time_history[:, 0], action_time_history[:, 1], 'g')
    ax3.set_xlabel('time (s)')
    ax3.set_ylabel('\ndyn NN action', color="g")
    ax3.tick_params('y', colors='g')

    plt.show()

    # TODO: if we remove derivative control and just look at simple P control (with some damping in the ODE), the dyn NN reproduces new positions well
    # TODO:     and it produces something for velocity that ooks fine at first. But notie that even though it converges, the velocity converges to
    # TODO:     something nonzero. This is impossible - the position oscillates very closely to 50 effecively converged, yet the steady state velocity is 10 units/second???
    # TODO:     How could the NN begin to think that? The actions are small when position converges, and the NN has been provided data that says velocity is small when
    # TODO:     current velocity is small and action is low, right?
    # TODO:     I just flat out do not understand why the NN is saying the velocity oscillates like you expect, but oscillates around a nonzero value you wouldn't

def visualizeActionAroundGoal(dyn_model, goal=25, n=100):
    # TODO:     NEW IDEA: what if I, after training, apply the NN with (goal, 0, linspace(-100, 100, 100))
    # TODO:     i.e. apply the NN with the goal state and a range of actions to see what it predicts for
    # TODO:     the range of possible actions.
    # TODO:     Plot the result, x axis is range of actions, y axis 1 is next position, y axis 2 is next velocity
    # TODO:     Maybe that way I can see where the prediction is accurate and where it is inaccurate
    actions = np.linspace(-100, 100, n)
    actions = np.atleast_2d(actions)
    actions = actions.T
    dyn_input = np.hstack((np.full((n, 2), fill_value=[goal, 0]), actions))
    actions = np.squeeze(actions)
    dyn_output = dyn_model.predict(dyn_input)

    fig, ax1 = plt.subplots()
    ax1.plot(actions, dyn_output[:, 0], 'r')
    ax1.set_xlabel("action around goal state [{:.2f}, 0]".format(goal))
    ax1.set_ylabel("predicted position", color = 'r')
    ax1.tick_params('y', colors='r')
    plt.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(actions, dyn_output[:, 1], 'b')
    #ax2.set_xlabel("action around goal state [{:.2f}, 0]".format(goal))
    ax2.set_ylabel("predicted velocity", color = 'b')
    ax2.tick_params('y', colors='b')

    plt.grid(True)
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
    dyn_model.add(keras.layers.Dense(DYN_HIDDEN_LAYERS_SIZES[0], activation=ACTIVATION, name="dyn_hidden_1", input_shape=(DYN_INPUT_DIMENSIONS,)))
    dyn_model.add(keras.layers.Dropout(DYN_DROPOUT_RATE, name="dyn_dropout_1"))

    for dense_layer_size in DYN_HIDDEN_LAYERS_SIZES[1:]:
        n += 1
        dyn_model.add(keras.layers.Dense(dense_layer_size, activation=ACTIVATION, name="dyn_hidden_" + str(n)))
        dyn_model.add(keras.layers.Dropout(DYN_DROPOUT_RATE, name="dyn_dropout_" + str(n)))

    dyn_model.add(keras.layers.Dense(DYN_OUTPUT_DIMENSIONS, activation="linear", name="dyn_output"))
    dyn_model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam())

    # TODO: minibatch learn the experiences subset
    # alternate learning the dynamics model via experience replay and generating more experiences
    dyn_batch_inputs, dyn_batch_outputs = randomDynamicsReplay(experiences, len(experiences))

    if NORMALIZE_EXPERIENCES:
        dyn_batch_inputs_mean = np.mean(dyn_batch_inputs, axis=0)
        dyn_batch_inputs_stddev = np.std(dyn_batch_inputs, axis=0)
        dyn_batch_inputs -= np.full(dyn_batch_inputs.shape, dyn_batch_inputs_mean)
        dyn_batch_inputs /= np.full(dyn_batch_inputs.shape, dyn_batch_inputs_stddev)

        dyn_batch_outputs_mean = np.mean(dyn_batch_outputs, axis=0)
        dyn_batch_outputs_stddev = np.std(dyn_batch_outputs, axis=0)
        dyn_batch_outputs -= np.full(dyn_batch_outputs.shape, dyn_batch_outputs_mean)
        dyn_batch_outputs /= np.full(dyn_batch_outputs.shape, dyn_batch_outputs_stddev)

    dyn_model.fit(dyn_batch_inputs, dyn_batch_outputs, epochs=EPOCHS, verbose=True)

    # TODO: after X minibatches have been learned...
    # TODO:     1) generate dropout samples of the NN
    # TODO:     2) perform a physics rollout, replacing the ODE integration with the NN dropout samples (timestep = control Hz timestep, or ODE timestep???)
    # TODO:     3) visually compare the ODE integration vs. population of NN dropout sample rollouts - MUST USE FIXED POLICY FOR ALL ROLLOUTS!!!

    # visualizeActionAroundGoal(dyn_model)

    if NORMALIZE_EXPERIENCES:
        singleNNInstance(dyn_model, dyn_batch_inputs_mean, dyn_batch_inputs_stddev, dyn_batch_outputs_mean, dyn_batch_outputs_stddev)
    else:
        singleNNInstance(dyn_model)

    # TODO: could the nonzero velocity be because position is larger than velocity? Maybe that dominates the signal because I'm usign Relu units.

    return


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 22})
    main()