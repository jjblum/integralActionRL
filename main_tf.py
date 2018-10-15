import keras
import tensorflow as tf
import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
# from multiprocessing import Process
# from time import sleep

import SimpleOscillator

DEBUG_PLOTS_FOR_OSCILLATORS = False
VERBOSE_PRINTOUT_FOR_OSCILLATORS = False
DEBUG_EXPERIENCES_SCATTER_PLOT = False

ODE_INSTANCES_BEFORE_LEARNING_STARTS = 100

STATE_DIMENSIONS = 2  # position, velocity
ACTION_DIMENSIONS = 1  # force
EXPERIENCE_DIMENSIONS = STATE_DIMENSIONS*2 + ACTION_DIMENSIONS + 1  # state 0, action, state 1, reward

DYN_INPUT_DIMENSIONS = STATE_DIMENSIONS + ACTION_DIMENSIONS
DYN_OUTPUT_DIMENSIONS = STATE_DIMENSIONS
DYN_HIDDEN_LAYERS_SIZES = [20, 20]
DYN_HIDDEN_ACTIVATIONS = ["linear", "linear"]
DYN_DROPOUT_RATE = 0.15
EPOCHS = 40
NORMALIZE_EXPERIENCES = False
DROPOUT_SAMPLE_COUNT = 0  # can be set to 0 to skip calculating any dropout samples

ODE_TIME_STEP = 0.1  # seconds
ODE_DEADLINE = 300  # seconds before ODE is prematurely terminated
ODE_DISTANCE_BETWEEN_EXPERIENCES = 0.1  # we want diverse training, so we resample a time history to avoid too many very similar experiences

OSCILLATOR_M = 1
OSCILLATOR_K = 0.1
OSCILLATOR_C = 0.001
OSCILLATOR_G = 0

TEST_GOAL = 25
TEST_FINAL_ACTION = OSCILLATOR_K*TEST_GOAL - OSCILLATOR_G
TEST_FINAL_VELOCITY = 0

PLOT_INCLUDE_VELOCITY = True
PLOT_INCLUDE_ACTION = True


def layerIndexFromLayerName(model, layer_name):
    index = None
    for i, layer in enumerate(model.layers):
        if layer.name == layer_name:
            index = i
            break
    return index


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


def createDynamicModelInput(position, velocity, action):
    # TODO: generalize this so it isn't just position, velocity, and a single action
    return np.reshape(np.array([position, velocity, action]), (1, DYN_INPUT_DIMENSIONS))


def singleODEInstance(goal=None):
    if goal is None:
        goal = TEST_GOAL
    # create the oscillator that will generate experiences
    osci = SimpleOscillator.SimpleOscillator(m=OSCILLATOR_M, k=OSCILLATOR_K, c=OSCILLATOR_C, goal=goal, g=OSCILLATOR_G, max_force=100, control_hz=5, policy="random_pid")

    # create the dynamic size array for experiences
    experiences = list()

    # run a single instance of the oscillator, storing state transitions in the experiences
    # print("Running a single instance of the ODE oscillator")
    osci_time = 0  # ODE simulation elapsed time in seconds
    terminal = False
    osci.setPrevious()
    while not terminal and osci_time < ODE_DEADLINE:
        action_updated = osci.getAction(osci_time)
        if action_updated:
            # collate the experience
            reward, terminal = generateReward(osci, osci_time)
            old_state = osci.getPreviousState()
            old_action = osci.getPreviousAction()
            new_state = list(osci.getState())
            experience = [old_state[0], old_state[1], old_action, new_state[0]-old_state[0], new_state[1]-old_state[1], reward]
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
        # print("ODE oscillator did not reach terminal state in {} seconds, terminating".format(ODE_DEADLINE))
        None

    # TODO: resample so redundant transitions are not included
    # resampling here means, viewing a complete ODE time history as a sequence of experiences, skip any experience that is too much like the previously retained experience
    experiences_array = np.array(experiences)
    interesting_experiences = [experiences_array[0]]
    for i in range(len(experiences)):
        d = np.linalg.norm(experiences_array[i, :2] - interesting_experiences[-1][:2])
        if d > ODE_DISTANCE_BETWEEN_EXPERIENCES:
            interesting_experiences.append(experiences_array[i])
            # print("{} ADD".format(d))
        else:
            None
            # print("{} skip".format(d))
    #interesting_experiences.append(experiences_array[-1])

    if DEBUG_PLOTS_FOR_OSCILLATORS:
        fig, ax1 = plt.subplots()
        plt.title("ODE result, goal = {}".format(goal))
        ax1.plot(np.array(interesting_experiences)[:, 0], 'rx-')
        ax1.grid(True)
        ax1.set_ylabel('dyn NN position', color="r")
        ax1.tick_params('y', colors='r')

        ax2 = ax1.twinx()
        ax2.plot(np.array(interesting_experiences)[:, 1], 'bx-')
        ax2.set_ylabel('dyn NN velocity', color="b")
        ax2.tick_params('y', colors='b')
        ax2.grid(True)
        plt.show()

    return [a.tolist() for a in interesting_experiences]


def singleNNInstance(dyn_model, dyn_inputs_mean=None, dyn_inputs_stddev=None, dyn_outputs_mean=None, dyn_outputs_stddev=None):
    osci = SimpleOscillator.SimpleOscillator(m=OSCILLATOR_M, k=OSCILLATOR_K, c=OSCILLATOR_C, goal=TEST_GOAL, g=OSCILLATOR_G, max_force=100, control_hz=5, policy="pid")
    state_time_history = list()
    action_time_history = list()
    print("Running a single instance of the dynamics NN oscillator")
    osci_time = 0
    terminal = False
    while not terminal and osci_time < ODE_DEADLINE:
        osci.setPrevious()  # setPrevious sets previousState and previousAction to current state and action

        action_updated = osci.getAction(osci_time, forced_to_take_action=True)  # take action using current state

        if not (osci_time == 0 or action_updated):
            print("t = {} didn't take an action!".format(osci_time))

        goal = osci.getGoal()
        reward, terminal = generateReward(osci, osci_time)  # uses current state

        old_state = osci.getPreviousState()
        old_action = osci.getPreviousAction()

        state_time_history.append([osci_time, old_state[0], old_state[1]])
        action_time_history.append([osci_time, old_action])

        dyn_input = np.reshape(np.array([old_state[0], old_state[1], old_action]), (1, DYN_INPUT_DIMENSIONS))

        if NORMALIZE_EXPERIENCES:
            dyn_input -= np.full(dyn_input.shape, dyn_inputs_mean)
            dyn_input /= np.full(dyn_input.shape, dyn_inputs_stddev)

        dyn_output = dyn_model.predict(dyn_input)

        if NORMALIZE_EXPERIENCES:
            dyn_output *= np.full(dyn_output.shape, dyn_outputs_stddev)
            dyn_output += np.full(dyn_output.shape, dyn_outputs_mean)

        new_state = old_state + dyn_output[0]
        osci.setState(new_state)

        osci_time += 1/5  ###################################### time steps must coincide with the timestep used to generate the NN inputs!!!!! i.e. 1/control_hz

    #state_time_history.append([osci_time, old_state[0], old_state[1]])
    #action_time_history.append([osci_time, action])
    state_time_history = np.array(state_time_history)
    action_time_history = np.array(action_time_history)
    return state_time_history, action_time_history

    """
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
    ax1.plot([0, state_time_history[-1, 0]], [TEST_GOAL, TEST_GOAL], 'r--')
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('dyn NN position', color="r")
    ax1.tick_params('y', colors='r')
    plt.title("{}, {}".format(DYN_HIDDEN_LAYERS_SIZES, DYN_HIDDEN_ACTIVATIONS))
    plt.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(state_time_history[:, 0], state_time_history[:, 2], 'b')
    ax2.plot([0, state_time_history[-1, 0]], [TEST_FINAL_VELOCITY, TEST_FINAL_VELOCITY], 'b--')
    #ax2.get_yaxis().set_visible(False)
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('dyn NN velocity', color="b")
    ax2.tick_params('y', colors='b')
    plt.grid(True)

    ax3 = ax1.twinx()
    ax3.plot(action_time_history[:, 0], action_time_history[:, 1], 'g')
    ax3.plot([0, action_time_history[-1, 0]], [TEST_FINAL_ACTION, TEST_FINAL_ACTION], 'g--')
    #ax3.get_yaxis().set_visible(False)
    ax3.set_xlabel('time (s)')
    ax3.set_ylabel('\ndyn NN action', color="g")
    ax3.tick_params('y', colors='g')
    plt.grid(True)

    plt.show()
    """


def visualizeActionAroundGoal(dyn_model, goal=TEST_GOAL, n=100):
    # NEW IDEA: what if I, after training, apply the NN with (goal, 0, linspace(-100, 100, 100))
    # i.e. apply the NN with the goal state and a range of actions to see what it predicts for
    # the range of possible actions.
    # Plot the result, x axis is range of actions, y axis 1 is next position, y axis 2 is next velocity
    # Maybe that way I can see where the prediction is accurate and where it is inaccurate
    actions = np.linspace(-100, 100, n)
    actions = np.atleast_2d(actions)
    actions = actions.T
    dyn_input = np.hstack((np.full((n, 2), fill_value=[goal, 0]), actions))
    actions = np.squeeze(actions)
    dyn_output = dyn_model.predict(dyn_input)

    fig, ax1 = plt.subplots()
    ax1.plot(actions, dyn_output[:, 0], 'r')
    ax1.set_xlabel("action around goal state [{:.2f}, 0]".format(goal))
    ax1.set_ylabel("predicted position", color='r')
    ax1.tick_params('y', colors='r')
    plt.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(actions, dyn_output[:, 1], 'b')
    #ax2.set_xlabel("action around goal state [{:.2f}, 0]".format(goal))
    ax2.set_ylabel("predicted velocity", color='b')
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


def pidBootstrap():
    experiences = list()
    for _ in range(ODE_INSTANCES_BEFORE_LEARNING_STARTS):
        experiences.extend(singleODEInstance(-50 + 100 * np.random.rand()))
        # experiences.extend(singleODEInstance(15 + 20*np.random.rand()))

    if DEBUG_EXPERIENCES_SCATTER_PLOT:
        experiences_array = np.array(experiences)
        # given position (x) and velocity (y), what was the following velocity (color scale)
        sp = plt.scatter(experiences_array[:, 0], experiences_array[:, 1], c=experiences_array[:, 4])
        plt.colorbar(sp)
        plt.title("pos. + vel. --> change in velocity")
        plt.xlabel("position")
        plt.ylabel("velocity")
        # plt.show()

    # TODO: do we need to normalize (mean = 0, std.dev. = 1) the experiences? Or use batch normalization layers?

    # setup a simple network representing state+action->new state transitions
    dyn_model = keras.Sequential()

    n = 1
    dyn_model.add(
        keras.layers.Dense(DYN_HIDDEN_LAYERS_SIZES[0], activation=DYN_HIDDEN_ACTIVATIONS[0], name="dyn_hidden_1",
                           input_shape=(DYN_INPUT_DIMENSIONS,)))
    # dyn_model.add(keras.layers.BatchNormalization(name="dyn_batchnorm_1")) # TODO: batch normalization?
    dyn_model.add(keras.layers.Dropout(DYN_DROPOUT_RATE, name="dyn_dropout_1"))

    for dense_layer_size in DYN_HIDDEN_LAYERS_SIZES[1:]:
        n += 1
        dyn_model.add(
            keras.layers.Dense(dense_layer_size, activation=DYN_HIDDEN_ACTIVATIONS[n - 1], name="dyn_hidden_" + str(n)))
        # dyn_model.add(keras.layers.BatchNormalization(name="dyn_batchnorm_" + str(n)))  # TODO: batch normalization?
        dyn_model.add(keras.layers.Dropout(DYN_DROPOUT_RATE, name="dyn_dropout_" + str(n)))

    dyn_model.add(keras.layers.Dense(DYN_OUTPUT_DIMENSIONS, activation="linear", name="dyn_output"))
    # dyn_model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam())
    # dyn_model.compile(loss="mean_absolute_error", optimizer=keras.optimizers.Adam())  # TODO: try different losses
    dyn_model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Nadam())



    # alternate learning the dynamics model via experience replay and generating more experiences
    dyn_batch_inputs, dyn_batch_outputs = randomDynamicsReplay(experiences, len(experiences))

    dyn_batch_inputs_mean = None
    dyn_batch_inputs_stddev = None
    dyn_batch_outputs_mean = None
    dyn_batch_outputs_stddev = None
    if NORMALIZE_EXPERIENCES:
        dyn_batch_inputs_mean = np.mean(dyn_batch_inputs, axis=0)
        dyn_batch_inputs_stddev = np.std(dyn_batch_inputs, axis=0)
        dyn_batch_inputs -= np.full(dyn_batch_inputs.shape, dyn_batch_inputs_mean)
        dyn_batch_inputs /= np.full(dyn_batch_inputs.shape, dyn_batch_inputs_stddev)

        dyn_batch_outputs_mean = np.mean(dyn_batch_outputs, axis=0)
        dyn_batch_outputs_stddev = np.std(dyn_batch_outputs, axis=0)
        dyn_batch_outputs -= np.full(dyn_batch_outputs.shape, dyn_batch_outputs_mean)
        dyn_batch_outputs /= np.full(dyn_batch_outputs.shape, dyn_batch_outputs_stddev)

    dyn_model.fit(dyn_batch_inputs, dyn_batch_outputs, epochs=EPOCHS, verbose=2, shuffle=True, validation_split=0.01)#, batch_size=128)

    # after X minibatches have been learned...
    #   1) generate dropout samples of the NN
    #          a) extract the weights from the model
    #          b) generate random dropout masks
    #          c) apply the masks to copies of the weights and create new models with these weights
    #      2) perform a physics rollout, replacing the ODE integration with the NN dropout samples (timestep = control Hz timestep, or ODE timestep???)
    #      3) visually compare the ODE integration vs. population of NN dropout sample rollouts - MUST USE FIXED POLICY FOR ALL ROLLOUTS!!!

    ideal_state_time_history, ideal_action_time_history = singleNNInstance(dyn_model, dyn_batch_inputs_mean, dyn_batch_inputs_stddev, dyn_batch_outputs_mean, dyn_batch_outputs_stddev)
    final_position = ideal_state_time_history[-1, 1]
    final_velocity = ideal_state_time_history[-1, 2]
    final_action = ideal_action_time_history[-1, 1]

    # TODO: make a single dropout sample model and in the loop just change its weights. Maybe it is slow due to the overhead of making 100 Keras models.
    # TODO: if you parallelize, you need each thread/process to make its own root sample model and modify it each iteration

    dropout_model = keras.Sequential()
    n = 1
    dropout_model.add(
        keras.layers.Dense(DYN_HIDDEN_LAYERS_SIZES[0], activation=DYN_HIDDEN_ACTIVATIONS[0], name="dyn_hidden_1",
                           input_shape=(DYN_INPUT_DIMENSIONS,)))
    # dropout_model.add(keras.layers.BatchNormalization(name="dyn_batchnorm_1"))  # TODO: batch normalization?
    for dense_layer_size in DYN_HIDDEN_LAYERS_SIZES[1:]:
        n += 1
        dropout_model.add(
            keras.layers.Dense(dense_layer_size, activation=DYN_HIDDEN_ACTIVATIONS[n - 1], name="dyn_hidden_" + str(n)))
        # dropout_model.add(keras.layers.BatchNormalization(name="dyn_batchnorm_" + str(n)))  # TODO: batch normalization?
    dropout_model.add(keras.layers.Dense(DYN_OUTPUT_DIMENSIONS, activation="linear", name="dyn_output"))

    state_time_history_samples = list()
    action_time_history_samples = list()
    for sample_id in range(DROPOUT_SAMPLE_COUNT):
        print("Performing dropout sample rollout #{}".format(sample_id + 1))
        for i in range(len(dyn_model.layers)):  # loop through the original NN's layers
            layer = dyn_model.layers[i]
            # If "dropout" or "batchnorm" is in the layer name, skip.
            # print(layer.name)
            if "dropout" in layer.name or "batchnorm" in layer.name:
                # print("skipping the {} layer".format(layer.name))
                continue

            layer_index = layerIndexFromLayerName(dropout_model, layer.name)  # find the matching layer

            weights_and_biases = layer.get_weights()  # extract the weights and biases from the model
            weights = weights_and_biases[0]
            biases = weights_and_biases[1]
            weights_mask = np.random.binomial(1, 1 - DYN_DROPOUT_RATE, size=weights.shape) / (1 - DYN_DROPOUT_RATE)
            masked_weights = np.multiply(weights_mask, weights)  # apply the dropout mask to the weights
            masked_weights_and_biases = [masked_weights, biases]
            dropout_model.layers[layer_index].set_weights(
                masked_weights_and_biases)  # manually apply the masked weights

        # do not need to compile the models for feedforward-only use
        s, a = singleNNInstance(dropout_model, dyn_batch_inputs_mean, dyn_batch_inputs_stddev, dyn_batch_outputs_mean, dyn_batch_outputs_stddev)
        s = s[:ideal_state_time_history.shape[0], :]  # trim the time histories so that their length matches the ideal's length
        a = a[:ideal_state_time_history.shape[0], :]
        state_time_history_samples.append(s)
        action_time_history_samples.append(a)

    """
    dropout_output_population = np.zeros(shape=(DROPOUT_SAMPLE_COUNT, DYN_OUTPUT_DIMENSIONS))
    for i in range(DROPOUT_SAMPLE_COUNT):
        dropout_model = dropout_models[i]
        dyn_input = createDynamicModelInput(TEST_GOAL, TEST_FINAL_VELOCITY, TEST_FINAL_ACTION)  # should result in [TEST_GOAL, TEST_FINAL_VELOCITY]
        dyn_output = dropout_model.predict(dyn_input)
        dropout_output_population[i, :] = dyn_output
    # print(dropout_output_population)
    print("mean = {}".format(np.mean(dropout_output_population, axis=0)))
    print("stdev = {}".format(np.std(dropout_output_population, axis=0)))

    # visualizeActionAroundGoal(dyn_model)

    if NORMALIZE_EXPERIENCES:
        singleNNInstance(dyn_model, dyn_batch_inputs_mean, dyn_batch_inputs_stddev, dyn_batch_outputs_mean, dyn_batch_outputs_stddev)
    else:
        singleNNInstance(dyn_model)

    """

    print("\tFinal position: {:.2f}\n\tFinal velocity: {:.2f}\n\tFinal action: {:.2f}".format(
        final_position, final_velocity, final_action))

    # plot ideal and samples
    fig, ax1 = plt.subplots()
    plt.title("{}, {}".format(DYN_HIDDEN_LAYERS_SIZES, DYN_HIDDEN_ACTIVATIONS))
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('dyn NN position', color="r")
    ax1.tick_params('y', colors='r')
    plt.grid(True)
    if PLOT_INCLUDE_VELOCITY:
        ax2 = ax1.twinx()
        ax2.set_xlabel('time (s)')
        ax2.set_ylabel('dyn NN velocity', color="b")
        ax2.tick_params('y', colors='b')
    if PLOT_INCLUDE_ACTION:
        ax3 = ax1.twinx()
        ax3.set_xlabel('time (s)')
        ax3.set_ylabel('\ndyn NN action', color="g")
        ax3.tick_params('y', colors='g')

    for i in range(DROPOUT_SAMPLE_COUNT):
        ax1.plot(state_time_history_samples[i][:, 0], state_time_history_samples[i][:, 1], 'r', linewidth=1.0,
                 alpha=0.5)
        if PLOT_INCLUDE_VELOCITY:
            ax2.plot(state_time_history_samples[i][:, 0], state_time_history_samples[i][:, 2], 'b', linewidth=1.0,
                     alpha=0.5)
        if PLOT_INCLUDE_ACTION:
            ax3.plot(action_time_history_samples[i][:, 0], action_time_history_samples[i][:, 1], 'g', linewidth=1.0,
                     alpha=0.5)

    if PLOT_INCLUDE_ACTION:
        ax3.plot([0, ideal_action_time_history[-1, 0]], [TEST_FINAL_ACTION, TEST_FINAL_ACTION], 'g--', linewidth=4.0)
        ax3.plot(ideal_action_time_history[:, 0], ideal_action_time_history[:, 1], 'g', linewidth=4.0)

    if PLOT_INCLUDE_VELOCITY:
        ax2.plot([0, ideal_state_time_history[-1, 0]], [TEST_FINAL_VELOCITY, TEST_FINAL_VELOCITY], 'b--', linewidth=2.0)
        ax2.plot(ideal_state_time_history[:, 0], ideal_state_time_history[:, 2], 'b', linewidth=4.0)

    ax1.plot([0, ideal_state_time_history[-1, 0]], [TEST_GOAL, TEST_GOAL], 'r--', linewidth=4.0)
    ax1.plot(ideal_state_time_history[:, 0], ideal_state_time_history[:, 1], 'r', linewidth=4.0)

    plt.show()

    """

    ax2.plot(state_time_history[:, 0], state_time_history[:, 2], 'b')
    ax2.plot([0, state_time_history[-1, 0]], [TEST_FINAL_VELOCITY, TEST_FINAL_VELOCITY], 'b--')
    # ax2.get_yaxis().set_visible(False)

    plt.grid(True)


    ax3.plot(action_time_history[:, 0], action_time_history[:, 1], 'g')
    ax3.plot([0, action_time_history[-1, 0]], [TEST_FINAL_ACTION, TEST_FINAL_ACTION], 'g--')
    # ax3.get_yaxis().set_visible(False)
    plt.grid(True)

    plt.show()
    """




def main(sess=None):
    if sess is None:
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

    pidBootstrap()
    # singleODEInstance(-50 + 100*np.random.rand())  # TODO: PID doesn't work very well when there is gravity, it takes a long time for integral action to accumulate

    return


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 22})
    main()

    # TODO:  MAJOR IDEA TO AVOID TOO MANY EXPERIENCES CLUSTERED AROUND THE ZERO (WHERE THE OSCILLATOR SLOWS WAY DOWN)
    # TODO:  resample! can't resample based on time...
    # TODO:  but could resample based on state? e.g. only retain an experience if it is sufficiently far from the previous one
    # TODO:  that way, you won't have a *huge* number of equillibrium transitions