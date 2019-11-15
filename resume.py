# Author: Tony Wu, OEI, HuaZhong University of Science and Technology
# This is the main file which calls all the functions and trains the network by updating weights
# The core algorithm is ReSuMe
# The test data: XOR problem

import numpy as np
import algorithm_reference
from neuron import Neuron
from matplotlib import pyplot as plt
from parameters import Param
import os
import pickle
import sys


def resume_process(learning_or_test):

    # Display the current state, leaning or test
    print(learning_or_test)
    if learning_or_test == 0:
        print("Starting test XOR problem...")
    elif learning_or_test == 1:
        print("Starting learning XOR problem...")
    else:
        print("Error in argument, plz enter 0 or 1 as argument, quitting")
        quit()

    # Before training, define training epoch
    training_epoch = Param.epoch
    if learning_or_test == 0:
        training_epoch = 1

    # Before training, define the objective output spike train
    objective_output_train = []
    fired_time_objective = []

    objective_output_train_zero_zero = np.zeros(Param.T)
    objective_output_train_zero_zero[Param.output_zero_delay] = 1
    fired_time_objective_zero_zero = Param.output_zero_delay
    objective_output_train.append(objective_output_train_zero_zero)
    fired_time_objective.append(fired_time_objective_zero_zero)

    objective_output_train_zero_one = np.zeros(Param.T)
    objective_output_train_zero_one[Param.output_one_delay] = 1
    fired_time_objective_zero_one = Param.output_one_delay
    objective_output_train.append(objective_output_train_zero_one)
    fired_time_objective.append(fired_time_objective_zero_one)

    objective_output_train_one_zero = np.zeros(Param.T)
    objective_output_train_one_zero[Param.output_one_delay] = 1
    fired_time_objective_one_zero = Param.output_one_delay
    objective_output_train.append(objective_output_train_one_zero)
    fired_time_objective.append(fired_time_objective_one_zero)

    objective_output_train_one_one = np.zeros(Param.T)
    objective_output_train_one_one[Param.output_zero_delay] = 1
    fired_time_objective_one_one = Param.output_zero_delay
    objective_output_train.append(objective_output_train_one_one)
    fired_time_objective.append(fired_time_objective_one_one)

    # Initialize the network
    network = []  # network[i] refer to the i layer of the network
    current_layer = []
    network.append(current_layer)  # layer 0 is empty(no neuron, just fire)

    current_layer = []
    for i in range(Param.num_hidden_layer):  # hidden layer
        new_neuron = Neuron()
        current_layer.append(new_neuron)
    network.append(current_layer)

    current_layer = []
    for i in range(Param.num_output_layer):  # output layer
        new_neuron = Neuron()
        current_layer.append(new_neuron)
    network.append(current_layer)

    # Initialize the Weights of synapses, define that row, col, sub
    w_synapses = []  # w_synapses[0] refer to the connection between input and hidden; w_synapses[1]...
    w_layer = np.zeros((Param.num_hidden_layer, Param.num_input_layer, Param.sub_connections))  # w_synapses[0]

    for i in range(Param.num_hidden_layer):
        for j in range(Param.num_input_layer):
            for sub in range(Param.sub_connections):
                w_layer[i][j][sub] = np.random.uniform(Param.w_random_min, Param.w_random_max)
    w_synapses.append(w_layer)

    w_layer = np.zeros((Param.num_output_layer, Param.num_hidden_layer, Param.sub_connections))  # w_synapses[1]
    for i in range(Param.num_output_layer):
        for j in range(Param.num_hidden_layer):
            for sub in range(Param.sub_connections):
                w_layer[i][j][sub] = np.random.uniform(Param.w_random_min, Param.w_random_max)
    w_synapses.append(w_layer)
    # print(w_synapses)

    # Initialize the potential Value of synapses, which stores the malues
    value_synapses = []  # w_synapses[0] refer to the connection between input and hidden; w_synapses[1]...
    value_layer = np.zeros((Param.num_hidden_layer, Param.num_input_layer, Param.sub_connections))  # w_synapses[0]

    for i in range(Param.num_hidden_layer):
        for j in range(Param.num_input_layer):
            for sub in range(Param.sub_connections):
                value_layer[i][j][sub] = 0.0
    value_synapses.append(value_layer)

    value_layer = np.zeros((Param.num_output_layer, Param.num_hidden_layer, Param.sub_connections))  # w_synapses[1]
    for i in range(Param.num_output_layer):
        for j in range(Param.num_hidden_layer):
            for sub in range(Param.sub_connections):
                value_layer[i][j][sub] = 0.0
    value_synapses.append(value_layer)

    # Initialize the delay of synapses, define that row, col, sub
    delay_synapses = []  # delay_synapses[0] refer to the connection between input and hidden; delay_synapses[1]...
    delay_layer = np.zeros((Param.num_hidden_layer, Param.num_input_layer, Param.sub_connections))  # delay_synapses[0]

    for i in range(Param.num_hidden_layer):
        for j in range(Param.num_input_layer):
            for sub in range(Param.sub_connections):
                delay_layer[i][j][sub] = np.random.randint(Param.sub_connections_delay_min,
                                                           Param.sub_connections_delay_max, dtype=int)
    delay_synapses.append(delay_layer)

    delay_layer = np.zeros((Param.num_output_layer, Param.num_hidden_layer, Param.sub_connections))  # delay_synapses[1]
    for i in range(Param.num_output_layer):
        for j in range(Param.num_hidden_layer):
            for sub in range(Param.sub_connections):
                delay_layer[i][j][sub] = np.random.randint(Param.sub_connections_delay_min,
                                                           Param.sub_connections_delay_max, dtype=int)
    delay_synapses.append(delay_layer)

    # Define the time axis, T = Param.T
    time_axis = Param.T

    # Define the fired delay of the neuron in input layer
    # In the future, it may be changed with array
    fired_first_delay = 0
    fired_second_delay = 0

    # Define the spike train of the input layer
    spike_train_input = []
    spike_neuron = np.zeros(Param.T)
    spike_train_input.append(spike_neuron)
    spike_neuron = np.zeros(Param.T)
    spike_train_input.append(spike_neuron)

    # Define the spike train of the hidden layer
    spike_train_hidden = []
    for j in range(Param.num_hidden_layer):
        spike_neuron = np.zeros(Param.T)
        spike_train_hidden.append(spike_neuron)

    # Define the spike train of the output layer
    spike_train_output = []
    spike_neuron = np.zeros(Param.T)
    spike_train_output.append(spike_neuron)

    recent_fired_time_input = []
    for i in range(Param.num_input_layer):
        recent_fired_time_input.append(Param.T)
    recent_fired_time_hidden = []
    for j in range(Param.num_hidden_layer):
        recent_fired_time_hidden.append(Param.T)
    recent_fired_time_output = Param.T

    # Define the "fired time" matrix(statically stores the fired time of each neuron)
    fired_time_input = []   # input layer
    for i in range(Param.num_input_layer):
        temp_input = []
        fired_time_input.append(temp_input)

    fired_time_hidden = []   # hidden layer
    for j in range(Param.num_hidden_layer):
        temp_input = []
        fired_time_hidden.append(temp_input)

    fired_time_output = []   # output layer
    for j in range(Param.num_output_layer):
        temp_input = []
        fired_time_output.append(temp_input)

    for steps in range(training_epoch):
        for training_stage in range(3, 4):
            # input [0,0],[0,1],[1,0],[1,1] in order
            if training_stage == 0:
                fired_first_delay = Param.input_zero_delay
                fired_second_delay = Param.input_zero_delay
            if training_stage == 1:
                fired_first_delay = Param.input_zero_delay
                fired_second_delay = Param.input_one_delay
            if training_stage == 2:
                fired_first_delay = Param.input_one_delay
                fired_second_delay = Param.input_zero_delay
            if training_stage == 3:
                fired_first_delay = Param.input_one_delay
                fired_second_delay = Param.input_one_delay

            # every time begin a new type of train, clear all the train array!
            spike_train_input = []
            spike_neuron = np.zeros(Param.T)
            spike_train_input.append(spike_neuron)
            spike_neuron = np.zeros(Param.T)
            spike_train_input.append(spike_neuron)

            spike_train_hidden = []
            for j in range(Param.num_hidden_layer):
                spike_neuron = np.zeros(Param.T)
                spike_train_hidden.append(spike_neuron)

            spike_train_output = []
            spike_neuron = np.zeros(Param.T)
            spike_train_output.append(spike_neuron)

            recent_fired_time_input = []
            for i in range(Param.num_input_layer):
                recent_fired_time_input.append(Param.T)
            recent_fired_time_hidden = []
            for j in range(Param.num_hidden_layer):
                recent_fired_time_hidden.append(Param.T)
            recent_fired_time_output = Param.T

            fired_time_input = []  # input layer
            for i in range(Param.num_input_layer):
                temp_input = []
                fired_time_input.append(temp_input)

            fired_time_hidden = []  # hidden layer
            for j in range(Param.num_hidden_layer):
                temp_input = []
                fired_time_hidden.append(temp_input)

            fired_time_output = []  # output layer
            for j in range(Param.num_output_layer):
                temp_input = []
                fired_time_output.append(temp_input)

            for i in range(Param.num_input_layer):
                for j in range(Param.num_hidden_layer):
                    for k in range(Param.sub_connections):
                        value_synapses[0][j][i][k] = 0
            for j in range(Param.num_hidden_layer):
                for o in range(Param.num_output_layer):
                    for k in range(Param.sub_connections):
                        value_synapses[1][o][j][k] = 0

            # initialize spikes in input layer
            spike_train_input[0][fired_first_delay] = 1
            spike_train_input[1][fired_second_delay] = 1

            finished = 0

            for t in range(time_axis):
                if finished == 1:
                    break
                # update the potential of neurons in hidden layer
                for i in range(Param.num_input_layer):
                    if spike_train_input[i][t] == 1:
                        fired_time_input[i].append(t)
                        recent_fired_time_input[i] = t
                        # if training_stage == 3:
                        #     print(t)
                    for j in range(Param.num_hidden_layer):
                        network[1][j].Potential = 0
                        for k in range(Param.sub_connections):
                            x = t - recent_fired_time_input[i] - delay_synapses[0][j][i][k]
                            if x >= 0:
                                # if (training_stage == 0) & (i == 0) & (j == 0) & (k == 0):
                                #     print(x)
                                # network[1][j].Potential += w_synapses[0][j][i][k] * algorithm_reference.\
                                #     spike_response_function(x, Param.t_spike_function)
                                # if (training_stage == 0) & (i == 0) & (j == 0) & (k == 0):
                                #     print(network[1][j].Potential)
                                value_synapses[0][j][i][k] = w_synapses[0][j][i][k] * algorithm_reference.\
                                    spike_response_function(x, Param.t_spike_function)
                                # if (training_stage == 0) & (i == 1) & (j == 3) & (k == 2):
                                #     print(value_synapses[0][j][i][k])
                for i in range(Param.num_input_layer):
                    for j in range(Param.num_hidden_layer):
                        for k in range(Param.sub_connections):
                            network[1][j].Potential += value_synapses[0][j][i][k]
                # if training_stage == 3:
                #     print(network[1][0].Potential)

                # check whether the neurons in hidden layer fires and then update the output layer
                for j in range(Param.num_hidden_layer):
                    if network[1][j].check_fire() == 1:  # fire
                        spike_train_hidden[j][t] = 1
                        fired_time_hidden[j].append(t)
                        recent_fired_time_hidden[j] = t
                        # if (j == 1) & (training_stage == 0):
                        #     print(t)
                    for o in range(Param.num_output_layer):
                        network[2][o].Potential = 0
                        for k in range(Param.sub_connections):
                            x = t - recent_fired_time_hidden[j] - delay_synapses[1][o][j][k]
                            if x >= 0:
                                # network[2][o].Potential += w_synapses[1][o][j][k] * algorithm_reference.\
                                #     spike_response_function(x, Param.t_spike_function)
                                value_synapses[1][o][j][k] = w_synapses[1][o][j][k] * algorithm_reference.\
                                    spike_response_function(x, Param.t_spike_function)
                                # if (training_stage == 0) & (j == 1) & (o == 0) & (k == 2):
                                #     print(value_synapses[1][o][j][k])

                for j in range(Param.num_hidden_layer):
                    for o in range(Param.num_output_layer):
                        for k in range(Param.sub_connections):
                            network[2][o].Potential += value_synapses[1][o][j][k]
                # if training_stage == 0:
                #     print(network[2][0].Potential)

                # check the output layer
                for o in range(Param.num_output_layer):
                    if network[2][o].check_fire() == 1:
                        spike_train_output[o][t] = 1
                        fired_time_output[o].append(t)
                        finished = 1
                        recent_fired_time_output = t
                        # print("fire!")
                        print(recent_fired_time_output)
                        break

            # after forward spreading, update the weights by resume algorithm
            # update when the output neuron fires
            # the synapses between hidden and output
            for j in range(Param.num_hidden_layer):
                for k in range(Param.sub_connections):
                    for si in fired_time_hidden[j]:
                        s = fired_time_objective[training_stage] - (si - delay_synapses[1][0][j][k])
                        if s > 0:
                            w_change_d = (Param.a + algorithm_reference.learning_window(s))/Param.num_hidden_layer
                            w_synapses[1][0][j][k] += w_change_d
                        for sa in fired_time_output[0]:
                            s = sa - (si - delay_synapses[1][0][j][k])
                            if s > 0:
                                w_change_a = (Param.a + algorithm_reference.learning_window(s))/Param.num_hidden_layer
                                w_synapses[1][0][j][k] -= w_change_a
            # print(w_synapses)
            # the synapses between input and hidden
            for i in range(Param.num_input_layer):
                for j in range(Param.num_hidden_layer):
                    for k in range(Param.sub_connections):
                        for si in fired_time_input[i]:
                            s = fired_time_objective[training_stage] - (si - delay_synapses[0][j][i][k])
                            if s > 0:
                                w_change_d = (Param.a + algorithm_reference.learning_window(s)) * \
                                             sum(w_synapses[1][0][j]) / \
                                             (Param.num_hidden_layer * Param.num_input_layer)
                                w_synapses[0][j][i][k] -= w_change_d
                            for sa in fired_time_output[0]:
                                s = sa - (si - delay_synapses[0][j][i][k])
                                if s > 0:
                                    w_change_a = (Param.a + algorithm_reference.learning_window(s)) * \
                                             sum(w_synapses[1][0][j]) / \
                                             (Param.num_hidden_layer * Param.num_input_layer)
                                    # print(w_change_a)
                                    w_synapses[0][j][i][k] += w_change_a
            # print(w_synapses[1][0][0][0])
            # print(w_synapses[0][0][0][0])
            # warning! if output layer has more than one neuron, these code above must be modified!!!


if __name__ == '__main__':

    learning_or_test = int(sys.argv[1])
    resume_process(learning_or_test)

