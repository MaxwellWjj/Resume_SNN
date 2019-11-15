# Author: Tony Wu, OEI, HuaZhong University of Science and Technology
# This file contains all the parameters of the network to be trained.


class Param:

    epoch = 150
    T = 100

    prest = 0
    pth = 0.7

    Aminus = 0.5
    Aplus = 1.2

    tminus = 5
    tplus = 5

    t_spike_function = 7
    t_after_potential_kernel = 12

    a = 0.05
    scaling_factor = 0.005

    sub_connections = 5
    sub_connections_delay_min = 0
    sub_connections_delay_max = 4

    w_random_min = 0.1/sub_connections
    w_random_max = 0.8/sub_connections

    num_input_layer = 2  # neuron number of input layer
    num_hidden_layer = 5  # neuron number of hidden layer
    num_output_layer = 1  # neuron number of output layer

    input_zero_delay = 6
    input_one_delay = 0
    output_zero_delay = 16
    output_one_delay = 10
