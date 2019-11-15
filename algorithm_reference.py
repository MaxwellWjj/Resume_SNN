# Author: Tony Wu, OEI, HuaZhong University of Science and Technology
# These are some formulas of ReSuMe algorithm


import numpy as np
from parameters import Param


def spike_response_function(x, const_time):
    response = (x/const_time)*np.exp(1-(x/const_time))
    return response


def learning_window(s):
    a = 0.0
    if s <= 0:
        a = -Param.Aminus * np.exp(s/Param.tminus)
    if s > 0:
        a = Param.Aplus * np.exp((-s)/Param.tplus)
    return a
