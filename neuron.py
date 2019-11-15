# Author: Tony Wu, OEI, HuaZhong University of Science and Technology
# This is neuron class which defines the dynamics of a neuron.
# All the parameters are initialised and methods are included to check


from parameters import Param


class Neuron:
    def __init__(self):
        self.Pth = Param.pth  # Pth: potential threshold
        self.Prest = 0  # Prest: potential in rest
        self.Potential = self.Prest  # currently potential

    def check_fire(self):   # check the neuron-whether it should be fired now.
        if self.Potential >= self.Pth:
            return 1
        elif self.Potential <= self.Prest:
            self.Potential = self.Prest
            return 0
        else:
            return 0

    def initialize(self, vth):
        self.Pth = vth
        self.Prest = Param.prest
        self.Potential = self.Prest
