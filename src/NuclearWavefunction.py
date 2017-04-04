# -*- coding: utf-8 -*-
# Gone through for release 10 Feb 2017
import copy

import matplotlib.pyplot as plt
import numpy as np

import Spacetime
import NuclearOperator


class nuclearWavefunction(object):
    "Defines a nuclear wavefunction and the operations which can occur on it"

    def __init__(self, SpaceToExistIn, groundStateNuclearHamiltonian= None, listOfInitialQuantumNumbers = None, randomThermalState = False):

        self.mySpace = SpaceToExistIn

        if groundStateNuclearHamiltonian is not None:
            if listOfInitialQuantumNumbers is None:
                self.xAmplitude = groundStateNuclearHamiltonian.groundStateAmplitude()
                self.initial_energy = groundStateNuclearHamiltonian.groundStateEnergy()
            else:
                self.xAmplitude = groundStateNuclearHamiltonian.energyEigenfunctionAmplitude(listOfInitialQuantumNumbers)
                self.initial_energy = groundStateNuclearHamiltonian.energyEigenvalue(listOfInitialQuantumNumbers)
        else:
            self.initial_energy = 0.0
            self.xAmplitude = self.mySpace.functionSpaceZero()

        self.kAmplitude = None


    def fourierTransform(self):
        "Define k-space from x-space"
        self.kAmplitude = self.mySpace.spatialFourierTransform(self.xAmplitude)
        self.xAmplitude = None

    def inverseFourierTransform(self):
        "Define x-space from k-space"
        self.xAmplitude = self.mySpace.spatialInverseFourierTransform(self.kAmplitude)
        self.kAmplitude = None

    def complexConjugate(self):
        "Re-Defines the Function to its complex Conjugate in position space"
        self.xAmplitude = np.conj(self.xAmplitude)

    def returnComplexConjugate(self):
        "Returns new object which is the Complex Conjugate of the current"
        out = copy.copy(self)
        out.complexConjugate()
        return out

    def integratedAmplitude(self):
        return self.mySpace.xIntegration(self.xAmplitude)


    def __mul__(self, other):
        "both must be in position space"
        output = copy.copy(self)
        try:
            #We excpect the other argument to be another spaceFunction object
            output.xAmplitude = output.xAmplitude * other.xAmplitude
        except AttributeError:
            #But it may be a scalar
            output.xAmplitude = output.xAmplitude * other
        return output

    def __neg__(self):
        "both must be in position space"
        output = copy.copy(self)
        output.xAmplitude = -output.xAmplitude
        return output

    def __add__(self, other):
        "both must be in position space"
        output = copy.copy(self)
        output.xAmplitude = output.xAmplitude + other.xAmplitude
        return output

    def __pow__(self, n):
        "must be in position space"
        output = copy.copy(self)
        output.xAmplitude = output.xAmplitude**n
        return output

    def __radd__(self, other):
        return self.__add__(other)
    def __rmul__(self, other):
        return self.__mul__(other)

    def overlap(self, otherWF):
        "Takes this wavefunction as the bra, the other as the ket and integrates"
        if isinstance(otherWF, ZeroNuclearWavefunction):
            return 0.0
        out = self.xAmplitude * np.conj(otherWF.xAmplitude)
        return self.mySpace.xIntegration(out)

    def norm(self):
        return self.overlap(self)**(.5)

    def plot(self):
        if self.mySpace.nuclearDimensionality > 2:
            raise unplotableNuclearWavefunction()
        else:
            figure =  plt.figure()
            if self.mySpace.nuclearDimensionality is 2:
                plt.contourf(np.abs(self.xAmplitude), 100, )
                plt.colorbar()
            else:
                plt.plot(np.abs(self.xAmplitude))
        return figure

class ZeroNuclearWavefunction(nuclearWavefunction):
    "A Function which knows it's zero and can thus save some time for computations"
    def __init__(self, space):
        self.mySpace = space
        self.xAmplitude = 0.0
        self.kAmplitude = 0.0


    def __mul__(self, other):
        return self

    def __add__(self, other):
        return other

    def __neg__(self):
        return self

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)


    def fourierTransform(self):
        pass

    def inverseFourierTransform(self):
        pass

    def complexConjugate(self):
        pass

    def returnComplexConjugate(self):
        return self

    def overlap(self, other):
        return 0.0

    #Copy just needs to return itself
    def __deepcopy__(self, dontcare):
        return self
    def __copy__(self):
        return self



class unplotableNuclearWavefunction(Exception):
    def __str__(self):
        return "You can't plot this wavefunction; it's got too many nuclear dimensions!"


if __name__ == "__main__":
    #Some testing Code
    mySpace = Spacetime.Spacetime(xMax = 10,
                 numberOfNuclearDimenions = 1,
                 numberOfElectronicDimensions = 2,
                 numberOfSimulationSpacePointsPerNuclearDimension = 100,
                 dt = .1)
    testHarmonicOscillator = NuclearOperator.harmonicOscillator(mySpace,
                 omega=1,
                 mass=1,
                 center=0,
                 energyOffset = 0)
    testNuclearHamiltonian = NuclearOperator.nuclearHamiltonian(mySpace,
                 listOfOneDimensionalHamiltonians = [testHarmonicOscillator] )
    testNuclearWF = NuclearOperator.NuclearWavefunction.nuclearWavefunction(mySpace,
                                        groundStateNuclearHamiltonian = testNuclearHamiltonian)



    Norm =  testNuclearWF.norm()#edges go nuts so pick one in the middle
    assert Norm - 1 < mySpace.DEFAULT_ZERO_TOLERANCE, """ Harmonic Oscillator not normalized!"""
