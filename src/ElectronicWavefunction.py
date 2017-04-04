# -*- coding: utf-8 -*-
#Edited for release 9 Feb 2017

import copy

import matplotlib.pyplot as plt
import numpy as np

import Spacetime
import NuclearOperator
import NuclearWavefunction


class electronicWavefunction(object):
    """A collection of nuclear wavefunctions to represent nuclear amplitudes on
    different electronic states"""

    def __init__(self, space, listOfNuclearWavefunctions, Normalize=False):
        """
        Initializes an electronicWavefunction Object.  
        listOfNuclearWavefunctions must be of length electronicDimension
        Can include the number the number 0 isntead and a ZeroNuclearWavefunction object will be inserted
        """
        self.mySpace = space
        self.dim  = self.mySpace.electronicDimensionality

        if len(listOfNuclearWavefunctions) is 0:
            listOfNuclearWavefunctions = [0] * self.dim
        elif len(listOfNuclearWavefunctions) is not self.dim:
            raise electronicDimensionMismatch()

        self.myVector = []
        self.initial_energy = 0.0
        for wf in listOfNuclearWavefunctions:
            wf = copy.copy(wf)
            if wf is 0:
                self.myVector.append(NuclearWavefunction.ZeroNuclearWavefunction(self.mySpace))
            else:
                self.myVector.append(wf)
                self.initial_energy = self.initial_energy + wf.initial_energy

        if Normalize:
            mult = self.norm()**.5
            for wf in self.myVector:
                wf.xAmplitude = wf.xAmplitude / mult

    def __getitem__(self, n):
        return self.myVector[n]

    def __setitem__(self, n, obj):
        self.myVector[n] = obj

    def complexCojugate(self):
        for i in range(self.dim):
            self[i].complexConjugate()

    def overlap(self, otherEWF):
        "Caclulates the inner product between this and other ewf"
        integrals = 0.0
        for i, wf in enumerate(self.myVector):
            integrals = integrals + wf.overlap(otherEWF[i])
        return integrals

    def norm(self):
        return self.overlap(self)

    def __mul__(self, other):
        "for constant other only"
        out = copy.copy(self)
        for i in range(self.dim):
            out[i] = other * out[i]
        return out
    
    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        out = copy.copy(self)
        for i in range(self.dim):
            out[i] = other[i] + out[i]
        return out

    def plot(self, saveString=None):
        if self.mySpace.nuclearDimensionality is not 1:
            raise Exception("Can only plot 1D electronic wavefunctions!")
        x = self.mySpace.xValues_list[0]
        fig = plt.figure()
        for i, wf in enumerate(self.myVector):
            y = np.abs(wf.xAmplitude)
            if not isinstance(y, np.ndarray):
                y = np.zeros(x.shape)
            plt.plot(x, y, label=str(i))
        plt.legend()
        return fig

    def zeroElectronicWavefunction(self):
        "A wavefunction which will start as zero but can be modified and end up not zero"
        return electronicWavefunction(self.mySpace, [])


class zeroElectronicWavefunction(electronicWavefunction):
    """A wavefunction which is smart enough to not waste time calculating"""

    def __init__(self, space):
        super(zeroElectronicWavefunction,self).__init__(space, [])

    def __setitem__(self, n, obj):
        raise Exception("DO NOT CHANGE ELEMENTS OF A ZERO WAVEFUNCTION")

    def overlap(self, otherEWF):
        return 0


    def __mul__(self, other):
        return self

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        return other


class electronicDimensionMismatch(Exception):
    def __str__(self):
        return "different electronic dimensions expected NO DICE"


if __name__ == "__main__":
    #Some test code below
    mySpace = Spacetime.Spacetime(xMax = 10,
                 numberOfNuclearDimenions = 1,
                 numberOfElectronicDimensions = 4,
                 numberOfSimulationSpacePointsPerNuclearDimension = 200,
                 dt = .01)

    omega0 = 2.0
    omegaOff = 1.0
    testHarmonicOscillator1 = NuclearOperator.harmonicOscillator(mySpace,
                 omega=omega0,
                 mass=1,
                 center=0,
                 energyOffset = 0)
    testHarmonicOscillator2 = NuclearOperator.harmonicOscillator(mySpace,
                 omega= 2 * omega0,
                 mass=1,
                 center=0,
                 energyOffset = 0)


    testHarmonicOscillator3 = NuclearOperator.harmonicOscillator(mySpace,
                 omega=omegaOff,
                 mass=1,
                 center=3,
                 energyOffset = 0)
    testHarmonicOscillator4 = NuclearOperator.harmonicOscillator(mySpace,
                 omega= .5*omegaOff,
                 mass=1,
                 center=2,
                 energyOffset = 0)

    testNuclearHamiltonian1 = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [testHarmonicOscillator1 ] )
    testNuclearHamiltonian2 = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [testHarmonicOscillator2 ] )
    testNuclearHamiltonian3 = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [testHarmonicOscillator3 ] )
    testNuclearHamiltonian4 = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [testHarmonicOscillator4 ] )


    testNuclearWavefunction1 = NuclearWavefunction.nuclearWavefunction(mySpace, groundStateNuclearHamiltonian = testNuclearHamiltonian1 )
    testNuclearWavefunction2 = NuclearWavefunction.nuclearWavefunction(mySpace, groundStateNuclearHamiltonian = testNuclearHamiltonian2 )
    testNuclearWavefunction3 = NuclearWavefunction.nuclearWavefunction(mySpace, groundStateNuclearHamiltonian = testNuclearHamiltonian3 )
    testNuclearWavefunction4 = NuclearWavefunction.nuclearWavefunction(mySpace, groundStateNuclearHamiltonian = testNuclearHamiltonian4 )

    testEWF1 = electronicWavefunction(mySpace,
                           listOfNuclearWavefunctions = [testNuclearWavefunction1, testNuclearWavefunction2, testNuclearWavefunction3, testNuclearWavefunction4],
                           Normalize=False)
    testEWF1.plot("lalalal")
    print testEWF1.norm()

    testEWF2 = electronicWavefunction(mySpace,
                           listOfNuclearWavefunctions = [testNuclearWavefunction1, testNuclearWavefunction2, testNuclearWavefunction3, testNuclearWavefunction4],
                           Normalize=True)
    testEWF2.plot("lalalal")
    print testEWF2.norm()
    print testEWF2.overlap(testEWF1)
