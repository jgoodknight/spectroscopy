# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 18:17:00 2012
Eeited for release 9 Feb 2017
@author: Joey
"""

import copy
import time

import numpy as np
import scipy.linalg

import Spacetime
import NuclearOperator
import NuclearWavefunction
import ElectronicWavefunction


EXPONENTIAL_APPROXIMATION_TRUNCATION = 40

class ElectronicOperator(object):
    "Abstract Class to hold matrix of nuclearOperators in electronic-dimension space"
    def __init__(self, space, listOfijNuclearOperators):
        self.checkImplemented() #Will throw an exception unless being used by subclass
        self.mySpace = space

        self.dim = self.mySpace.electronicDimensionality

        #set up matrix to be the proper size so we can index it
        self.myMatrix = self.zeroMatrix()

        # self.myListOfIJNuclearOperatorTuples = listOfijNuclearOperators

        #then put the actual operators in there
        for ijNucOpTuple in listOfijNuclearOperators:
            i = ijNucOpTuple[0]
            j = ijNucOpTuple[1]
            NucOp = ijNucOpTuple[2]
            self.myMatrix[i][j] = NucOp

    def __rmul__(self, other):
        return self.__mul__(other)



    def exponential(self, coefficient):
        #we must start by flattening the nuclear operators
        flatDimension = len(self.myMatrix[0][0].flatten())
        flattenedMatrix = np.zeros((flatDimension, self.dim, self.dim), dtype=np.complex)

        for i in range(self.dim):
            for j in range(self.dim):
                flattenedMatrix[:, i, j] = self.myMatrix[i][j].flatten()
        #multiply everything by the desired coefficient
        flattenedMatrix = flattenedMatrix * coefficient
        #now we do the exponentiation
        flattenedExponentiatedMatrix = np.zeros((flatDimension, self.dim, self.dim), dtype=np.complex)
        for k in range(flatDimension):
            flattenedExponentiatedMatrix[k] = scipy.linalg.expm(flattenedMatrix[k])
        #now with the expoentiation all done, we must reshape everything
        output = copy.deepcopy(self)
        for i in range(self.dim):
            for j in range(self.dim):
                output.myMatrix[i][j] = self.myMatrix[i][j].reshapeAndSetAsSurfaceForOutput(flattenedExponentiatedMatrix[:, i, j])
        return output




    def __add__(self, other):
        """Adds the content of two Electronic Operators
        Will Throw Exception for bad input"""
        out = copy.deepcopy(self)
        try:
            for i in range(self.dim):
                for j in range(self.dim):
                    out.myMatrix[i][j] = self.myMatrix[i][j] + other.myMatrix[i][j]
        except(AttributeError):
            for i in range(self.dim):
                for j in range(self.dim):
                    out.myMatrix[i][j] = self.myMatrix[i][j] + other
        return out

    def __mul__(self, other):
        """Will either multiply two electronic operators or apply an electronic operator
        to an electronic wavefunction or take a constant and multiply all elements of the matrix
        by that constant"""
        #ElectronicOperator multiplication

        if isinstance(other, ElectronicOperator):
            if self.dim is not other.dim:
                raise Exception("Matrices not same dimension")
            out = self.__class__(self.mySpace, [])
            for i in range(self.dim):
                for j in range(self.dim):
                    for k in range(self.dim):
                        out.myMatrix[i][j] = out.myMatrix[i][j] + (self.myMatrix[i][k] * other.myMatrix[k][j])
            return out

        if isinstance(other, ElectronicWavefunction.electronicWavefunction):
            if self.dim is not other.dim:
                raise Exception("Matrices not same dimension")
            out = other.__class__(self.mySpace, [])
            for i in range(self.dim):
                for j in range(self.dim):
                    out[i]  = out[i] + self.myMatrix[i][j].APPLY(other[j])
            return out
        #try just multiplying each matrix element
        out = self.__class__(self.mySpace, [])
        for i in range(self.dim):
            for j in range(self.dim ):
                out.myMatrix[i][j] = self.myMatrix[i][j] * other
        return out

    def printMe(self):
        "Used for debugging purposes to roughly make sure items aren't zero"
        for i in range(self.dim):
            for j in range(self.dim):
                print (i, j, np.average(self.myMatrix[i][j].surface))
            print "--"
        print "------"

    def __str__(self):
        return str(self.myMatrix)

    def __pow__(self, power):
        out = copy.deepcopy(self)
        for i in range(power - 1):
            out = out * self
        return out

    def zeroMatrix(self):
         "returns properly sized matrix of the proper zero nuclear operator"
         outer = []
         for i in range(self.dim):
             inner = []
             for j in range(self.dim):
                 newObj = self.__class__.zeroNuclearOperatorFunction(self.mySpace)
                 inner.append(newObj)
             outer.append(inner)
         return outer

    def indentity(self):
         "returns properly sized matrix of the proper unit nuclear operator"
         output = self.zeroMatrix()
         for i in range(self.dim):
             output[i][i] = self.__class__.constantNuclearOperatorFunction(self.mySpace, 1.0)
         return output

    def checkImplemented(self):
        raise Exception("Do not use superclass; define electronic operator as being in position or momentum space!")


class ElectronicPositionOperator(ElectronicOperator):
    "Class to hold matrix of nuclearPositionOperators"
    #Class variables
    zeroNuclearOperatorFunction = NuclearOperator.zeroPositionOperator
    constantNuclearOperatorFunction = NuclearOperator.constantPositionNuclearOperator


    def excitationOperator(self):
        "Choses upper-triangular matrix which excites electronic states"
        newListOfijOperatorTuples = []
        for i, jList in enumerate(self.myMatrix):
            for j, nucOp in enumerate(self.myMatrix[i]):
                if( i > j ):
                    newListOfijOperatorTuples.append((i, j, nucOp))
        return ElectronicPositionOperator(self.mySpace, newListOfijOperatorTuples)

    def relaxationOperator(self):
        "Choses lower-triangular matrix which relaxes electronic states"
        newListOfijOperatorTuples = []
        for i, jList in enumerate(self.myMatrix):
            for j, nucOp in enumerate(self.myMatrix[i]):
                if( i < j ):
                    newListOfijOperatorTuples.append((i, j, nucOp))
        return ElectronicPositionOperator(self.mySpace, newListOfijOperatorTuples)

    def checkImplemented(self):
        pass


class ElectronicMomentumOperator(ElectronicOperator):
    "Class to hold matrix of nuclearMomentumOperators.  Mainly used for Kinetic Energy operators"
    #Class variables
    zeroNuclearOperatorFunction = NuclearOperator.zeroMomentumOperator
    constantNuclearOperatorFunction = NuclearOperator.constantMomentumNuclearOperator

    def checkImplemented(self):
        pass


class ElectronicHamiltonian(object):
    "A Hamiltonian electronic degrees of freedom"
    def __init__(self, space, listOfijNuclearOperators):
        self.mySpace = space

        self.__ijOperatorlist = listOfijNuclearOperators

        self.electronicDim = self.mySpace.electronicDimensionality

        kineticTuples = []
        potentialTuples = []
        omegas = []
        for i, element in enumerate(self.__ijOperatorlist):
            if isinstance(element[2], NuclearOperator.momentumOperator):
                kineticTuples.append(element)
            elif isinstance(element[2], NuclearOperator.positionOperator):
                potentialTuples.append(element)
                omegas.append(element[2].omega_low)
                omegas.append(element[2].omega_high)
            elif isinstance(element[2], NuclearOperator.nuclearHamiltonian):
                kineticTuples.append((element[0], element[1], element[2].myKineticOperator))
                potentialTuples.append((element[0], element[1], element[2].myPotentialOperator))
                omegas.append(element[2].omega_low)
                omegas.append(element[2].omega_high)
                if element[0] ==0 and element[1] == 0:
                    self.nuclear_ground_state = element[2]
        omegas = np.array(omegas)
        omegas = omegas[np.where(omegas > 0)]
        self.omega_low = np.min(omegas)
        self.omega_high = np.max(omegas)

        self.kineticTuples = kineticTuples
        self.potentialTuples = potentialTuples
        self.myMomentumOperator = ElectronicMomentumOperator(self.mySpace, kineticTuples)
        self.myPositionOperator = ElectronicPositionOperator(self.mySpace, potentialTuples)

    def APPLY(self, electronicWavefunction):
        "Applies the Hamiltonian to an electronic wavefunction"
        return (self.myMomentumOperator * electronicWavefunction) + (self.myPositionOperator * electronicWavefunction)

    def myPropagator(self, overrideDT = None):
        return ElectronicPropagator(self.mySpace, self.myPositionOperator, self.myMomentumOperator, overrideDT = overrideDT)


    def addDiagonalDisorder(self, covarianceMatrix):
        "Returns a copy of itself with diagnonal noise for 2 or 4 electronic dimensions only"
        output = copy.deepcopy(self)
        #generate random noise
        meanValues = [0.0] * covarianceMatrix.shape[0]
        noises = np.random.multivariate_normal(meanValues, covarianceMatrix)
        groundStateNoiseOperator = NuclearOperator.constantPositionNuclearOperator(self.mySpace, 0.0)

        listOfTuples = []

        listOfTuples.append((0, 0, groundStateNoiseOperator))

        if self.electronicDim == 2:
            excitedStateNoiseOperator = NuclearOperator.constantPositionNuclearOperator(self.mySpace, noises[0])
            listOfTuples.append((1, 1, excitedStateNoiseOperator))
        elif self.electronicDim == 4:
            excitedStateNoiseOperator01 = NuclearOperator.constantPositionNuclearOperator(self.mySpace, noises[0])
            excitedStateNoiseOperator10 = NuclearOperator.constantPositionNuclearOperator(self.mySpace, noises[1])
            excitedStateNoiseOperator11 = excitedStateNoiseOperator10 + excitedStateNoiseOperator01
            listOfTuples.append((1, 1, excitedStateNoiseOperator01))
            listOfTuples.append((2, 2, excitedStateNoiseOperator10))
            listOfTuples.append((3, 3, excitedStateNoiseOperator11))
        else:
            Warning("reverting to no additional noise; must program in how to add noise for a %s dimensional system" % str(self.electronicDim))
            return output
        #create a diagonal electronic operator to hold the noise operators
        noiseElectronicOperator = ElectronicPositionOperator(self.mySpace, listOfTuples)
        #add to the current position Operator and return
        output.myPositionOperator = output.myPositionOperator + noiseElectronicOperator
        return output

    def groundStateElectronicWavefunction(self):
        "Returns the lowest energy electronic wavefuntion possible"
        ground_H = self.nuclear_ground_state
        ground_nuc_wf = NuclearWavefunction.nuclearWavefunction(self.mySpace, groundStateNuclearHamiltonian = ground_H )
        ewf_to_return = ElectronicWavefunction.electronicWavefunction(self.mySpace, listOfNuclearWavefunctions = [ground_nuc_wf, 0, 0, 0, 0], Normalize=True)
        return ewf_to_return


class ElectronicPropagator(object):
    "a medium-accuracy [O(dt^3)] propagator for an electronic hamiltonian"
    def __init__(self, space, positionOperator, momentumOperator, overrideDT = None):
        self.mySpace = space
        self.myPositionOperator = positionOperator
        self.myMomentumOperator = momentumOperator

        if overrideDT is not None:
            self.dt = overrideDT
        else:
            self.dt = self.mySpace.dt

        timePropagationKineticCoefficient = -.5j * self.dt / self.mySpace.hbar
        timePropagationPotentialCoefficient = -1.0j * self.dt / self.mySpace.hbar

        self.momentumPropagator = self.myMomentumOperator.exponential(timePropagationKineticCoefficient)
        self.positionPropagator = self.myPositionOperator.exponential(timePropagationPotentialCoefficient)


    def APPLY(self, wavefunction):
        wavefunction = self.momentumPropagator * wavefunction
        wavefunction = self.positionPropagator * wavefunction
        wavefunction = self.momentumPropagator * wavefunction
        return wavefunction






if __name__ == "__main__":
    #Some Test Code
    #Does the identity work?
    mySpace = Spacetime.Spacetime(xMax = 10,
                 numberOfNuclearDimenions = 1,
                 numberOfElectronicDimensions = 4,
                 numberOfSimulationSpacePointsPerNuclearDimension = 200,
                 dt = .01)
    constantPositionOperator = NuclearOperator.constantPositionNuclearOperator(mySpace, 1)
    testElectronicOperator = ElectronicPositionOperator(mySpace,
                                                        [(0, 0, constantPositionOperator),
                                                         (1, 1, constantPositionOperator),
                                                         (2, 2, constantPositionOperator),
                                                         (3, 3, constantPositionOperator)])

    testElectronicOperator.printMe()
    testElectronicOperator = testElectronicOperator * testElectronicOperator
    print "Identity"
    testElectronicOperator = testElectronicOperator.exponentialTaylorApproximation(1.0j * np.pi, 16)
    testElectronicOperator.printMe()
    print "Exp[2Identity]"


    #create a nilpotent matrix to test exponentiation routine
    mySpace = Spacetime.Spacetime(xMax = 10,
                 numberOfNuclearDimenions = 1,
                 numberOfElectronicDimensions = 3,
                 numberOfSimulationSpacePointsPerNuclearDimension = 200,
                 dt = .01)
    constantPositionOperator00 = NuclearOperator.constantPositionNuclearOperator(mySpace, 5)
    constantPositionOperator01 = NuclearOperator.constantPositionNuclearOperator(mySpace, -3)
    constantPositionOperator02 = NuclearOperator.constantPositionNuclearOperator(mySpace, 2)
    constantPositionOperator10 = NuclearOperator.constantPositionNuclearOperator(mySpace, 15)
    constantPositionOperator11 = NuclearOperator.constantPositionNuclearOperator(mySpace, -9)
    constantPositionOperator12 = NuclearOperator.constantPositionNuclearOperator(mySpace, 6)
    constantPositionOperator20 = NuclearOperator.constantPositionNuclearOperator(mySpace, 10)
    constantPositionOperator21 = NuclearOperator.constantPositionNuclearOperator(mySpace, -6)
    constantPositionOperator22 = NuclearOperator.constantPositionNuclearOperator(mySpace, 4)
    testElectronicOperator = ElectronicPositionOperator(mySpace,
                                                        [(0, 0, constantPositionOperator00),
                                                         (0, 1, constantPositionOperator01),
                                                         (0, 2, constantPositionOperator02),
                                                         (1, 0, constantPositionOperator10),
                                                         (1, 1, constantPositionOperator11),
                                                         (1, 2, constantPositionOperator12),
                                                         (2, 0, constantPositionOperator20),
                                                         (2, 1, constantPositionOperator21),
                                                         (2, 2, constantPositionOperator22)])

    testElectronicOperator.printMe()
    print "Nilpotent"
    zero = testElectronicOperator**2
    zero.printMe()
    print "Zero"
    testElectronicOperator = testElectronicOperator.exponentialTaylorApproximation(1.0, 5)
    testElectronicOperator.printMe()
    print "Identity + Nilpotent"


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
                 center=-2,
                 energyOffset = .1)
    testHarmonicOscillator2 = NuclearOperator.harmonicOscillator(mySpace,
                 omega= 2 * omega0,
                 mass=1,
                 center=0,
                 energyOffset = 0)


    testHarmonicOscillator3 = NuclearOperator.harmonicOscillator(mySpace,
                 omega=omegaOff,
                 mass=1,
                 center=2,
                 energyOffset = 2)
    testHarmonicOscillator4 = NuclearOperator.harmonicOscillator(mySpace,
                 omega= .5*omegaOff,
                 mass=1,
                 center=3,
                 energyOffset = 0)

    testNuclearHamiltonian1 = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [testHarmonicOscillator1 ] )
    testNuclearHamiltonian2 = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [testHarmonicOscillator2 ] )
    testNuclearHamiltonian3 = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [testHarmonicOscillator3 ] )
    testNuclearHamiltonian4 = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [testHarmonicOscillator4 ] )


    testNuclearWavefunction1 = NuclearWavefunction.nuclearWavefunction(mySpace, groundStateNuclearHamiltonian = testNuclearHamiltonian1 )
    testNuclearWavefunction2 = NuclearWavefunction.nuclearWavefunction(mySpace, groundStateNuclearHamiltonian = testNuclearHamiltonian2 )
    testNuclearWavefunction3 = NuclearWavefunction.nuclearWavefunction(mySpace, groundStateNuclearHamiltonian = testNuclearHamiltonian3 )
    testNuclearWavefunction4 = NuclearWavefunction.nuclearWavefunction(mySpace, groundStateNuclearHamiltonian = testNuclearHamiltonian4 )

    testEWF1 = ElectronicWavefunction.electronicWavefunction(mySpace,
                           listOfNuclearWavefunctions = [testNuclearWavefunction1, testNuclearWavefunction2, testNuclearWavefunction3, testNuclearWavefunction4],
                           Normalize=False)
    testEWF1.plot('q')
    testElectronicHamiltonian = ElectronicHamiltonian(mySpace, [(0,0, testNuclearHamiltonian1),
                                                                (1,1, testNuclearHamiltonian2),
                                                                (2,2, testNuclearHamiltonian3),
                                                                (3,3, testNuclearHamiltonian4)])
    testEWF1 = testElectronicHamiltonian.APPLY(testEWF1)
    testEWF1.plot('q')

    #create operator to excite from one level to another
    transitionDipoleOperator = NuclearOperator.constantPositionNuclearOperator(mySpace, 1)
    transitionOperator = ElectronicPositionOperator(mySpace, [(0, 1, transitionDipoleOperator), (1, 0, transitionDipoleOperator)])
    testEWF1 = ElectronicWavefunction.electronicWavefunction(mySpace,
                           listOfNuclearWavefunctions = [testNuclearWavefunction1, 0, 0, 0],
                           Normalize=False)
    testEWF1.plot('q')
    testEWF1 = transitionOperator * testEWF1
    testEWF1.plot('q')
    testEWF1 = transitionOperator * testEWF1
    testEWF1.plot('q')

    # Test ElectronicPropagator
    mySpace = Spacetime.Spacetime(xMax = 10,
                 numberOfNuclearDimenions = 1,
                 numberOfElectronicDimensions = 4,
                 numberOfSimulationSpacePointsPerNuclearDimension = 200,
                 dt = .05)
    omega0 = 2.0
    omegaOff = 1.0
    testHarmonicOscillator1 = NuclearOperator.harmonicOscillator(mySpace,
                 omega=omega0,
                 mass=1,
                 center=-2,
                 energyOffset = .1)
    testHarmonicOscillator2 = NuclearOperator.harmonicOscillator(mySpace,
                 omega= 2 * omega0,
                 mass=1,
                 center=0,
                 energyOffset = 0)


    testHarmonicOscillator3 = NuclearOperator.harmonicOscillator(mySpace,
                 omega=omegaOff,
                 mass=1,
                 center=2,
                 energyOffset = 2)
    testHarmonicOscillator4 = NuclearOperator.harmonicOscillator(mySpace,
                 omega= .5*omegaOff,
                 mass=1,
                 center=3,
                 energyOffset = 0)

    testNuclearHamiltonian1 = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [testHarmonicOscillator1 ] )
    testNuclearHamiltonian2 = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [testHarmonicOscillator2 ] )
    testNuclearHamiltonian3 = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [testHarmonicOscillator3 ] )
    testNuclearHamiltonian4 = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [testHarmonicOscillator4 ] )


    testNuclearWavefunction1 = NuclearWavefunction.nuclearWavefunction(mySpace, groundStateNuclearHamiltonian = testNuclearHamiltonian1 )
    testNuclearWavefunction2 = NuclearWavefunction.nuclearWavefunction(mySpace, groundStateNuclearHamiltonian = testNuclearHamiltonian2 )
    testNuclearWavefunction3 = NuclearWavefunction.nuclearWavefunction(mySpace, groundStateNuclearHamiltonian = testNuclearHamiltonian3 )
    testNuclearWavefunction4 = NuclearWavefunction.nuclearWavefunction(mySpace, groundStateNuclearHamiltonian = testNuclearHamiltonian4 )

    testElectronicHamiltonian = ElectronicHamiltonian(mySpace, [(0,0, testNuclearHamiltonian3),
                                                                (1,1, testNuclearHamiltonian1),
                                                                (2,2, testNuclearHamiltonian4),
                                                                (3,3, testNuclearHamiltonian2)])
    testEWF = ElectronicWavefunction.electronicWavefunction(mySpace,
                           listOfNuclearWavefunctions = [testNuclearWavefunction1, testNuclearWavefunction2, testNuclearWavefunction3, testNuclearWavefunction4],
                           Normalize=True)

    EXPONENT_ASSUMPTIONS = {'diagonalKinetic': True,
                       'diagonalPotential' : False,
                       'OneNonDiagonal2by2' : False,
                       '2by2Indeces' : (0, 0)}

    testElectronicPropagator = testElectronicHamiltonian.myPropagator(EXPONENT_ASSUMPTIONS)
    testEWF.plot()
    for i in range(6):
        testEWF = testElectronicPropagator.APPLY(testEWF)
        testEWF.plot()
    testElectronicPropagator.APPLY(testEWF)
    testEWF.plot()
    testElectronicPropagator.momentumPropagator.printMe()
    testElectronicPropagator.positionPropagator.printMe()

    explicitTime = time.time()
    explicitElectronicPropagator = testElectronicHamiltonian.myPropagator(EXPONENT_ASSUMPTIONS)
    explicitTime = time.time() - explicitTime

    noAssumeTime = time.time()
    noAssumeElectronicPropagator = testElectronicHamiltonian.myPropagator()
    noAssumeTime = time.time() - noAssumeTime

    explicitElectronicPropagator.positionPropagator.printMe()
    noAssumeElectronicPropagator.positionPropagator.printMe()

    print "explicit", explicitTime
    print "no assumptions", noAssumeTime
