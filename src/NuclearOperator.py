# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 12:35:19 2012
CLEANED UP FOR RELEASE 10 Feb 2017
@author: Joey
"""
import math
import copy
import inspect
import itertools
import warnings

import numpy as np
import matplotlib as plt
import numpy.polynomial.hermite as herm
import scipy.integrate
import scipy.misc
import scipy.special

import sympy
import sympy.mpmath
import sympy.functions.combinatorial.factorials as fact

import matplotlib.pyplot as plt

import Spacetime
import NuclearWavefunction


class nuclearOperator(object):
    """
    abstract superclass which defines an operator in both x and k space to act on a
    function which naturally exists in x-space
    """

    def __add__(self, other):
        out = copy.copy(self)
        try:
            out.surface = out.surface + other.surface
        except(AttributeError):
            out.surface = out.surface + other
        return out

    def __neg__(self):
        out = copy.copy(self)
        out.surface = -1.0 * out.surface
        return out

    def __mul__(self, other):
        out = copy.copy(self)
        try:
            out.surface = self.surface * other.surface
        except(AttributeError):
            out.surface = self.surface * other
        return out

    def __rmul__(self, other):
        return self.__mul__(other)
    def __radd__(self, other):
        return self.__add__(other)

    def flatten(self):
        "Returns a flattened version of the grid representation of the operator"
        return self.surface.flatten()

    def reshapeAndSetAsSurfaceForOutput(self, flattenedNuclearOperator):
        "Takes in a flattened Operator, reshapes it and sets as the surface"

        output = copy.copy(self)
        newSurface = np.reshape(flattenedNuclearOperator, self.mySpace.functionSpaceDimension)
        try:
            output.surface.shape
            #OK, it's a nonzeroOperator, just set the surface and go on
            output.surface = newSurface
        except AttributeError:
            #zero operator, needs a different class
            output = self.__class__.nonZeroOperatorClass(self.mySpace, newSurface)

        return output


    def __div__(self, other):
        out = copy.copy(self)
        try:
            out.surface  = out.surface / other.surface
        except(AttributeError):
            out.surface = self.surface / other
        return out

    def __pow__(self, n):
        out = copy.copy(self)
        out.surface = np.power(out.surface, n)
        return out

    def cosh(self):
        out = copy.copy(self)
        out.surface = np.cosh(out.surface)
        return out

    def sinh(self):
        out = copy.copy(self)
        out.surface = np.sinh(out.surface)
        return out

    def cos(self):
        out = copy.copy(self)
        out.surface = np.cos(out.surface)
        return out

    def sin(self):
        out = copy.copy(self)
        out.surface = np.sin(out.surface)
        return out

    def exp(self):
        out = copy.copy(self)
        out.surface = np.exp(out.surface)
        return out

class momentumOperator(nuclearOperator):
    "A Class which will Fourier Transform the target wavefunction into k-space before and after application"

    def __init__(self, space, surface):
        self.mySpace = space
        self.surface = surface


    def APPLY(self, wavefunction):
        if isinstance(wavefunction, NuclearWavefunction.ZeroNuclearWavefunction):
            return wavefunction
        out = copy.copy(wavefunction)
        out.fourierTransform()
        out.kAmplitude = out.kAmplitude * self.surface
        out.inverseFourierTransform()
        return out

class positionOperator(nuclearOperator):
    "An Operator which just acts on the native position state of the wavefunction"

    def __init__(self, space, surface):
        self.mySpace = space
        self.surface = surface

    def APPLY(self, wavefunction):
        if isinstance(wavefunction, NuclearWavefunction.ZeroNuclearWavefunction):
            return wavefunction
        out = copy.copy(wavefunction)
        out.xAmplitude = out.xAmplitude * self.surface
        return out


class zeroNuclearOperator(nuclearOperator):
    "An operator which knows it is zero so it doesn't waste time doing operations"

    def __add__(self, other):
        return other

    def __neg__(self):
        return self

    def __mul__(self, other):
        return self

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        return self

    def __pow__(self, n):
        return self

    def APPLY(self, wavefunction):
        return NuclearWavefunction.ZeroNuclearWavefunction(self.mySpace)

    def flatten(self):
        "Returns a flattened version of the grid representation of the operator.  Used primarily for matrix exponents"
        return self.mySpace.functionSpaceZero().flatten()

    def reshapeAndSetAsSurface(self, flattenedOperator):
        "Takes in a flattened Operator, reshapes it and sets as the surface.  Used primarily for matrix exponents"
        if np.any(np.abs(flattenedOperator) > self.mySpace.zeroTolerance):
            newSurface = np.reshape(flattenedOperator, self.mySpace.functionSpaceDimension)
            return self
        self.surface = newSurface

#ZERO OPERATORS
class zeroMomentumOperator(zeroNuclearOperator):
    "Momentum Nuclear Operator smart enough to know it's zero"
    nonZeroOperatorClass = momentumOperator
    def __init__(self, space):
        self.mySpace = space
        self.surface = 0.0j
#define a reference to this zero class in the nonzero class
momentumOperator.zeroOperatorClass = zeroMomentumOperator

class zeroPositionOperator(zeroNuclearOperator):
    "Position Nuclear Operator smart enough to know it's zero"
    nonZeroOperatorClass = positionOperator
    def __init__(self, space):
        self.mySpace = space
        self.surface = 0.0j
#define a reference to this zero class in the nonzero class
positionOperator.zeroOperatorClass = zeroPositionOperator

#CONSTANT OPERATORS
class constantPositionNuclearOperator(positionOperator):
    "A Nuclear Operator Constant Over all position-space"
    def __init__(self, space, constant):
        zero = space.functionSpaceZero()
        self.omega_low = constant
        self.omega_high = constant
        super(constantPositionNuclearOperator, self).__init__(space, zero + constant)


class constantMomentumNuclearOperator(momentumOperator):
    "A Nuclear Operator Constant Over all momentum-space"
    def __init__(self, space, constant):
        zero = space.functionSpaceZero()
        self.omega_low = 0.0
        self.omega_high = 0.0
        super(constantMomentumNuclearOperator, self).__init__(space, zero + constant)

#FUNCTIONAL OPERATORS
class functionalPositionNuclearOperator(positionOperator):
    "A Nuclear Operator defined as a funtion over position-space"
    def __init__(self, space, function):
        outSurf = space.multivariableFunctionValueInX(function)
        self.omega = 0
        super(functionalPositionNuclearOperator, self).__init__(space, outSurf)

class functionalMomentumNuclearOperator(momentumOperator):
    "A Nuclear Operator defined as a function over momentum-space"
    def __init__(self, space, function):
        outSurf = space.multivariableFunctionValueInK(function)
        self.omega_low = 0.0
        self.omega_high = 0.0
        super(functionalMomentumNuclearOperator, self).__init__(space, outSurf)


class userSpecifiedSurface_positionOperator(positionOperator):
    "A Nuclear Operator defined as a funtion over position-space"
    def __init__(self, space, data):
        outSurf = data
        self.omega = 0
        self.omega_low = 0
        self.omega_high = 0
        super(userSpecifiedSurface_positionOperator, self).__init__(space, outSurf)


class nuclearHamiltonian(object):
    """Object to define the kinetic and potential energies of a nuclear environment

    MUST give either a list of one dimensional hamiltonians or a list of masses and a potential function"""

    def __init__(self, space,
                 listOfOneDimensionalHamiltonians = None,
                 listOfMasses = None,
                 potentialEnergyFunction = None):
        self.mySpace = space

        n = self.mySpace.nuclearDimensionality
        #figure out how the Hamiltonian was defined and make sure it's valid given the space
        if listOfOneDimensionalHamiltonians is not None:
            if self.mySpace.nuclearDimensionality != len(listOfOneDimensionalHamiltonians):
                raise nuclearHamiltonianDimensionMismatch()
            self.uncoupled = True
        else: #if the user did not supply a list of one dimenstional hamiltonians, make sure there are enough masses
            if len(inspect.getargspec(potentialEnergyFunction)[0]) != n and n != len(listOfMasses):
                raise nuclearHamiltonianDimensionMismatch()
            self.uncoupled = False


        if self.uncoupled:
            #set potential and kinetic surfaces
            #get the poentials of each coordinate
            listOfPotentialFunctionHandles = map(lambda x: x.potentialFunction, listOfOneDimensionalHamiltonians)
            #take the each coordinate's potential V(x1), v(x2), ... and turn into V(x1)+V(x2)+... etc for every possible point
            potentialSurface = self.mySpace.returnAmplitudeFromListOfFunction_handles(listOfPotentialFunctionHandles, additive=True, kspace = False)
            #store the surface
            self.myPotentialOperator = positionOperator(self.mySpace, potentialSurface)
            self.listOfOneDimensionalHamiltonians = listOfOneDimensionalHamiltonians
            #do the same for the kinetic energy operator
            listOfKineticFunctionHandles = map(lambda x: x.kineticFunction, listOfOneDimensionalHamiltonians)
            kineticSurface = self.mySpace.returnAmplitudeFromListOfFunction_handles(listOfKineticFunctionHandles, additive=True, kspace = True)
            self.myKineticOperator = momentumOperator(self.mySpace, kineticSurface)

            self.omega_low = min(map(lambda x: x.omega_low, listOfOneDimensionalHamiltonians))
            self.omega_high = max(map(lambda x: x.omega_high, listOfOneDimensionalHamiltonians))

        else:
            #set potential and kinetic surfaces
            #we just need to set the potential surface using the given function
            potentialSurface = self.mySpace.multivariableFunctionValueInX(potentialEnergyFunction)
            self.myPotentialOperator = positionOperator(self.mySpace, potentialSurface)
            #create a list of kinetic function values from all the given masses
            kineticValuesWithoutMass = np.power(self.mySpace.kValues, 2.0) * self.mySpace.hbar()**2.0 / 2.0
            listOfKineticFunctionValues = map(lambda m: kineticValuesWithoutMass / m, listOfMasses)
            #store these
            kineticSurface = self.mySpace.returnAmplitudeFromListOfFunctionValues(listOfKineticFunctionValues, additive=True)
            self.myKineticOperator = momentumOperator(self.mySpace, kineticSurface)
            self.omega_low = 0.0
            self.omega_high = 0.0
            raise Warning("Unset characteristic vibrational frequency for nuclear hamiltonians defined through multivariate functions!")

    def APPLY(self, nuclearWavefunction):
        "Takes a nuclear Psi and turns it into H*Psi"
        appliedKinetic = self.myKineticOperator.APPLY(nuclearWavefunction)
        appliedPotential = self.myPotentialOperator.APPLY(nuclearWavefunction)
        return appliedKinetic + appliedPotential


    def energyEigenfunctionAmplitude(self, listOfQuantumNumbers):
        "Takes a list of quantum numbers correspdonging to nuclear coordinates and gives the eigenfunction amplitude for the list"
        if not self.uncoupled:
            raise Exception("Eigenstates are only defined for uncoupled Hamiltonians")
        if len(listOfQuantumNumbers) != self.mySpace.nuclearDimensionality:
            raise Exception("Insufficient quantum Numbers")
        eigenfunction_handles_list = []
        for i, n in enumerate(listOfQuantumNumbers):
            eigenfunction_handles_list.append(self.listOfOneDimensionalHamiltonians[i].energyEigenfunction(n))
        return self.mySpace.returnAmplitudeFromListOfFunction_handles(eigenfunction_handles_list, additive = False, kspace = False)

    def energyEigenvalue(self, listOfQuantumNumbers):
        "Takes a list of quantum numbers correspdonging to nuclear coordinates and gives the energy for the list"
        if not self.uncoupled:
            raise Exception("Eigenstates are only defined for uncoupled Hamiltonians")
        if len(listOfQuantumNumbers) != self.mySpace.nuclearDimensionality:
            raise Exception("Insufficient quantum Numbers")
        energy = 0.0
        for i, hamiltonian in enumerate(self.listOfOneDimensionalHamiltonians):
            energy = energy + hamiltonian.energyEigenvalue(listOfQuantumNumbers[i])
        return energy

    def groundStateAmplitude(self):
        "Ground State amplitude of each 1D nuclear hamiltonian"
        zeros = []
        for i in range(self.mySpace.nuclearDimensionality):
            zeros.append(0.0)
        return self.energyEigenfunctionAmplitude(zeros)

    def randomWavefunctionDrawnFromThermalDistributions(self, temperatureInKelvin):
        "Take each 1D hamiltonian, randomly draw a thermal state from it and return a nuclear WF from those"
        wavefunctionValuesList = []
        for i, hamiltonian in enumerate(self.listOfOneDimensionalHamiltonians):
            wavefunctionValuesList.append(hamiltonian.drawEigenstateFromThermalDistribution(temperatureInKelvin))
        newAmplitude = self.mySpace.returnAmplitudeFromListOfFunctionValues(wavefunctionValuesList)
        newWF = NuclearWavefunction.nuclearWavefunction(self.mySpace, self)
        newWF.xAmplitude = newAmplitude
        return newWF

    def groundStateEnergy(self):
        "Ground State energy of each 1D nuclear hamiltonian"
        zeros = []
        for i in range(self.mySpace.nuclearDimensionality):
            zeros.append(0.0)
        return self.energyEigenvalue(zeros)

    def myNuclearPropagator(self, overRideDT = None, HIGH_ACCURACY=False, LOW_ACCURACY=False):
        "Returns a nuclear propagator corresponding to this Hamiltonian and the given time space"
        if LOW_ACCURACY:
            return nuclearPropagatorLOW_ACCURACY(self.mySpace, self, overRideDT)
        elif HIGH_ACCURACY:
            return nuclearPropagatorHIGH_ACCURACY(self.mySpace, self, overRideDT)
        else:
            return nuclearPropagator(self.mySpace, self, overRideDT)

    def frankCondonOverlapSpectrum(self, otherNuclearHamiltonian, numberOfLevelsToCalculate):
        "Function to iterate over the (approximately-first) first n energy levels and generate the "
        nLevels = numberOfLevelsToCalculate
        dim = self.mySpace.nuclearDimensionality
        maxN = int(nLevels / dim)

        iterators = []
        for i in range(dim):
            iterators.append(range(maxN))

        calculatedLevels = 0

        groundStateQuantumNumbers = np.zeros(dim)
        myGroundStateAmplitude = self.energyEigenfunctionAmplitude(groundStateQuantumNumbers)
        groundStateEnergy = self.energyEigenvalue(groundStateQuantumNumbers)

        energies = []
        overlaps = []

        for quantumNumbers in itertools.product(*iterators):
            if calculatedLevels > nLevels:
                break
            absEnergy = otherNuclearHamiltonian.energyEigenvalue(quantumNumbers) - groundStateEnergy
            thiStateFunction = otherNuclearHamiltonian.energyEigenfunctionAmplitude(quantumNumbers)
            overlap =  self.mySpace.xIntegration(myGroundStateAmplitude * np.conj(thiStateFunction))
            energies.append(absEnergy)
            overlaps.append(overlap * np.conj(overlap))
            calculatedLevels = calculatedLevels + 1

        return energies, overlaps
##################################################################
###NUCLEAR PROPAGATORS
class nuclearPropagatorLOW_ACCURACY(nuclearOperator):
    "Class which uses split operator porpagation to second order in dt"
    def __init__(self, space, nuclearHamiltonian, overRideDT=None):
        self.myHamiltonian = nuclearHamiltonian
        self.mySpace = space
        if overRideDT is None:
            self.dt = self.mySpace.dt
        else:
            self.dt = overRideDT

        #set up the kinetic propagator,
        kineticSurface = self.myHamiltonian.myKineticOperator.surface
        kineticSurface = np.exp(-1.0j *self.dt * kineticSurface / self.mySpace.hbar)
        self.myKineticOperator = momentumOperator(self.mySpace, kineticSurface)

        #set up the potential propagator
        potentialSurface = self.myHamiltonian.myPotentialOperator.surface
        potentialSurface = np.exp(-1.0j * self.dt * potentialSurface / self.mySpace.hbar)

        self.myPotentialOperator = positionOperator(self.mySpace, potentialSurface)

    def APPLY(self, wavefunction):
        wavefunction = self.myKineticOperator.APPLY(wavefunction)
        wavefunction = self.myPotentialOperator.APPLY(wavefunction)
        return wavefunction

class nuclearPropagator(nuclearOperator):
    "Class which uses split operator porpagation to third order in dt"
    def __init__(self, space, nuclearHamiltonian, overRideDT=None):
        self.myHamiltonian = nuclearHamiltonian
        self.mySpace = space
        if overRideDT is None:
            self.dt = self.mySpace.dt
        else:
            self.dt = overRideDT

        #set up the kinetic propagator,
        kineticSurface = self.myHamiltonian.myKineticOperator.surface
        kineticSurface = np.exp(-1.0j * .5 *self.dt * kineticSurface / self.mySpace.hbar)
        self.myKineticOperator = momentumOperator(self.mySpace, kineticSurface)

        #set up the potential propagator
        potentialSurface = self.myHamiltonian.myPotentialOperator.surface
        potentialSurface = np.exp(-1.0j * self.dt * potentialSurface / self.mySpace.hbar)

        self.myPotentialOperator = positionOperator(self.mySpace, potentialSurface)

    def APPLY(self, wavefunction):
        wavefunction = self.myKineticOperator.APPLY(wavefunction)
        wavefunction = self.myPotentialOperator.APPLY(wavefunction)
        wavefunction = self.myKineticOperator.APPLY(wavefunction)
        return wavefunction

class nuclearPropagatorHIGH_ACCURACY(nuclearOperator):
    "Class which uses split operator propagation to fourth order in dt"
    def __init__(self, space, nuclearHamiltonian, overRideDT=None):
        #raise Exception("DO NOT USE HIGH-ACCURACY PROPAGATOR; FUNCTIONALITY NOT CODED YET")
        self.myHamiltonian = nuclearHamiltonian
        self.mySpace = space
        if overRideDT is None:
            self.dt = self.mySpace.dt
        else:
            self.dt = overRideDT

        gamma = 1.0 / (2.0 - 2.0**(1.0/3.0))
        lam = -1.0j * self.dt / self.mySpace.hbar
        #set up the kinetic propagator,
        kineticSurfaceLambda = self.myHamiltonian.myKineticOperator.surface * lam * .5
        kineticSurface1 = np.exp( gamma * kineticSurfaceLambda )
        kineticSurface2 = np.exp( (1.0 - gamma) * kineticSurfaceLambda )
        self.myKineticOperator1 = momentumOperator(self.mySpace, kineticSurface1)
        self.myKineticOperator2 = momentumOperator(self.mySpace, kineticSurface2)

        #set up the potential propagator
        potentialSurfaceLambda = self.myHamiltonian.myPotentialOperator.surface * lam
        potentialSurface1 = np.exp( gamma * potentialSurfaceLambda )
        potentialSurface2 = np.exp( (1.0 - 2.0 *gamma) * potentialSurfaceLambda )
        self.myPotentialOperator1 = positionOperator(self.mySpace, potentialSurface1)
        self.myPotentialOperator2 = positionOperator(self.mySpace, potentialSurface2)


    def APPLY(self, wavefunction):
        wavefunction = self.myKineticOperator1.APPLY(wavefunction)
        wavefunction = self.myPotentialOperator1.APPLY(wavefunction)

        wavefunction = self.myKineticOperator2.APPLY(wavefunction)
        wavefunction = self.myPotentialOperator2.APPLY(wavefunction)
        wavefunction = self.myKineticOperator2.APPLY(wavefunction)

        wavefunction = self.myPotentialOperator1.APPLY(wavefunction)
        wavefunction = self.myKineticOperator1.APPLY(wavefunction)
        return wavefunction

########DEFINE SOLUBLE HAMILTONIANS
##################################

class oneDimentionalHamiltonian(object):
    "Abstract class to define a soluble single-variable hamiltonian"
    def potentialFunction(self):
        raise NotImplementedError("Must implement a potential function!")

    def energyEigenfunction(self, n):
        raise NotImplementedError("Must implement eigenfunction method!")

    def energyEigenvalue(self, n):
        raise NotImplementedError("Must implement eigenfunction method!")

    def kineticFunction(self, k):
        return self.mySpace.hbar**2.0 * k**2.0 / (2.0 * self.mass)


    def overlap(self, otherOneDHamiltonian, excitedQuantumNumber, myQuantumNumber):
        "takes the other basis function as being the bra"
        raise Exception("Not coded for rectangles yet!")
        if(excitedQuantumNumber < 0 or myQuantumNumber < 0 ):
            return 0.0
        myState = self.energyEigenfunction(myQuantumNumber)
        dx = self.mySpace.Dx
        otherStateFunction = otherOneDHamiltonian.energyEigenfunction(excitedQuantumNumber)
        overlap =  scipy.integrate.simps(myState * np.conj(otherStateFunction), dx = dx)
        return overlap




class harmonicOscillator(oneDimentionalHamiltonian):
    "1D Hamiltonian for a Quantum system where V(x) = kx^2"
    def __init__(self,
                 space,
                 omega=1.0,
                 mass=1.0,
                 center=0.0,
                 energyOffset = 0.0):
        self.mySpace = space
        self.mass = mass
        self.omega = omega
        self.center = center
        self.energyOffset = energyOffset

        self.alpha0 = self.mass * self.omega / ( 2.0 * self.mySpace.hbar )

        self.omega_low = omega
        self.omega_high = omega

        self.sigma = np.sqrt(self.mySpace.hbar / (self.mass * self.omega))

    def potentialFunction(self, x):
        return .5 * self.mass * self.omega**2.0 * (x - self.center)**2.0 + self.energyOffset

    def energyEigenfunction(self, n):
        n = int(n)
        if n < 0:
            return lambda x: 0.0*x #exists for compatability in certain functions
        firstTerm= (2.0**n * math.factorial(n))**(-0.5)
        secondTerm = (self.mass * self.omega / (np.pi * self.mySpace.hbar))**.25
        coefficients = np.zeros(n+2)
        coefficients[n]= 1
        return lambda x_values: firstTerm * secondTerm * np.exp(- self.mass * self.omega * (x_values - self.center)**2.0 / (2.0 * self.mySpace.hbar)) * herm.hermval(np.sqrt(self.mass * self.omega / self.mySpace.hbar) * (x_values - self.center), coefficients)


    def energyEigenvalue(self, n):
        if n < 0:
            return 0.0
        return self.mySpace.hbar * self.omega * (.5 + n) + self.energyOffset

    def analyticWavepacket(self, x0, p0, alpha0, t):
        "For Debugging purposes, analytic solution for wavepacket motion from Tannor"
        xt = x0 * math.cos(self.omega * t) + (p0 / (self.mass * self.omega)) * math.sin(self.omega * t)
        pt = p0 * math.cos(self.omega * t) - self.mass * self.omega * x0 * math.sin(self.omega * t)
        a = self.mass * self.omega / (2.0 * self.mySpace.hbar)
        alphat = a * (alpha0 * math.cos(self.omega * t) + 1.0j * a * math.sin(self.omega * t)) / ( 1.0j * alpha0 * math.sin(self.omega * t) + a * math.cos(self.omega * t))
        gammat = .5 * (pt * xt - p0 * x0) + 0.5j * self.mySpace.hbar * np.log(( 1.0j * alpha0 * math.sin(self.omega * t) + a * math.cos(self.omega * t)) / 2.0)
        x = self.mySpace.xValues
        return np.exp(-alphat * (x - xt)**2 + (1.0j/ self.mySpace.hbar) * pt * (x - xt) + (1.0j/ self.mySpace.hbar) * gammat)


    def analyticParitionFunctionValue(self, temperatureInKelvin):
        "Canonical Partition Function Value for this Hamiltonian"
        thermalEnergy = self.mySpace.unitHandler.BOLTZMANNS_CONSTANT_JOULES_PER_KELVIN * temperatureInKelvin
        thermalEnergy = self.mySpace.unitHandler.energyUnitsFromJoules(thermalEnergy)
        partitionEnergy = self.mySpace.hbar * self.omega * (.5)
        return 0.5 * math.sinh(partitionEnergy / thermalEnergy)**-1.0

    def proportionAtStateNAndTemperatureT(self, stateN, temperatureInKelvin):
        "Thermal state proportion for this Hamiltonian"
        thermalEnergy = self.mySpace.unitHandler.BOLTZMANNS_CONSTANT_JOULES_PER_KELVIN * temperatureInKelvin
        thermalEnergy = self.mySpace.unitHandler.energyUnitsFromJoules(thermalEnergy)
        partitionEnergy = self.mySpace.hbar * self.omega * (.5 + stateN)
        boltzmanFactor = math.exp(-partitionEnergy / thermalEnergy)
        return boltzmanFactor / self.analyticParitionFunctionValue(temperatureInKelvin)

    def drawEigenstateFromThermalDistribution(self, temperatureInKelvin):
        "Returns an eigenfunciton randomly chosen from the thermal distribution"
        u = np.random.rand()
        n=0
        cumulativeProbability = 0.0
        while True:
            cumulativeProbability = cumulativeProbability + self.proportionAtStateNAndTemperatureT(n, temperatureInKelvin)
            if u < cumulativeProbability:
                return self.energyEigenfunction(n)
            else:
                n = n + 1


    def coherentThermalState(self, temperatureInKelvin, seriesTruncation = 100):
        "Returns the coherent thermal state... for some reason"
        amplitude = self.energyEigenfunction(0)
        groundCoefficient = self.proportionAtStateNAndTemperatureT(0, temperatureInKelvin)**.5
        amplitude = amplitude * groundCoefficient
        for ii in range(1, seriesTruncation):
            newCoefficient = self.proportionAtStateNAndTemperatureT(ii, temperatureInKelvin)**.5
            newAmplitude = self.energyEigenfunction(ii)
            amplitude = amplitude + newAmplitude * newCoefficient
        return amplitude


class userSpecifiedPotential_Hamiltonian(oneDimentionalHamiltonian):
    "Used to make a 1D Hamiltonian with arbitrary V(x), perhaps from calculated data.  Can take in as input calculated wavefunctions as well in a dictionary indexed by quantum number"
    def __init__(self, space, mass, potential_surface, xValues, spatial_index = 0, add_absorbing_boundary_right = True, wavefunction_dictionary = None):
        self.mass = mass
        self.mySpace = space

        self.potential_surface = potential_surface
        self.xValues = xValues
        self.dx = xValues[1] - xValues[0]

        self.x_min_index = np.argmin(self.potential_surface)
        self.x_min = self.xValues[self.x_min_index]

        self.closest_harmonic_oscillator = self.bestGuessHamonicOscillator()

        self.omega_low = self.closest_harmonic_oscillator.omega
        self.omega_high = self.closest_harmonic_oscillator.omega

        self.spatial_index = spatial_index

        self.found_ground_state = False

        off_amplitude = 1E-3
        percentage_x_space = .05

        x_0 = xValues[-1]
        x_s = x_0 - self.dx * xValues.shape[0] * percentage_x_space
        if add_absorbing_boundary_right:
            self._Abs_amplitude = np.abs(np.std(self.potential_surface))
            #we want 5% of the points in simulation space to have imaginary potential so we solve for k:
            k = (x_s - x_0)/np.sqrt(np.arccosh(np.sqrt(1.0 / off_amplitude)))
            im_pot = -1.0j *self._Abs_amplitude * np.cosh( (np.real(xValues) - x_0)**2 / (k)**2)**(-2)
        else:
            im_pot = 0.0j

        self.potential_surface = self.potential_surface + im_pot

        if wavefunction_dictionary is not None:
            self.max_n = max( [x for x in wavefunction_dictionary.keys() if not isinstance(x, str)])
            self.available_energies = wavefunction_dictionary['E']
            self.available_wf_amplitudes = {}
            for i in range(self.max_n + 1):
                dat = wavefunction_dictionary[i]
                self.available_wf_amplitudes[i] = dat
        else:
            self.max_n = -1


    def potentialFunction(self, x):
        interpolant = scipy.interpolate.interp1d(self.xValues + 0.0j, self.potential_surface, bounds_error=False, fill_value="extrapolate")
        v = interpolant(x)
        return v

    def energyEigenfunction(self, n):
        if n < self.max_n and n >=0:
            dat = self.available_wf_amplitudes[n]
            interpolator_function = scipy.interpolate.interp1d(self.xValues, dat, bounds_error=False, fill_value="extrapolate")
            return interpolator_function
        else:
            warnings.warn("THIS IS CALL FOR AN EIGENFUNCTION TO A HAMILTONIAN WITH A USER SPECIFIED POTENTIAL IS JUST FOR THE BEST-APPROXIMATION HARMONIC OSCILLATOR")
            return self.closest_harmonic_oscillator.energyEigenfunction(n)


    def energyEigenvalue(self, n):
        if n < self.max_n and n >=0:
            return self.available_energies[n]
        else:
            warnings.warn("THIS IS CALL FOR AN EIGENVALUE TO A HAMILTONIAN WITH A USER SPECIFIED POTENTIAL IS JUST FOR THE BEST-APPROXIMATION HARMONIC OSCILLATOR")
            return self.closest_harmonic_oscillator.energyEigenvalue(n)

    def bestGuessHamonicOscillator(self):
        "Find the HO Approximation to this Hamiltonian, using the minimum point on the surface"
        grad_squared = np.gradient(np.gradient(self.potential_surface)) / (self.dx)**2
        k = grad_squared[self.x_min_index]
        omega_guess = np.sqrt(k / self.mass)
        offset_value = self.potential_surface[self.x_min_index]

        HO_guess = harmonicOscillator(self.mySpace,
                                     omega=omega_guess,
                                     mass=self.mass,
                                     center=self.x_min,
                                     energyOffset = offset_value)

        return HO_guess


class morsePotential(oneDimentionalHamiltonian):
    """Used to create a 1D Hamiltonian object with a V(x) of the form of a morse oscillator
    For implementation details, see the wikipedia article on the morse potential"""
    def __init__(self,
                 space,
                 a=.25,
                 De=1.0,
                 mass=1.0,
                 center=0.0,
                 energyOffset = 0.0,
                 absorbingPotential = True):
        """give an omega, mass and center value and possibly an energyOffsetValue"""
        self.mySpace = space
        self.a = a
        self.mass = mass
        self.De = De
        self.center = center
        self.energyOffset = energyOffset

        self.omega_0 = self.a * math.sqrt(2 * self.De / self.mass)
        self.omega = self.omega_0

        self.maxN = int((2.0 * self.De ) / self.omega_0) - 1

        self.absorbingPotential = absorbingPotential
        if absorbingPotential:
            self.startOfAbsorbingPotential = self.center / 3.0
            self.strength = .001

        omega_levels = []
        for i in range(0, self.maxN + 1):
            omega = self.energyEigenvalue(i)
            omega_levels.append(omega)


        self.omega_low = min(omega_levels)
        self.omega_high = max(omega_levels)

    def potentialFunction(self, x):
        naturalPotential = self.De * (1 - np.exp(-self.a * (x - self.center)))**2 + self.energyOffset
        imaginaryPotential = 0
        try:
            imaginaryPotentialZeros = np.zeros(x.shape)
        except:
            if x > self.startOfAbsorbingPotential:
                imaginaryPotential = -1.0j *self.strength * np.cosh( (np.real(x) - self.mySpace.xMax)**2 / (self.mySpace.Dx*30)**2)**(-2)
            else:
                imaginaryPotential = 0
            return naturalPotential + imaginaryPotential

        if self.absorbingPotential:
            imaginaryPotential = -1.0j *self.strength * np.cosh( (np.real(x) - self.mySpace.xMax)**2 / (self.mySpace.Dx*30)**2)**(-2)
            imaginaryPotential = np.where(x > self.startOfAbsorbingPotential, imaginaryPotential, imaginaryPotentialZeros)
        return naturalPotential + imaginaryPotential

    def energyEigenfunction(self, n):
        if n < 0:
            return lambda x: 0.0*x

        #constants
        Lambda = np.sqrt(2.0 * self.mass * self.De) / (self.a * self.mySpace.hbar)
        alpha = 2.0 * Lambda - 2.0*n - 1.0

        #sympy functions
        normalizer = np.sqrt( self.a * ( 2.0 * Lambda - 2.0*n - 1.0 )  / (sympy.mpmath.gamma(n+1) * sympy.mpmath.gamma(2*Lambda - n) )) * float(fact.factorial(n))

        totalFunction = lambda x: normalizer * sympy.mpmath.power(2.0 * Lambda * sympy.exp(-self.a * (x - self.center)), Lambda - n - .5) * sympy.mpmath.exp(-Lambda * sympy.exp(-self.a * (x - self.center))) * sympy.mpmath.laguerre(n, alpha, self.a * (x - self.center))
        warnings.warn("I'm not yet entirely sure this will work with the rectangle code...  Further testing needed")
        return totalFunction

    def energyEigenvalue(self, n):
        if n < 0:
            return 0.0
        nPlusHalf = n + .5
        hbarOmegaNPlusHalf = self.mySpace.hbar * self.omega_0 * nPlusHalf
        return hbarOmegaNPlusHalf - hbarOmegaNPlusHalf**2.0 / (4.0 * self.De)

    def correspondingHarmonicOscillator(self):
        output = harmonicOscillator(self.mySpace, self.omega_0, self.mass, self.center, self.energyOffset)
        return output



class nuclearHamiltonianDimensionMismatch(Exception):
    def __init__(self, value=0):
        self.value = value

    def __str__(self):
        return "Given spacetime expects different number of nuclear coordinates"


if __name__ == "__main__":
    #Some Test Code
    maxIndex = 3
    ##1 dimensional test
    mySpace = Spacetime.Spacetime(xMax = 10,
                 numberOfNuclearDimenions = 1,
                 numberOfElectronicDimensions = 2,
                 numberOfSimulationSpacePointsPerNuclearDimension = 100,
                 dt = .1)
    testHarmonicOscillator = harmonicOscillator(mySpace,
                 omega=1,
                 mass=1,
                 center=0,
                 energyOffset = 0)
    testNuclearHamiltonian = nuclearHamiltonian(mySpace,
                 listOfOneDimensionalHamiltonians = [testHarmonicOscillator] )
    testNuclearWF = NuclearWavefunction.nuclearWavefunction(mySpace,
                                        groundStateNuclearHamiltonian = testNuclearHamiltonian)


    HAppliedToTestWF = testNuclearHamiltonian.APPLY(testNuclearWF)

    #Try applying H for a range of eigenfunctions
    for i in range(maxIndex):
        qnList = [i]
        testNuclearWF = NuclearWavefunction.nuclearWavefunction(mySpace,
                                        groundStateNuclearHamiltonian = testNuclearHamiltonian,
                                        listOfInitialQuantumNumbers = qnList)
        HAppliedToTestWF = testNuclearHamiltonian.APPLY(testNuclearWF)
        E = testNuclearHamiltonian.energyEigenvalue(qnList)
        plt.figure()
        plt.plot(mySpace.xValues, testNuclearWF.xAmplitude, label = r"$\Psi_0$")
        plt.plot(mySpace.xValues, HAppliedToTestWF.xAmplitude, label = r"$\mathcal{H}\Psi_0$")
        HPsiDivPsi =  HAppliedToTestWF.xAmplitude / testNuclearWF.xAmplitude
        plt.plot(mySpace.xValues, HPsiDivPsi, label = r"$\frac{\mathcal{H}\Psi}{\Psi}$")
        plt.xlim((-5, 5))
        plt.ylim((-2.5, E + .1))
        plt.title("n = %s" % str(i))
        plt.legend()
        eFromH = np.max(HAppliedToTestWF.xAmplitude) / np.max(testNuclearWF.xAmplitude)
        print E,  eFromH, E-eFromH
        assert E - eFromH < mySpace.DEFAULT_ZERO_TOLERANCE , """ Hpsi != Epsi"""


    ##2 dimensional test
    mySpace = Spacetime.Spacetime(xMax = 10,
                 numberOfNuclearDimenions = 2,
                 numberOfElectronicDimensions = 2,
                 numberOfSimulationSpacePointsPerNuclearDimension = 100,
                 dt = .1)

    testHarmonicOscillator1 = harmonicOscillator(mySpace,
                 omega=1,
                 mass=1,
                 center=0,
                 energyOffset = 0)

    testHarmonicOscillator2 = harmonicOscillator(mySpace,
                 omega=3,
                 mass=1,
                 center=0,
                 energyOffset = 0)

    testNuclearHamiltonian2D = nuclearHamiltonian(mySpace,
                 listOfOneDimensionalHamiltonians = [testHarmonicOscillator1, testHarmonicOscillator2] )



    #Try applying H for a range of eigenfunctions
    for i in range(maxIndex):
        for j in range(i+1):
            qnList = [i, j]
            testNuclearWF = NuclearWavefunction.nuclearWavefunction(mySpace,
                                            groundStateNuclearHamiltonian = testNuclearHamiltonian2D,
                                            listOfInitialQuantumNumbers = qnList)
            HAppliedToTestWF = testNuclearHamiltonian2D.APPLY(testNuclearWF)
            E = testNuclearHamiltonian2D.energyEigenvalue(qnList)
            plt.figure()
            plt.contourf(mySpace.xValues, mySpace.xValues, testNuclearWF.xAmplitude, label = r"$\Psi_0$")
            plt.title(r"$\Psi_{%s, %s}$" % (str(i), str(j)))
            plt.colorbar()

            plt.figure()
            plt.contourf(mySpace.xValues, mySpace.xValues, HAppliedToTestWF.xAmplitude, label = r"$\Psi_0$")
            plt.title(r"$\mathcal{H}\Psi_{%s, %s}$" % (str(i), str(j)))
            plt.colorbar()

            eFromH =  np.max(HAppliedToTestWF.xAmplitude) / np.max(testNuclearWF.xAmplitude) #np.abs(HPsiDivPsi[20,20])
            print E,  eFromH, E-eFromH
            assert E - eFromH < mySpace.DEFAULT_ZERO_TOLERANCE , """ Hpsi != Epsi"""



    mySpace = Spacetime.Spacetime(xMax = 10,
                 numberOfNuclearDimenions = 2,
                 numberOfElectronicDimensions = 2,
                 numberOfSimulationSpacePointsPerNuclearDimension = 80,
                 dt = .1)

    testMorseOscillator1 = morsePotential(mySpace,
                 a=.25,
                 De=1.0,
                 mass=1,
                 center=-3,
                 energyOffset = 0)

    testMorseOscillator2 = morsePotential(mySpace,
                 a=.25,
                 De=1.0,
                 mass=1,
                 center=-2,
                 energyOffset = 0)

    testNuclearHamiltonian2D = nuclearHamiltonian(mySpace,
                 listOfOneDimensionalHamiltonians = [testMorseOscillator1, testMorseOscillator2] )



    #Try applying H for a range of eigenfunctions
    for i in range(maxIndex):
        for j in range(i+1):
            qnList = [i, j]
            testNuclearWF = NuclearWavefunction.nuclearWavefunction(mySpace,
                                            groundStateNuclearHamiltonian = testNuclearHamiltonian2D,
                                            listOfInitialQuantumNumbers = qnList)
            HAppliedToTestWF = testNuclearHamiltonian2D.APPLY(testNuclearWF)
            E = testNuclearHamiltonian2D.energyEigenvalue(qnList)
            plt.figure()
            plt.contourf(mySpace.xValues, mySpace.xValues, testNuclearWF.xAmplitude, label = r"$\Psi_0$")
            plt.title(r"$\Psi_{%s, %s}$" % (str(i), str(j)))
            plt.colorbar()

#            plt.figure()
#            plt.contourf(mySpace.xValues, mySpace.xValues, testNuclearWF.xAmplitude, label = r"$\Psi_0$")
#            plt.title(r"$\mathcal{H}\Psi_{%s, %s}$" % (str(i), str(j)))
#            plt.colorbar()


            HPsiDivPsi =  HAppliedToTestWF.xAmplitude / testNuclearWF.xAmplitude
            plt.figure()
            plt.contourf(mySpace.xValues, mySpace.xValues, HPsiDivPsi,100, label = r"$\Psi_0$")
            plt.title(r"$\frac{\mathcal{H}\Psi_{%s, %s}}{\Psi_{%s, %s}}$" % (str(i), str(j), str(i), str(j)))
            plt.colorbar()

            eFromH =  np.max(HAppliedToTestWF.xAmplitude) / np.max(testNuclearWF.xAmplitude)
            print E,  eFromH, E-eFromH
            assert E - eFromH < mySpace.DEFAULT_ZERO_TOLERANCE , """ Hpsi != Epsi"""

    ##1 dimensional test of propagator
    mySpace = Spacetime.Spacetime(xMax = 10,
                 numberOfNuclearDimenions = 1,
                 numberOfElectronicDimensions = 2,
                 numberOfSimulationSpacePointsPerNuclearDimension = 200,
                 dt = .1)
    testHarmonicOscillator = harmonicOscillator(mySpace,
                 omega=1,
                 mass=1,
                 center=0,
                 energyOffset = 0)
    testOffsetHarmonicOscillator = harmonicOscillator(mySpace,
                 omega=2,
                 mass=1,
                 center=2,
                 energyOffset = 0)
    testNuclearHamiltonian = nuclearHamiltonian(mySpace,
                 listOfOneDimensionalHamiltonians = [testHarmonicOscillator] )
    testNuclearWF = NuclearWavefunction.nuclearWavefunction(mySpace,
                                        groundStateNuclearHamiltonian = testOffsetHarmonicOscillator)

    testNuclearPropagatorMid = testNuclearHamiltonian.myNuclearPropagator(HIGH_ACCURACY=False, LOW_ACCURACY=False)
    testNuclearPropagatorLow = testNuclearHamiltonian.myNuclearPropagator(HIGH_ACCURACY=False, LOW_ACCURACY=True)
    #testNuclearPropagatorHigh = testNuclearHamiltonian.myNuclearPropagator(HIGH_ACCURACY=True, LOW_ACCURACY=False)

    plt.plot(mySpace.xValues, testNuclearWF.xAmplitude)
    highWF = copy.copy(testNuclearWF)
    lowWF = copy.copy(testNuclearWF)
    midWF = copy.copy(testNuclearWF)

    lowNorms = [testNuclearWF.norm()]
    midNorms = [testNuclearWF.norm()]
    #highNorms = [testNuclearWF.norm()]

    for i in range(12):
        #highWF = testNuclearPropagatorHigh.APPLY(highWF)
        lowWF = testNuclearPropagatorLow.APPLY(lowWF)
        midWF = testNuclearPropagatorMid.APPLY(midWF)

        plt.figure()
        plt.title(str(i))
        plt.plot(mySpace.xValues, np.abs(highWF.xAmplitude), label="high accuracy")
        plt.plot(mySpace.xValues, np.abs(lowWF.xAmplitude), label="low accuracy")
        plt.plot(mySpace.xValues, np.abs(midWF.xAmplitude), label="mid accuracy")
        plt.legend()

        lowNorms.append(lowWF.norm())
        midNorms.append(midWF.norm())
        #highNorms.append(highWF.norm())

    plt.figure()
    plt.plot(lowNorms, label="Low")
    plt.plot(midNorms, label="mid")
    #plt.plot(highNorms, label="high")
    plt.title("norms")
    #plt.yscale('log')
    plt.legend()
