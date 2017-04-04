# -*- coding: utf-8 -*-
#Gone over for release 10 Feb 2017
import time
import math

import numpy as np
import scipy
import scipy.integrate
import matplotlib.pyplot as plt
import warnings

import TimeFunction


class unitHandler(object):
    "Abstract Class to handle units for Spacetime object"
    BOLTZMANNS_CONSTANT_JOULES_PER_KELVIN = 1.3806488E-23 #Joules per Kelvin
    ELECTRON_MASS_KILOGRAMS = 9.10938E-31 #kg
    HBAR_JOULE_SECONDS      = 1.054572E-34
    H_JOULE_SECONDS         = 2.0 * np.pi * HBAR_JOULE_SECONDS
    C_METERS_PER_SECOND     = 2.998E8
    AMU_KILOGRAMS           = 1.660468E-24
    #ENERGY
    def energyUnitsFromWavenumbers(self, wavenumbers):
        return wavenumbers / self.energyScaleWavenumbers

    def energyUnitsFromJoules(self, joules):
        return joules / self.energyScaleJoules

    def wavenumbersFromEnergyUnits(self, energies):
        return energies * self.energyScaleWavenumbers

    #TIME
    def timeUnitsFromSeconds(self, seconds):
        return seconds / self.timeScaleSeconds

    def femtosecondsFromTime(self, time):
        return time * self.timeScaleSeconds * 10.0**(15.0)

    #LENGTH
    def lengthUnitsFromMeters(self, meters):
        return meters / self.lengthScaleMeters

    def lengthUnitsFromBohrRadii(self, BohrRadii):
        return BohrRadii * self.UnityMassInElectronMasses

    def metersFromLengthUnits(self, length):
        return length * self.lengthScaleMeters

    #MASS
    def massUnitsFromAmu(self, amu):
        return amu / self.massScaleAmu


class wavenumberBasedUnits(unitHandler):
    "System of units based on 1 energy unit being x number of wavenumbers and the mass of the iodine diatom being 1"
    HBAR = 1.0
    H_C  = 1.0
    H = HBAR * 2.0 * np.pi
    C = H_C / H

    def __init__(self, numberOfWavenumbersInEnergyUnit = 100.0):
        self.SCALE_JOULES = 1.986446E-23 *  numberOfWavenumbersInEnergyUnit # joules, numberOfWavenumbersInEnergyUnit wavenumbers
        self.SCALE_MASS   = 2.107298E-25 # kilograms, the mass of an iodine atom ARBITRARILY CHOSEN

        self.SCALE_TIME   = unitHandler.HBAR_JOULE_SECONDS / self.SCALE_JOULES  # seconds, hbar / E_0
        self.SCALE_LENGTH = unitHandler.HBAR_JOULE_SECONDS / math.sqrt( self.SCALE_JOULES *  self.SCALE_MASS)  # meters, hbar (E_0 m_0)^-.5

        self.energyUnit = numberOfWavenumbersInEnergyUnit #wavenumbers

        self.energyScaleWavenumbers = self.energyUnit

        self.lengthScaleMeters = self.SCALE_LENGTH
        self.timeScaleSeconds = self.SCALE_TIME

        self.energyScaleJoules = self.SCALE_JOULES

        self.massScaleAmu = self.SCALE_MASS / unitHandler.AMU_KILOGRAMS

    def distanceUnitsFromSqrtCentimeters(self, sqrtCm):
        #IS THIS RIGHT!?!?!?
        warnings.warn("untested distanceUnitsFromSqrtCentimeters")
        return sqrtCm / np.sqrt(self.energyScaleWavenumbers)
    def sqrtCentimetersFromDistanceUnits(self, dist):
        return dist * np.sqrt(self.energyScaleWavenumbers)


class scaledAtomicUnits(unitHandler):
    "Represents standard atomic units which can have  as unity mass any multiple of m_e"
    SCALE_METERS = 5.29e-11 #meters per alu
    SCALE_SECONDS = 2.419E-17 #seconds per atu
    SCALE_JOULES = 4.3597E-18 #joules per aeu
    SCALE_AMU = 1.0 / 1822.81 #proton mass per electron mass


    ALPHA = 0.00729735257

    SCALE_WAVENUMBERS = 219474.6 #wavenumbers per hartree

    HBAR = 1.0
    H = HBAR * 2.0 * np.pi
    COULUMB_CONSTANT = 1.0
    ELECTRON_CHARGE = 1.0

    SPEED_OF_LIGHT = 1.0 / ALPHA

    def __init__(self, UnityMassInElectronMasses = 1.0):

        self.UnityMassInElectronMasses = float(UnityMassInElectronMasses)

        self.electronMass = 1.0 / self.UnityMassInElectronMasses

        self.lengthScaleMeters = scaledAtomicUnits.SCALE_METERS / self.UnityMassInElectronMasses
        self.timeScaleSeconds = scaledAtomicUnits.SCALE_SECONDS / self.UnityMassInElectronMasses

        self.energyScaleJoules = scaledAtomicUnits.SCALE_JOULES * self.UnityMassInElectronMasses
        self.energyScaleWavenumbers = scaledAtomicUnits.SCALE_WAVENUMBERS * self.UnityMassInElectronMasses

        self.massScaleAmu = scaledAtomicUnits.SCALE_AMU * self.UnityMassInElectronMasses




class Spacetime(object):
    """Class which stores the parameters of function and hilbert space,
    as well as units.  Spacetimes are centered about zero in space but
    start at 0 for time ATOMIC UNITS"""


    DEFAULT_ZERO_TOLERANCE = 1e-8 #what the computer will treat as effectively zero
    GAUSSIAN_FILTER_FACTOR = 10 #tried a bunch and 10 seemed to be a sweet spot for getting spectral heights correct


    hbar = 1.0

    def __init__(self,
                 xMax_tuple,
                 numberOfNuclearDimenions,
                 numberOfElectronicDimensions,
                 numberOfSimulationSpacePointsPerNuclearDimension_tuple,
                 xMax_tuple_angstroms = None,
                 dt = None,
                 dt_SECONDS = None,
                 zeroTolerance = DEFAULT_ZERO_TOLERANCE,
                 UnityMassInElectronMasses = None,
                 wavenumberBasedUnits_scaling = None):

        #What we're OK with calling effectively 0
        self.zeroTolerance = zeroTolerance

        ##DEFINE Dimensionality
        self.nuclearDimensionality = numberOfNuclearDimenions
        self.electronicDimensionality = numberOfElectronicDimensions



        if wavenumberBasedUnits_scaling is not None:
            self.unitHandler = wavenumberBasedUnits(wavenumberBasedUnits_scaling)
        elif UnityMassInElectronMasses is not None:
            self.unitHandler = scaledAtomicUnits(UnityMassInElectronMasses)
        else:
            self.unitHandler = scaledAtomicUnits(1.0)


        Spacetime.hbar = self.unitHandler.HBAR

        #Define time
        if dt is not None:
            self.dt = float(dt)
        elif dt_SECONDS is not None:
            self.dt = float(self.unitHandler.timeUnitsFromSeconds(dt_SECONDS))
        else:
            raise Exception("Need to specify a time step dude!")



        if xMax_tuple == None and xMax_tuple_angstroms != None:
            xMax_tuple = tuple(map( lambda x: self.unitHandler.lengthUnitsFromMeters(x * 1e-10), xMax_tuple_angstroms))
            print "x will stretch from +/- in natural units: ",  xMax_tuple



        #Define Space
        assert(self.nuclearDimensionality == len(xMax_tuple))
        assert(self.nuclearDimensionality == len(numberOfSimulationSpacePointsPerNuclearDimension_tuple))


        self.xMax_values = map(lambda x: 1.0 * x, xMax_tuple)
        self.numberOfSimulationSpacePointsPerNuclearDimension_tuple = numberOfSimulationSpacePointsPerNuclearDimension_tuple
        self.xLength_values = map(lambda x: 2.0*x,  self.xMax_values)

        self.Dx_values = []
        self.xValues_list = []
        for i in range(self.nuclearDimensionality):
            assert numberOfSimulationSpacePointsPerNuclearDimension_tuple[i] % 4 == 0, "number of points in space is not divisible by 4"
            self.Dx_values.append(self.xLength_values[i] / float(self.numberOfSimulationSpacePointsPerNuclearDimension_tuple[i]))
            x_vals = np.linspace(-self.xMax_values[i], self.xMax_values[i] - self.Dx_values[i], self.numberOfSimulationSpacePointsPerNuclearDimension_tuple[i])
            self.xValues_list.append(np.array(x_vals, dtype=np.complex))

        ##Define MOMENTUM-SPACE PARAMETERS

        self.Dk_values = []
        self.kMax_values =  []
        self.kLength_values = []
        self.kValues_list = []
        for i in range(self.nuclearDimensionality):
            self.Dk_values.append(2 * np.pi / (float(self.numberOfSimulationSpacePointsPerNuclearDimension_tuple[i]) * self.Dx_values[i]))
            self.kMax_values.append(np.pi / self.Dx_values[i])
            self.kLength_values.append(2 * self.kMax_values[i])
            k_vals = np.linspace(-self.kMax_values[i], self.kMax_values[i] - self.Dk_values[i], self.numberOfSimulationSpacePointsPerNuclearDimension_tuple[i])
            k_vals = np.fft.fftshift(k_vals)
            self.kValues_list.append(np.array(k_vals, dtype=np.complex))


        ft_coefficient = 1.0
        ift_coefficient = 1.0
        for i in range(self.nuclearDimensionality):
            ft_coefficient = ft_coefficient * ( self.xLength_values[i] / self.numberOfSimulationSpacePointsPerNuclearDimension_tuple[i])
            ift_coefficient = ift_coefficient * (self.kLength_values[i] / (2.0 * np.pi))
        self.ft_coefficient = ft_coefficient
        self.ift_coefficient = ift_coefficient
        #Dimension of all arrays in x and k space
        self.functionSpaceDimension      = np.ones(self.nuclearDimensionality, dtype=np.int)*np.array(self.numberOfSimulationSpacePointsPerNuclearDimension_tuple, dtype=np.int)



    ##DEFINE ZERO FUNCTIONS FOR ALL RELEVANT SPACES
    def functionSpaceZero(self):
        "function to return a new spatial-zero function ndarray"
        return np.zeros(self.functionSpaceDimension, dtype=np.complex)


    def functionSpacetimeZero(self, numberOfTimeSteps):
        "function to return the zero fucntion in spacetime"
        arrayDimension = np.append(numberOfTimeSteps, self.functionSpaceDimension)
        out = np.zeros(arrayDimension, dtype=np.complex)
        return out


    #OTHER FUNCTION-DEFINING METHODS
    def returnAmplitudeFromListOfFunctionValues(self, listOfFunctionValues, additive=False):
        """Helper function for the amplitude setting function
        [f(x), g(y), ...]
        additive=True  => F(x, y, ...) = f(x) + g(y) + ...
        additive=False => F(x, y, ...) = f(x) * g(y) * ..."""
        output = self.functionSpaceZero()
        if additive:
            initialValue = 0.0
        else:
            initialValue = 1.0
        for indexTuple, value in np.ndenumerate(output):
            newValue = initialValue
            for tupleIndex, tupleValue in enumerate(indexTuple):
                if additive:
                    newValue += listOfFunctionValues[tupleIndex][tupleValue]
                else:
                    newValue *= listOfFunctionValues[tupleIndex][tupleValue]
            output[indexTuple] = newValue
        return output

    def returnAmplitudeFromListOfFunction_handles(self, listOfFunction_handles, additive=False, kspace = False):
        """Helper function for the amplitude setting function handles
        [f, g, ...]
        additive=True  => F(x, y, ...) = f(x) + g(y) + ...
        additive=False => F(x, y, ...) = f(x) * g(y) * ..."""
        function_values = []
        for i in range(self.nuclearDimensionality):
            if kspace:
                function_values.append(listOfFunction_handles[i](self.kValues_list[i]))
            else:
                function_values.append(listOfFunction_handles[i](self.xValues_list[i]))
        return self.returnAmplitudeFromListOfFunctionValues(function_values, additive)

    def multivariableFunctionValueInX(self, multivariableFunction):
        "Turns V(x, y, z, ...) into a function ampltiude"
        output =  self.functionSpaceZero()
        for indexTuple, value in np.ndenumerate(output):
            args = []
            for dimension_index, x_index in enumerate(indexTuple):
                args.append(self.xValues_list[dimension_index][x_index])
            output[indexTuple] = multivariableFunction(*args)
        return output

    def multivariableFunctionValueInK(self, multivariableFunction):
        "Turns V(x, y, z, ...) (but in k space) into a function ampltiude"
        output =  self.functionSpaceZero()
        for indexTuple, value in np.ndenumerate(output):
            args = []
            for dimension_index, k_index in enumerate(indexTuple):
                args.append(self.kValues_list[dimension_index][k_index])
            output[indexTuple] = multivariableFunction(*args)
        return output


    #Define the fourier Transform on this space
    def spatialFourierTransform(self, Amplitude):
        """Gives the FT of the given Amplitude"""
        return self.ft_coefficient * (np.fft.fftn(Amplitude))

    def spatialInverseFourierTransform(self, Amplitude):
        """Gives the inverse FT of the given Amplitude"""
        return self.ift_coefficient * (np.fft.ifftn(Amplitude))



    #INTEGRATION METHODS
    def xIntegration(self, Amplitude):
        "Integrates a given amplitude in x-space"
        toBeIntegrated = Amplitude
        for i in range(self.nuclearDimensionality):
            toBeIntegrated = scipy.integrate.simps(toBeIntegrated, dx=self.Dx_values[i])
        return toBeIntegrated

    def kIntegration(self, Amplitude):
        "Integrates a given amplitude in k-space"
        toBeIntegrated = Amplitude
        for i in range(self.nuclearDimensionality):
            toBeIntegrated = scipy.integrate.simps(toBeIntegrated, dx=self.Dk_values[i])
        return toBeIntegrated

    def tIntegration(self, timeAmplitude):
        "Integrates a given amplitude in time along the first axis"
        toBeIntegrated = timeAmplitude
        toBeIntegrated = scipy.integrate.simps(toBeIntegrated, dx=self.dt)
        return toBeIntegrated



    #@staticmethod
    def genericOneDimensionalFourierTransformFromZero(self, timeCoordinate, amplitudes, gaussianFilter = True):
        "Returns a tuple of the inverse coordinate and the fourier amplitudes"
        n = timeCoordinate.size
        omegas = 2.0 * np.pi * np.fft.fftfreq(n, timeCoordinate[1] - timeCoordinate[0])
        if gaussianFilter:
            tLength = timeCoordinate[-1] - timeCoordinate[0]
            if timeCoordinate[0] == 0.0:
                tCenter = 0.0
                stdev = tLength / Spacetime.GAUSSIAN_FILTER_FACTOR
            else:
                tCenter = timeCoordinate[0] + tLength
                stdev = tLength / Spacetime.GAUSSIAN_FILTER_FACTOR

            filterFunction = TimeFunction.GaussianCosinePulse(self, centerOmega = 0,  timeSpread = stdev, amplitude = 1.0, centerTime = tCenter)

            filterValues = filterFunction.valueAtTime(timeCoordinate)
            filteredData = filterValues * amplitudes

            filteredDataFFT =  np.fft.fftshift(np.fft.fft(filteredData))

            return np.fft.fftshift(omegas), filteredDataFFT
        #fftshift just flips the negative frequencies behind the positive frequencies
        return np.fft.fftshift(omegas), np.fft.fftshift(np.fft.fft(amplitudes))


    def peakFinder(self, x, y, threshold):
        #find the peaks
        DY = np.gradient(y)
        DY = DY / np.max(DY)

        indecesAboveThreshold, = np.where(y > threshold)
        peaksIndeces = []
        for index in indecesAboveThreshold:
            if np.real(DY[index - 1]) > 0.0 and np.real(DY[index + 1]) < 0.0:
                peaksIndeces.append(index)

        peakX = x[peaksIndeces]
        peakY = y[peaksIndeces]

        return peakX, peakY

    def rotateTuple(self, tupleToRotate, xAngle, yAngle, zAngle):
        "Returns copy of the tuple, rotated by three angles in 3-space"
        x = tupleToRotate[0]
        y = tupleToRotate[1]
        z = tupleToRotate[2]

        phi = xAngle
        theta = yAngle
        psi = zAngle

        R11 = np.cos(theta) * np.cos(psi)
        R21 = np.cos(theta) * np.sin(psi)
        R31 = - np.sin(theta)
        R12 = - np.cos(phi) * np.sin(psi) + np.sin(phi)*np.sin(theta)*np.cos(psi)
        R22 = np.cos(phi) * np.cos(psi) + np.sin(phi)*np.sin(theta)*np.sin(psi)
        R32 = np.sin(phi) * np.cos(theta)
        R13 = np.sin(phi)*np.sin(psi) + np.cos(phi)*np.sin(theta)*np.cos(psi)
        R23 = -np.sin(phi)*np.cos(psi) + np.cos(phi)*np.sin(theta)*np.sin(psi)
        R33 = np.cos(phi)*np.cos(theta)

        newX = R11 * x + R12 * y + R13 * z
        newY = R21 * x + R22 * y + R23 * z
        newZ = R31 * x + R32 * y + R33 * z

        return (newX, newY, newZ)

    def randomRotateTuple(self, tupleToRotate):
        "Returns a random angle and a copy of the tuple, rotated by three random angles in 3-space"
        xAngle = np.random.uniform(0.0, 2.0 * np.pi )
        yAngle = np.random.uniform(0.0, 2.0 * np.pi )
        zAngle = np.random.uniform(0.0, 2.0 * np.pi )

        return (xAngle, yAngle, zAngle), self.rotateTuple(tupleToRotate, xAngle, yAngle, zAngle)


    def plotSpaceFunction(self, spaceFunction):
        d = len(spaceFunction.shape)
        if d>2:
            print "NO PLOTTING FOR MORE THAN TWO DIMENSIONS"
            return None
        if d==1:
            x = self.xValues
            y = spaceFunction
            fig = plt.figure()
            plt.plot(x, y)
            return fig
        else:
            x = self.xValues
            y = self.xValues
            z = spaceFunction
            fig = plt.figure()
            plt.contourf(x, y, z)
            return fig



class MismatchedSpacetimeError(Exception):
    def __str__(self):
        return "Given arguments do not occupy the same spacetime!"


if __name__ == "__main__":
    #Some test code
    startTime = time.time()
    mySpace = Spacetime(
                 xMax=10,
                 numberOfNuclearDimenions=2,
                 numberOfElectronicDimensions=2,
                 numberOfSimulationSpacePointsPerNuclearDimension=100,
                 dt=.05)
    assert mySpace.functionSpacetimeZero(90).shape == (90, 100, 100), "Shape error in time!"
    assert mySpace.functionSpaceZero().shape == (100, 100), "Shape error!"
    f = lambda x, y: np.exp(-x**2/9 - y**2 / 8)*(1 + np.sin(x)**2 + np.cos(y)**2)
    test = mySpace.multivariableFunctionValueInX(f)
    mySpace.plotSpaceFunction(test)
    print "time elapsed in seconds: ", time.time() - startTime

    ##Test the fourier transform routines
    mySpace = Spacetime(
                 xMax=10,
                 numberOfNuclearDimenions=1,
                 numberOfElectronicDimensions=2,
                 numberOfSimulationSpacePointsPerNuclearDimension=100,
                 dt=.05)

    f = lambda x: np.exp(-x**2 / 5.0)
    FFTf = lambda x: np.sqrt(2.5) * np.exp(-5.0 *x**2 / 4.0)

    xVals = mySpace.xValues
    kVals = mySpace.kValues

    fVals = f(xVals)
    analyticFFTfVals = FFTf(kVals)
    numericFFTfVals = mySpace.spatialFourierTransform(fVals)
    numericInverseFFTfVals = mySpace.spatialInverseFourierTransform(numericFFTfVals)
    plt.figure()
    plt.plot(xVals, fVals, label="f")
    plt.plot(kVals, analyticFFTfVals, label="analytic")
    plt.plot(kVals, numericFFTfVals, label="numeric")
    plt.plot(xVals, numericInverseFFTfVals, label="back again")
    plt.legend()

    mySpace = Spacetime(
                 xMax=10,
                 numberOfNuclearDimenions=2,
                 numberOfElectronicDimensions=2,
                 numberOfSimulationSpacePointsPerNuclearDimension=100,
                 dt=.05)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(xVals, xVals)
    xSquared = mySpace.xValues**2
    yCubed = mySpace.xValues**3
    z = mySpace.returnAmplitudeFromListOfFunctionValues([xSquared, yCubed], additive=False)
    ax.plot_surface(X, Y, z)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    z = mySpace.multivariableFunctionValueInX(lambda x, y: x**2 * y**3)
    ax.plot_surface(X, Y, z)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    z = mySpace.multivariableFunctionValueInX(lambda y, x: x**2 * y**3)
    ax.plot_surface(X, Y, z)
