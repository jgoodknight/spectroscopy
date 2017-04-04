# -*- coding: utf-8 -*-
#Gone through for release 10 Feb 2017
import copy
import time

import matplotlib.pyplot as plt
import scipy.integrate
try:
    from matplotlib import animation
except:
    animation = object() #allows this to still run on clusters without the animation package installed
import numpy as np
import scipy

import Spacetime
import NuclearOperator
import NuclearWavefunction
import ElectronicWavefunction
import ElectronicOperator
import TimeNuclearWavefunction


class timeElectronicWavefunction(object):
    """A collection of electronic wavefunctions to represent the evolution of
    an electronic/nuclear state"""

    def __init__(self, SpaceToExistIn):

        self.mySpace = SpaceToExistIn
        self.dim = self.mySpace.electronicDimensionality

        self.listOfTimePositionAmplitudes = None

        self.timeSeries = []


        self.__currentTimeIndex = None #the most recently calculated index

    def allocateSpace(self, nSteps):
        "guess that you will use nSteps of time steps"
        self.listOfTimePositionAmplitudes = []
        for i in range(self.dim):
            self.listOfTimePositionAmplitudes.append(self.mySpace.functionSpacetimeZero(nSteps))

    def zero_copy(self):
        "Make an identical copy, except that it's zero"
        output = copy.deepcopy(self)
        for i in range(self.dim):
            old_amp = self.listOfTimePositionAmplitudes[i]
            old_amp_shape = old_amp.shape
            zero = np.zeros(old_amp_shape, np.complex)
            output.listOfTimePositionAmplitudes[i] = zero
        return output

    def allocateMoreSpace(self, nSteps):
        "guess that you will use nSteps more of time steps"
        #calculate new size
        oldSize = self.listOfTimePositionAmplitudes[0].shape[0]
        newSize = oldSize + nSteps
        #allocate that space
        newListOfTimePositionAmplitudes = []
        for i in range(self.dim):
            newListOfTimePositionAmplitudes.append(self.mySpace.functionSpacetimeZero(newSize))
        #preserve old data
        for elecIndex in range(self.dim):
            newListOfTimePositionAmplitudes[elecIndex][0:oldSize] = self.listOfTimePositionAmplitudes[elecIndex]

        self.listOfTimePositionAmplitudes = newListOfTimePositionAmplitudes


    def setInitialWavefunction(self, initialWF):
        "set the weavefunction at time zero to be the given wavefunction"
        self.initialWF = initialWF
        self.__currentTimeIndex = 0
        self[self.__currentTimeIndex] = initialWF
        self.timeSeries.append(0.0)

    def applyOperatorsAndAdvance(self, listOfOperators, overrideDT = None):
        "applies the given list of operators and returns the new Electronic Wavefunction"
        if overrideDT is not None:
            dt = overrideDT
        else:
            dt = self.mySpace.dt
        self.timeSeries.append(self.timeSeries[-1] + dt)

        self.__currentTimeIndex = self.__currentTimeIndex + 1

        #now to apply the operator
        newEWF = self[self.__currentTimeIndex -1]
        for operator in listOfOperators:
            newEWF = operator.APPLY( newEWF )

        self[self.__currentTimeIndex] = newEWF

        return newEWF


    def applyOperatorsNTimes(self, listOfOperators, N, overrideDT = None):
        "Applys a list of operators N times and stores the output, returns the last applied state"
        out = self.currentElectronicWavefunction()
        for i in range(N+2):
            out = self.applyOperatorsAndAdvance(listOfOperators, overrideDT)
        return out

    def applyOperatorsNTimesOnInitialWavefunction(self, listOfElectronicOperators, N, initialWF, overrideDT = None):
        "Returns last electronic wavefunction to be calculated"
        self.allocateSpace(N + 1)
        self.setInitialWavefunction(initialWF)

        #check that dt is the same
        if overrideDT is not None:
            dt = overrideDT
        else:
            dt = self.mySpace.dt

        #now to apply the operator
        for timeIndex in range(N): #not +1?
            newEWF = self.applyOperatorsAndAdvance(listOfElectronicOperators, overrideDT = dt)

        return newEWF


    def currentTime(self):
        "Most current time used in a calculation"
        return self.timeSeries[self.__currentTimeIndex]

    def currentElectronicWavefunction(self):
        "most currently calculated function"
        return self[self.__currentTimeIndex]

    def length(self):
        "number of calculated steps in time"
        return len(self.timeSeries)

    def shape(self):
        "primarily used for error checking to see the amount of data, returns a string"
        output = ""
        for i in range(self.dim):
            output = output + str(self.listOfTimePositionAmplitudes[i].shape)+ ", "
        return output

    def normSeries(self):
        "For error-checking: how does the norm of the wavefunction change as a function of time"
        norms = []
        for WF in self:
            norms.append(WF.norm())
        return np.array(norms)

    def overlapSeies_with_constant_EWF_after_operator_application(self, constant_EWF, operator):
        values = []
        for WF in self:
            values.append(WF.overlap(operator * constant_EWF))
        return np.array(values)

    def autocorrelationInFrequencySpace(self):
        "Autocorrelation as a function of frequency"
        t, ACF = self.autocorrelation()
        return self.mySpace.genericOneDimensionalFourierTransformFromZero(t, ACF)

    def timeExpectationValue(self, operator):
        "Takes an operator and brakets it with this wavefunction for all extant times"
        timeValues = []
        for ii in range(self.listOfTimePositionAmplitudes[0].shape[0]):
            ewf = self[ii]
            operatorActedOnEWF = operator * ewf
            newValue = ewf.overlap(operatorActedOnEWF)
            timeValues.append(newValue)
        return np.array(timeValues)

    def timeExpectationValueOfPolarizationAndOverlapWithElectricField(self, dipoleTuple, electricFieldTuple):
        "Takes a tuple of dipole operators and a tuple of electric field operators to calculate.  I don't actually think this should be used anywhere..."
        output = 0.0
        output = self.timeExpectationValue(dipoleTuple[0]) * electricFieldTuple[0].valueAtTime(self.timeSeries)
        output = output + self.timeExpectationValue(dipoleTuple[1]) * electricFieldTuple[1].valueAtTime(self.timeSeries)
        output = output + self.timeExpectationValue(dipoleTuple[2]) * electricFieldTuple[2].valueAtTime(self.timeSeries)
        return scipy.integrate.simps(output, dx = self.mySpace.dt)



    def timeOverlapWithOtherBraEWFOfPolarizationAndOverlapWithElectricField(self, braEWF, dipoleTuple, electricFieldTuple):
        """This is the workhorse function which treats self as the ket, and calculates
        the overlap with the supplied wavefunction after applying the dipole operator
        then takes the dot product with the supplied electric field and integrates"""
        output = []
        for i, EWF in enumerate(self):
            xVal = EWF.overlap(dipoleTuple[0] * braEWF[i]) * electricFieldTuple[0].valueAtTime(self.timeSeries[i])
            yVal = EWF.overlap(dipoleTuple[1] * braEWF[i]) * electricFieldTuple[1].valueAtTime(self.timeSeries[i])
            zVal = EWF.overlap(dipoleTuple[2] * braEWF[i]) * electricFieldTuple[2].valueAtTime(self.timeSeries[i])
            output.append(xVal + yVal + zVal)
        return scipy.integrate.simps(output, dx = self.mySpace.dt)



    def timeOverlapWithOtherBraEWFOfPolarization(self, braEWF, dipoleTuple):
        """Function which treats self as the ket, and calculates the expectation value
        of the dipole operator then outputs a time vector"""
        xOutput = []
        yOutput = []
        zOutput = []
        for i, EWF in enumerate(self):
            xVal = EWF.overlap(dipoleTuple[0] * braEWF[i])
            yVal = EWF.overlap(dipoleTuple[1] * braEWF[i])
            zVal = EWF.overlap(dipoleTuple[2] * braEWF[i])
            xOutput.append(xVal)
            yOutput.append(yVal)
            zOutput.append(zVal)
        return (np.array(xOutput), np.array(yOutput), np.array(zOutput))


    def grabTimeNuclearWavefunction(self, index):
        "Will give you a nuclear time wavefunction for the given electronic index"
        output = TimeNuclearWavefunction.timeNuclearWavefunction(self.mySpace)
        output.timePositionAmplitude = self.listOfTimePositionAmplitudes[index]
        return output


    ##DEFINE ITERATION METHODS OVER TIME
    def __iter__(self):
        self.counter = 0
        return self

    def next(self):
        try:
            self.counter = self.counter + 1
            return self[self.counter - 1]
        except IndexError:
            raise StopIteration

    def __getitem__(self,index):
        "outputs the spatial electronic wavefunction at time index index"
        index = int(index)
        nucWFlist = []
        for ii in range(self.dim):
            newNucWF = NuclearWavefunction.nuclearWavefunction(self.mySpace)
            newNucWF.xAmplitude = self.listOfTimePositionAmplitudes[ii][index]
            nucWFlist.append(newNucWF)
        out = ElectronicWavefunction.electronicWavefunction(self.mySpace, nucWFlist)
        return out

    def __setitem__(self, index, ewf):
        "sets the spatial electronic wavefunction at time index, index, to be ewf"

        if index < 0:
            index = index + self.__currentTimeIndex

        for elecIndex in range(self.dim):
            self.listOfTimePositionAmplitudes[elecIndex][index] = ewf[elecIndex].xAmplitude
        self.__currentTimeIndex = index


    def animate1D(self, fileName, numberOfFrames=None):
        "Animate a 1D nuclear wavefunction as it evolves in time"
        d = self.mySpace.nuclearDimensionality
        if d != 1:
            raise NuclearWavefunction.unplotableNuclearWavefunction()

        listOfPlottingAmplitudes = map(np.abs, self.listOfTimePositionAmplitudes)
        yMin = min(map(np.min, listOfPlottingAmplitudes))
        yMax = max(map(np.max, listOfPlottingAmplitudes))

        xVals = self.mySpace.xValues

        fig = plt.figure()
        for ii in range(self.dim):
            im = plt.plot(xVals, listOfPlottingAmplitudes[ii][0], label = str(ii))
        plt.ylim((yMin, yMax))
        plt.legend()
        ax = fig.gca()

        def animate(i, data,  ax, fig):
            ax.cla()
            for ii in range(self.dim):
                try:
                    im = plt.plot(xVals, listOfPlottingAmplitudes[ii][i],  label = str(ii))
                except:
                    return None
            plt.ylim((yMin, yMax))
            ax = fig.gca()
            plt.legend()
            plt.title(str(i))
            return im,

        anim = animation.FuncAnimation(fig, animate,
                               frames = numberOfFrames, interval=20, blit=True, fargs=(listOfPlottingAmplitudes, ax, fig))
        anim.save(fileName, fps=20)



    def animate2D(self, electronicIndex, fileName, numberOfFrames):
        "Animate a 2D nuclear wavefunction as it evolves in time"
        d = self.mySpace.nuclearDimensionality
        if d != 2:
            raise NuclearWavefunction.unplotableNuclearWavefunction()

        plottingAmplitude = np.abs(self.listOfTimePositionAmplitudes[electronicIndex])
        zMin = np.min(plottingAmplitude)
        zMax = np.max(plottingAmplitude)

        contourLevels = 100

        contourSpacings = np.linspace(zMin, zMax, contourLevels)

        fig = plt.figure()
        im = plt.contourf(plottingAmplitude[0], contourSpacings)
        ax = fig.gca()

        def animate(i, data,  ax, fig):
            ax.cla()
            im = ax.contourf(data[i], contourSpacings)
            plt.title(str(i))
            return im,

        anim = animation.FuncAnimation(fig, animate,
                               frames = numberOfFrames, interval=20, blit=True, fargs=(plottingAmplitude, ax, fig))
        anim.save(fileName, fps=20)



    #override arithmetic
    def __mul__(self, other):
        output = copy.copy(self)
        for ii in range(self.dim):
            output.listOfTimePositionAmplitudes[ii] = output.listOfTimePositionAmplitudes[ii] * other
        return output

    def __neg__(self):
        output = copy.copy(self)
        for ii in range(self.dim):
            output.listOfTimePositionAmplitudes[ii] = -output.listOfTimePositionAmplitudes[ii]
        return output

    def __add__(self, other):
        output = copy.copy(self)
        for ii in range(self.dim):
            output.listOfTimePositionAmplitudes[ii] = output.listOfTimePositionAmplitudes[ii] + other.listOfTimePositionAmplitudes[ii]
        return output


if __name__ == "__main__":
    #Some test code
    mySpace = Spacetime.Spacetime(xMax = 10,
                 numberOfNuclearDimenions = 2,
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

    testNuclearHamiltonian1 = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [testHarmonicOscillator1, testHarmonicOscillator2 ] )
    testNuclearHamiltonian2 = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [testHarmonicOscillator2, testHarmonicOscillator3 ] )
    testNuclearHamiltonian3 = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [testHarmonicOscillator3, testHarmonicOscillator4 ] )
    testNuclearHamiltonian4 = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [testHarmonicOscillator4, testHarmonicOscillator1 ] )


    testNuclearWavefunction1 = NuclearWavefunction.nuclearWavefunction(mySpace, groundStateNuclearHamiltonian = testNuclearHamiltonian1 )
    testNuclearWavefunction2 = NuclearWavefunction.nuclearWavefunction(mySpace, groundStateNuclearHamiltonian = testNuclearHamiltonian2 )
    testNuclearWavefunction3 = NuclearWavefunction.nuclearWavefunction(mySpace, groundStateNuclearHamiltonian = testNuclearHamiltonian3 )
    testNuclearWavefunction4 = NuclearWavefunction.nuclearWavefunction(mySpace, groundStateNuclearHamiltonian = testNuclearHamiltonian4 )

    electronicCoupling = NuclearOperator.constantPositionNuclearOperator(mySpace, .5)
    assumptionsForExponent = {'diagonalKinetic': True,
                       'diagonalPotential' : False,
                       'OneNonDiagonal2by2' : False,
                       '2by2Indeces' : (1, 2)}
    testElectronicHamiltonian = ElectronicOperator.ElectronicHamiltonian(mySpace, [(0,0, testNuclearHamiltonian2),
                                                                (1,1, testNuclearHamiltonian3),
                                                                (2,2, testNuclearHamiltonian3),
                                                                (2,1, electronicCoupling),
                                                                (1,2, electronicCoupling),
                                                                (3,3, testNuclearHamiltonian1)])
    testEWF = ElectronicWavefunction.electronicWavefunction(mySpace,
                           listOfNuclearWavefunctions = [testNuclearWavefunction1, testNuclearWavefunction2, testNuclearWavefunction3, testNuclearWavefunction4],
                           Normalize=True)
    testElectronicPropagator = testElectronicHamiltonian.myPropagator(assumptionsDICT=assumptionsForExponent)


    testTimeEWF = timeElectronicWavefunction(mySpace)
    print "starting propagation"
    startTime = time.time()
    testTimeEWF.applyOperatorNTimesOnInitialWavefunction(testElectronicPropagator, 200, testEWF)
    print "elapsed time, ", time.time() - startTime
    t, s = testTimeEWF.autocorrelation()
    w, sw = testTimeEWF.autocorrelationInFrequencySpace()

    plt.figure()
    plt.plot(t, np.abs(s))
    plt.figure()
    plt.plot(w, np.abs(sw))

    print t.shape

#
#    testTimeEWF = timeElectronicWavefunction(mySpace)
#    testTimeEWF.setExpectedNumberOfSteps(200)
#    testTimeEWF.setInitialWavefunction(testEWF)
#    print "starting propagation"
#    startTime = time.time()
#    for i in range(200):
#        testTimeEWF.applyAndExtendOnce([testElectronicPropagator])
#    print "elapsed time, ", time.time() - startTime
#    t, s = testTimeEWF.autocorrelation()
#    w, sw = testTimeEWF.autocorrelationInFrequencySpace()
#
#    plt.figure()
#    plt.plot(t, np.abs(s))
#    plt.figure()
#    plt.plot(w, np.abs(sw))
#    print t.shape
    #testTimeEWF.animate2D(1, 'testCoupledElectronicPropagation2D_2.mp4', 200)
