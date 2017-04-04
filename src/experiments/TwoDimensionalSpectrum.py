# -*- coding: utf-8 -*-
"""
Created on Fri Jan 04 13:23:27 2013
UNTESTED AND DEVELOPED!
@author: Joey
"""
import copy
import time
import warnings
warnings.warn("This pacakge of 2D spectroscopy is not testing or implemented!")

import numpy as np
import scipy.integrate

import scipy.fftpack
import multiprocessing
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
try:
    from matplotlib import animation
except:
    animation = object()

#import spectroscopy.Spacetime as Spacetime
import spectroscopy.TimeElectronicWavefunction as TimeElectronicWavefunction
#import spectroscopy.ElectronicOperator as ElectronicOperator
#import spectroscopy.TimeFunction as TimeFunction
#import spectroscopy.NuclearWavefunction as NuclearWavefunction
#import spectroscopy.NuclearOperator as NuclearOperator
#import spectroscopy.ElectronicWavefunction as ElectronicWavefunction
import spectroscopy.TimePerturbation as TimePerturbation

import experimentBase

class PopulationTimeScan(experimentBase.experiment):
    "2D electronic Spectrum for multiple population evolution times"
    def __init__(self, space,
                 electronicHamiltonian,
                 MuxMuyMuzElectronicOperatorTuple,
                 initialElectronicWavefunction,
                 pumpBeamTuple,
                 maximumPopulationTime,
                 numberOfPopulationTimePoints,
                 maximumPumpTime,
                 numberOfPumpTimePoints,
                 maximumProbeTime,
                 numberOfProbeTimePoints,
                 opticalGap = 0.0,
                 IDstring = "",
                 numberOfProcessors = None):
        #store all the variables
        self.mySpace = space

        self.MuX, self.MuY, self.MuZ = MuxMuyMuzElectronicOperatorTuple
        self.pumpEx, self.pumpEy, self.pumpEz = pumpBeamTuple

        self.intitialEWF = initialElectronicWavefunction
        self.myElectronicHamiltonian = electronicHamiltonian
        self.myFreePropagator = electronicHamiltonian.myPropagator()

        #We are given a target maximum waiting and a specific number of time points to take.
        #We then have to scale that to the closest multiple of mySpace.dt to make the propagation work properly
        self.minPopulationTime = 0.0
        self.numberOfPopulationTimePoints = numberOfPopulationTimePoints
        targetMaxPopulationTime = maximumPumpTime
        targetPopulationDT = (targetMaxPopulationTime - self.minPopulationTime) / numberOfPopulationTimePoints
        self.timeStepsPerPopulationStep = int(np.round(targetPopulationDT / self.mySpace.dt, 0))
        #What if it's zero!?  We can't have that
        if self.timeStepsPerPopulationStep == 0:
            self.timeStepsPerPopulationStep = 1.
        self.populationDT = self.timeStepsPerPopulationStep * self.mySpace.dt
        self.maxPopulationTime = self.minPopulationTime + self.populationDT * (self.numberOfPopulationTimePoints - 1)
        self.populationTimes = np.arange(self.minPopulationTime, self.maxPopulationTime, self.populationDT)


        self.minPumpTime = 0.0
        self.pumpTimePoints = numberOfPumpTimePoints
        targetMaxPumpTime = maximumPumpTime
        targetPumpDT = (targetMaxPumpTime - self.minPumpTime) / numberOfPumpTimePoints
        self.timeStepsPerPumpStep = int(np.round(targetPumpDT / self.mySpace.dt, 0))
        #What if it's zero!?  WE can't have that
        if self.timeStepsPerPumpStep == 0:
            self.timeStepsPerPumpStep = 1.
        self.pumpDT = self.timeStepsPerPumpStep * self.mySpace.dt
        self.maxPumpTime = self.minPumpTime + self.pumpDT * (self.pumpTimePoints - 1)
        self.pumpTimes = np.arange(self.minPumpTime, self.maxPumpTime, self.pumpDT)


        self.minProbeTime = 0.0
        self.ProbeTimePoints = numberOfProbeTimePoints
        targetMaxProbeTime = maximumProbeTime
        targetProbeDT = (targetMaxProbeTime - self.minProbeTime) / numberOfProbeTimePoints
        self.timeStepsPerProbeStep = int(np.round(targetProbeDT / self.mySpace.dt, 0))
        #What if it's zero!?  WE can't have that
        if self.timeStepsPerProbeStep == 0:
            self.timeStepsPerProbeStep = 1.
        self.probeDT = self.timeStepsPerProbeStep * self.mySpace.dt
        self.maxProbeTime = self.minProbeTime + self.probeDT * (self.ProbeTimePoints - 1)
        self.probeTimes = np.arange(self.minProbeTime, self.maxProbeTime, self.probeDT)

        self.firstProbeEx = copy.deepcopy(self.pumpEx)
        self.firstProbeEy = copy.deepcopy(self.pumpEy)
        self.firstProbeEz = copy.deepcopy(self.pumpEz)

        self.probePower = self.firstProbeEx.totalPower() + self.firstProbeEy.totalPower() + self.firstProbeEz.totalPower()

        self.calculated = False
        self.IDstring = IDstring

        self.opticalGap = opticalGap

        #maximum time the experiment will go on in any case
        #the padding for all four pulses
        self.maximumTime = 4.0 * 3.0 * 2.0 * max(map(lambda x: x.timePillow, pumpBeamTuple))
        #plus the maxmimum probe and pump spacing
        self.maximumTime = self.maximumTime + self.maxProbeTime + self.maxPumpTime
        #plus the population time
        self.maximumTime = self.maximumTime + self.maxPopulationTime
        self.maximumTimeSteps = int(np.round(self.maximumTime / self.mySpace.dt))

        if numberOfProcessors is None:
            self.numberOfProcessors = multiprocessing.cpu_count() - 2
        else:
            self.numberOfProcessors = numberOfProcessors



    def calculate(self):
        overallStartTime = time.time()
        if self.calculated:
            print "You've already done the calculation, look at the results instead!"
            return
        setupStartTime = time.time()
        self.ppTimeSeries = []

        self.ppSignals = []

        myPool = multiprocessing.Pool(self.numberOfProcessors)

        zeroOrderTimeWavefunction = TimeElectronicWavefunction.timeElectronicWavefunction(self.mySpace)
        zeroOrderTimeWavefunction.applyOperatorsNTimesOnInitialWavefunction([self.myFreePropagator],
                                                                            N = self.maximumTimeSteps,
                                                                            initialWF = self.intitialEWF)

        pulse1xPLUS = copy.deepcopy(self.firstProbeEx.plusFunction())
        pulse1xMINUS = copy.deepcopy(self.firstProbeEx.minusFunction())

        pulse1yPLUS = copy.deepcopy(self.firstProbeEy.plusFunction())
        pulse1yMINUS = copy.deepcopy(self.firstProbeEy.minusFunction())

        pulse1zPLUS = copy.deepcopy(self.firstProbeEz.plusFunction())
        pulse1zMINUS = copy.deepcopy(self.firstProbeEz.minusFunction())

        pulse1PLUStuple = (pulse1xPLUS, pulse1yPLUS, pulse1zPLUS)
        pulse1MINUStuple = (pulse1xMINUS, pulse1yMINUS, pulse1zMINUS)

        k1PlusInteractor = TimePerturbation.electricDipoleOneInteraction(space = self.mySpace,
                                                                       electronicHamiltonian = self.myElectronicHamiltonian,
                                                                       MuxMuyMuzElectronicOperatorTuple = (self.MuX, self.MuY, self.MuZ),
                                                                       ExEyEzTimeFunctionTuple = pulse1PLUStuple,
                                                                       maxTime = self.maximumTime)

        k1MinusInteractor = TimePerturbation.electricDipoleOneInteraction(space = self.mySpace,
                                                                       electronicHamiltonian = self.myElectronicHamiltonian,
                                                                       MuxMuyMuzElectronicOperatorTuple = (self.MuX, self.MuY, self.MuZ),
                                                                       ExEyEzTimeFunctionTuple = pulse1MINUStuple,
                                                                       maxTime = self.maximumTime)

        k1PlusWavefunction = k1PlusInteractor.goFromTimeWavefunction(zeroOrderTimeWavefunction)
        k1MinusWavefunction = k1MinusInteractor.goFromTimeWavefunction(zeroOrderTimeWavefunction)

#        print zeroOrderTimeWavefunction.shape(), "\n", k1MinusWavefunction.shape(), "\n", k1MinusWavefunction.shape(), "\n"
        k1MINUS_k2PLUSpumpedWavefunctions = []
        k1PLUS_k2MINUSpumpedWavefunctions = []

        pulse2Tuples = []

        for pumpTime in self.pumpTimes:
            pulse2tuple = (self.firstProbeEx.pulseCopyJumpedForward(pumpTime), self.firstProbeEy.pulseCopyJumpedForward(pumpTime), self.firstProbeEz.pulseCopyJumpedForward(pumpTime))
            pulse2Tuples.append(pulse2tuple)

            pulse2xPLUS = pulse2tuple[0].plusFunction()
            pulse2xMINUS = pulse2tuple[0].minusFunction()

            pulse2yPLUS = pulse2tuple[1].plusFunction()
            pulse2yMINUS = pulse2tuple[1].minusFunction()

            pulse2zPLUS = pulse2tuple[2].plusFunction()
            pulse2zMINUS = pulse2tuple[2].minusFunction()

            pulse2PLUStuple = (pulse2xPLUS, pulse2yPLUS, pulse2zPLUS)
            pulse2MINUStuple = (pulse2xMINUS, pulse2yMINUS, pulse2zMINUS)

            k2PlusInteractor = TimePerturbation.electricDipoleOneInteraction(space = self.mySpace, electronicHamiltonian = self.myElectronicHamiltonian, MuxMuyMuzElectronicOperatorTuple = (self.MuX, self.MuY, self.MuZ), ExEyEzTimeFunctionTuple = pulse2PLUStuple, maxTime = self.maximumTime)

            k2MinusInteractor = TimePerturbation.electricDipoleOneInteraction(space = self.mySpace, electronicHamiltonian = self.myElectronicHamiltonian, MuxMuyMuzElectronicOperatorTuple = (self.MuX, self.MuY, self.MuZ), ExEyEzTimeFunctionTuple = pulse2MINUStuple, maxTime = self.maximumTime)


            k1MINUS_k2PLUSpumpedWavefunctions.append(interactionHelper(k2PlusInteractor, k1MinusWavefunction))
            k1PLUS_k2MINUSpumpedWavefunctions.append(interactionHelper(k2MinusInteractor, k1PlusWavefunction))



        k1MINUS_k2PLUSpumpedWavefunctions = myPool.map(BIG_RED_BUTTON, k1MINUS_k2PLUSpumpedWavefunctions)
        k1PLUS_k2MINUSpumpedWavefunctions = myPool.map(BIG_RED_BUTTON, k1PLUS_k2MINUSpumpedWavefunctions)

#        print zeroOrderTimeWavefunction.length(), k1MINUS_k2PLUSpumpedWavefunctions[-1].length(), k1PLUS_k2MINUSpumpedWavefunctions[-1].length()
#        print zeroOrderTimeWavefunction.length(), k1MINUS_k2PLUSpumpedWavefunctions[0].length(), k1PLUS_k2MINUSpumpedWavefunctions[0].length()

        self.resphasingTimeSignals = np.zeros((len(self.populationTimes), len(self.pumpTimes), len(self.probeTimes)), dtype = np.complex)
        self.nonresphasingTimeSignals = np.zeros((len(self.populationTimes), len(self.pumpTimes), len(self.probeTimes)), dtype = np.complex)


        self.rawResphasingFrequencySignals = np.zeros((len(self.populationTimes), len(self.pumpTimes), len(self.probeTimes)), dtype = np.complex)
        self.rawNonresphasingFrequencySignals = np.zeros((len(self.populationTimes), len(self.pumpTimes), len(self.probeTimes)), dtype = np.complex)

        self.filteredResphasingFrequencySignals = np.zeros((len(self.populationTimes), len(self.pumpTimes), len(self.probeTimes)), dtype = np.complex)
        self.filteredNonresphasingFrequencySignals = np.zeros((len(self.populationTimes), len(self.pumpTimes), len(self.probeTimes)), dtype = np.complex)

        #create Filter
        xDist = self.pumpTimes[-1] - self.pumpTimes[0]
        yDist = self.probeTimes[-1] - self.probeTimes[0]
        stdX = xDist / 6.0
        stdY = yDist / 6.0
        X, Y = np.meshgrid(self.pumpTimes, self.probeTimes)
        filterFunction =  np.exp(-(X - xDist/2.0)**2.0 / (2.0 * stdX**2.0)) * np.exp(-(Y - yDist/2.0)**2.0 / (2.0 * stdY**2.0)) / (2.0 * np.pi * stdX * stdY)


        for Tindex, Tvalue in enumerate(self.populationTimes):

            pulse3tuples = []
            k1MINUS_k2PLUS_k3PLUSpumpedWavefunctions = []
            k1PLUS_k2MINUS_k3PLUSpumpedWavefunctions = []

            for i, pulse2Tuple in enumerate(pulse2Tuples):
                pulse3tuple = (pulse2Tuple[0].pulseCopyJumpedForward(Tvalue), pulse2Tuple[1].pulseCopyJumpedForward(Tvalue), pulse2Tuple[2].pulseCopyJumpedForward(Tvalue))
                pulse3tuples.append(pulse3tuple)

                pulse3PLUStuple = (pulse3tuple[0].plusFunction(), pulse3tuple[1].plusFunction(), pulse3tuple[2].plusFunction())

                k3PlusInteractor = TimePerturbation.electricDipoleOneInteraction(space = self.mySpace, electronicHamiltonian = self.myElectronicHamiltonian, MuxMuyMuzElectronicOperatorTuple = (self.MuX, self.MuY, self.MuZ), ExEyEzTimeFunctionTuple = pulse3PLUStuple, maxTime = self.maximumTime)

                k1MINUS_k2PLUS_k3PLUSpumpedWavefunctions.append(interactionHelper(k3PlusInteractor, k1MINUS_k2PLUSpumpedWavefunctions[i]))
                k1PLUS_k2MINUS_k3PLUSpumpedWavefunctions.append(interactionHelper(k3PlusInteractor, k1PLUS_k2MINUSpumpedWavefunctions[i]))

            k1MINUS_k2PLUS_k3PLUSpumpedWavefunctions = myPool.map(BIG_RED_BUTTON, k1MINUS_k2PLUS_k3PLUSpumpedWavefunctions)
            k1PLUS_k2MINUS_k3PLUSpumpedWavefunctions = myPool.map(BIG_RED_BUTTON, k1PLUS_k2MINUS_k3PLUSpumpedWavefunctions)


#            print zeroOrderTimeWavefunction.length(), k1MINUS_k2PLUS_k3PLUSpumpedWavefunctions[-1].length(), k1PLUS_k2MINUS_k3PLUSpumpedWavefunctions[-1].length()
#            print zeroOrderTimeWavefunction.length(), k1MINUS_k2PLUS_k3PLUSpumpedWavefunctions[0].length(), k1PLUS_k2MINUS_k3PLUSpumpedWavefunctions[0].length()

            for i, pulse3Tuple in enumerate(pulse3tuples):
                for j, probeTime in enumerate(self.probeTimes):
                    pulse4MINUS = (pulse3Tuple[0].pulseCopyJumpedForward(probeTime).minusFunction(), pulse3Tuple[1].pulseCopyJumpedForward(probeTime).minusFunction(), pulse3Tuple[2].pulseCopyJumpedForward(probeTime).minusFunction())
#                    print i, j, "\n", zeroOrderTimeWavefunction.shape(), "\n", k1MINUS_k2PLUS_k3PLUSpumpedWavefunctions[i].shape(), "\n", k1PLUS_k2MINUS_k3PLUSpumpedWavefunctions[i].shape()
                    rephasingSignal = (zeroOrderTimeWavefunction + k1MINUS_k2PLUS_k3PLUSpumpedWavefunctions[i]).timeExpectationValueOfPolarizationAndOverlapWithElectricField((self.MuX, self.MuY, self.MuZ), pulse4MINUS)
                    nonrephasingSignal = (zeroOrderTimeWavefunction + k1PLUS_k2MINUS_k3PLUSpumpedWavefunctions[i]).timeExpectationValueOfPolarizationAndOverlapWithElectricField((self.MuX, self.MuY, self.MuZ), pulse4MINUS)
                    self.resphasingTimeSignals[Tindex, i, j] = rephasingSignal
                    self.nonresphasingTimeSignals[Tindex, i, j] = nonrephasingSignal

            self.calculationTime = time.time() - overallStartTime

            self.rawResphasingFrequencySignals[Tindex] = scipy.fftpack.fftn(self.resphasingTimeSignals[Tindex] )
            self.rawNonresphasingFrequencySignals[Tindex]  = scipy.fftpack.fftn(self.nonresphasingTimeSignals[Tindex] )

            self.filteredResphasingFrequencySignals[Tindex] = scipy.fftpack.fftn(self.resphasingTimeSignals[Tindex] * filterFunction )
            self.filteredNonresphasingFrequencySignals[Tindex] = scipy.fftpack.fftn(self.nonresphasingTimeSignals[Tindex] * filterFunction )


        self.pumpFrequencies = 2 * np.pi * scipy.fftpack.fftshift(scipy.fftpack.fftfreq(len(self.pumpTimes), d = self.pumpTimes[2] - self.pumpTimes[1]))
        self.probeFrequencies = 2 * np.pi * scipy.fftpack.fftshift(scipy.fftpack.fftfreq(len(self.probeTimes), d = self.probeTimes[2] - self.probeTimes[1]))

        self.pumpFrequenciesWavenumbers = self.mySpace.unitHandler.wavenumbersFromEnergyUnits(self.pumpFrequencies)
        self.probeFrequenciesWavenumbers = self.mySpace.unitHandler.wavenumbersFromEnergyUnits(self.probeFrequencies)

        return self


    def plotTime(self, Tindex = 0, newFig = True):
        if newFig:
            plt.figure()
        pumpTime = self.pumpTimes
        pumpTime = self.mySpace.unitHandler.femtosecondsFromTime(pumpTime)

        probeTime = self.probeTimes
        probeTime = self.mySpace.unitHandler.femtosecondsFromTime(probeTime)

        plt.contourf(pumpTime, probeTime, np.abs(self.nonresphasingTimeSignals[Tindex] + self.resphasingTimeSignals[Tindex]))
        plt.title("2D Spectrum (Untransformed)")
        plt.xlabel(r"$T_{pump}$")
        plt.ylabel(r"$T_{probe}$")


    def plotFrequency(self, Tindex = 0, newFig = True):
        if newFig:
            plt.figure()
        pumpFrequencies = self.pumpFrequencies
        pumpFrequencies = self.mySpace.unitHandler.wavenumbersFromEnergyUnits(pumpFrequencies)

        probeFrequencies = self.probeFrequencies
        probeFrequencies = self.mySpace.unitHandler.wavenumbersFromEnergyUnits(probeFrequencies)

        plt.contourf(self.pumpFrequenciesWavenumbers, self.probeFrequenciesWavenumbers, np.abs(self.filteredNonresphasingFrequencySignals[Tindex] + self.filteredResphasingFrequencySignals[Tindex]))
        plt.title("2D Spectrum")
        plt.xlabel(r"$\omega_{pump}$ (wavenumbers)")
        plt.ylabel(r"$\omega_{probe}$ (wavenumbers)")

    def animateFrequency(self, filename):
        "Animate a 2D nuclear wavefunction as it evolves in time"
        plottingAmplitude = np.abs(self.filteredNonresphasingFrequencySignals + self.filteredResphasingFrequencySignals)
        zMin = np.min(plottingAmplitude)
        zMax = np.max(plottingAmplitude)

        contourLevels = 100

        contourSpacings = np.linspace(zMin, zMax, contourLevels)

        yVals = self.probeFrequencies
        xVals = self.pumpFrequencies

        fig = plt.figure()
        im = plt.contourf(xVals, yVals, plottingAmplitude[0], contourSpacings)
        ax = fig.gca()

        def animate(i, data,  ax, fig):
            ax.cla()
            im = ax.contourf(xVals, yVals, data[i], contourSpacings)
            plt.title(str(i))
            return im,

        anim = animation.FuncAnimation(fig, animate, frames = self.rawNonresphasingFrequencySignals.shape[0], interval=20, blit=True, fargs=(plottingAmplitude, ax, fig) )
        anim.save(filename, fps=20)





class OnePopulationTime(experimentBase.experiment):
    "2D electronic Spectrum for just one population evolution time"
    def __init__(self, space,
                 electronicHamiltonian,
                 MuxMuyMuzElectronicOperatorTuple,
                 initialElectronicWavefunction,
                 pumpBeamTuple,
                 populationTime,
                 maximumPumpTime,
                 numberOfPumpTimePoints,
                 maximumProbeTime,
                 numberOfProbeTimePoints,
                 opticalGap = 0.0,
                 IDstring = "",
                 numberOfProcessors = None):
        #store all the variables
        self.mySpace = space

        self.MuX, self.MuY, self.MuZ = MuxMuyMuzElectronicOperatorTuple
        self.pumpEx, self.pumpEy, self.pumpEz = pumpBeamTuple

        self.intitialEWF = initialElectronicWavefunction
        self.myElectronicHamiltonian = electronicHamiltonian
        self.myFreePropagator = electronicHamiltonian.myPropagator()

        #We are given a target maximum waiting and a specific number of time points to take.
        #We then have to scale that to the closest multiple of mySpace.dt to make the propagation work properly
        self.T = populationTime

        self.minPumpTime = 0.0
        self.pumpTimePoints = numberOfPumpTimePoints
        targetMaxPumpTime = maximumPumpTime
        targetPumpDT = (targetMaxPumpTime - self.minPumpTime) / numberOfPumpTimePoints
        self.timeStepsPerPumpStep = int(np.round(targetPumpDT / self.mySpace.dt, 0))
        #What if it's zero!?  WE can't have that
        if self.timeStepsPerPumpStep == 0:
            self.timeStepsPerPumpStep = 1.
        self.pumpDT = self.timeStepsPerPumpStep * self.mySpace.dt
        self.maxPumpTime = self.minPumpTime + self.pumpDT * (self.pumpTimePoints - 1)
        self.pumpTimes = np.arange(self.minPumpTime, self.maxPumpTime, self.pumpDT)


        self.minProbeTime = 0.0
        self.ProbeTimePoints = numberOfProbeTimePoints
        targetMaxProbeTime = maximumProbeTime
        targetProbeDT = (targetMaxProbeTime - self.minProbeTime) / numberOfProbeTimePoints
        self.timeStepsPerProbeStep = int(np.round(targetProbeDT / self.mySpace.dt, 0))
        #What if it's zero!?  WE can't have that
        if self.timeStepsPerProbeStep == 0:
            self.timeStepsPerProbeStep = 1.
        self.probeDT = self.timeStepsPerProbeStep * self.mySpace.dt
        self.maxProbeTime = self.minProbeTime + self.probeDT * (self.ProbeTimePoints - 1)
        self.probeTimes = np.arange(self.minProbeTime, self.maxProbeTime, self.probeDT)

        self.firstProbeEx = copy.deepcopy(self.pumpEx)
        self.firstProbeEy = copy.deepcopy(self.pumpEy)
        self.firstProbeEz = copy.deepcopy(self.pumpEz)

        self.probePower = self.firstProbeEx.totalPower() + self.firstProbeEy.totalPower() + self.firstProbeEz.totalPower()

        self.calculated = False
        self.IDstring = IDstring

        self.opticalGap = opticalGap

        #maximum time the experiment will go on in any case
        #the padding for all four pulses
        self.maximumTime = 4.0 * 3.0 * 2.0 * max(map(lambda x: x.timePillow, pumpBeamTuple))
        #plus the maxmimum probe and pump spacing
        self.maximumTime = self.maximumTime + self.maxProbeTime + self.maxPumpTime
        #plus the population time
        self.maximumTime = self.maximumTime + self.T
        self.maximumTimeSteps = int(np.round(self.maximumTime / self.mySpace.dt))

        if numberOfProcessors is None:
            self.numberOfProcessors = multiprocessing.cpu_count() - 2
        else:
            self.numberOfProcessors = numberOfProcessors



    def calculate(self):
        overallStartTime = time.time()
        if self.calculated:
            print "You've already done the calculation, look at the results instead!"
            return
        setupStartTime = time.time()

        self.ppTimeSeries = []

        self.ppSignals = []

        myPool = multiprocessing.Pool(self.numberOfProcessors)

        self.zeroOrderTimeWavefunction = TimeElectronicWavefunction.timeElectronicWavefunction(self.mySpace)
        self.zeroOrderTimeWavefunction.applyOperatorsNTimesOnInitialWavefunction([self.myFreePropagator],
                                                                            N = self.maximumTimeSteps,
                                                                            initialWF = self.intitialEWF)

        pulse1xPLUS = copy.deepcopy(self.firstProbeEx.plusFunction())
        pulse1xMINUS = copy.deepcopy(self.firstProbeEx.minusFunction())

        pulse1yPLUS = copy.deepcopy(self.firstProbeEy.plusFunction())
        pulse1yMINUS = copy.deepcopy(self.firstProbeEy.minusFunction())

        pulse1zPLUS = copy.deepcopy(self.firstProbeEz.plusFunction())
        pulse1zMINUS = copy.deepcopy(self.firstProbeEz.minusFunction())

        pulse1PLUStuple = (pulse1xPLUS, pulse1yPLUS, pulse1zPLUS)
        pulse1MINUStuple = (pulse1xMINUS, pulse1yMINUS, pulse1zMINUS)

        k1PlusInteractor = TimePerturbation.electricDipoleOneInteraction(space = self.mySpace,
                                                                       electronicHamiltonian = self.myElectronicHamiltonian,
                                                                       MuxMuyMuzElectronicOperatorTuple = (self.MuX, self.MuY, self.MuZ),
                                                                       ExEyEzTimeFunctionTuple = pulse1PLUStuple,
                                                                       maxTime = self.maximumTime)

        k1MinusInteractor = TimePerturbation.electricDipoleOneInteraction(space = self.mySpace,
                                                                       electronicHamiltonian = self.myElectronicHamiltonian,
                                                                       MuxMuyMuzElectronicOperatorTuple = (self.MuX, self.MuY, self.MuZ),
                                                                       ExEyEzTimeFunctionTuple = pulse1MINUStuple,
                                                                       maxTime = self.maximumTime)

        self.k1PlusWavefunction = k1PlusInteractor.goFromTimeWavefunction(self.zeroOrderTimeWavefunction)
        self.k1MinusWavefunction = k1MinusInteractor.goFromTimeWavefunction(self.zeroOrderTimeWavefunction)

#        print "0 %s \nK1+ %s \nK1- %s" % (str(self.zeroOrderTimeWavefunction.length()), str(self.k1PlusWavefunction.length()), str(self.k1MinusWavefunction.length()) )
        self.k1MINUS_k2PLUSpumpedWavefunctions = []
        self.k1PLUS_k2MINUSpumpedWavefunctions = []

        pulse2Tuples = []


        pumpLoopStartTime = time.time()
        print "setup time: ", pumpLoopStartTime - setupStartTime


        for pumpTime in self.pumpTimes:
            pulse2tuple = (self.firstProbeEx.pulseCopyJumpedForward(pumpTime), self.firstProbeEy.pulseCopyJumpedForward(pumpTime), self.firstProbeEz.pulseCopyJumpedForward(pumpTime))
            pulse2Tuples.append(pulse2tuple)

            pulse2xPLUS = pulse2tuple[0].plusFunction()
            pulse2xMINUS = pulse2tuple[0].minusFunction()

            pulse2yPLUS = pulse2tuple[1].plusFunction()
            pulse2yMINUS = pulse2tuple[1].minusFunction()

            pulse2zPLUS = pulse2tuple[2].plusFunction()
            pulse2zMINUS = pulse2tuple[2].minusFunction()

            pulse2PLUStuple = (pulse2xPLUS, pulse2yPLUS, pulse2zPLUS)
            pulse2MINUStuple = (pulse2xMINUS, pulse2yMINUS, pulse2zMINUS)

            k2PlusInteractor = TimePerturbation.electricDipoleOneInteraction(space = self.mySpace,
                                                                           electronicHamiltonian = self.myElectronicHamiltonian,
                                                                           MuxMuyMuzElectronicOperatorTuple = (self.MuX, self.MuY, self.MuZ),
                                                                           ExEyEzTimeFunctionTuple = pulse2PLUStuple,
                                                                           maxTime = self.maximumTime)

            k2MinusInteractor = TimePerturbation.electricDipoleOneInteraction(space = self.mySpace,
                                                                           electronicHamiltonian = self.myElectronicHamiltonian,
                                                                           MuxMuyMuzElectronicOperatorTuple = (self.MuX, self.MuY, self.MuZ),
                                                                           ExEyEzTimeFunctionTuple = pulse2MINUStuple,
                                                                           maxTime = self.maximumTime)


            self.k1MINUS_k2PLUSpumpedWavefunctions.append(interactionHelper(k2PlusInteractor, self.k1MinusWavefunction))
            self.k1PLUS_k2MINUSpumpedWavefunctions.append(interactionHelper(k2MinusInteractor, self.k1PlusWavefunction))

        self.k1MINUS_k2PLUSpumpedWavefunctions = myPool.map(BIG_RED_BUTTON, self.k1MINUS_k2PLUSpumpedWavefunctions)
        self.k1PLUS_k2MINUSpumpedWavefunctions = myPool.map(BIG_RED_BUTTON, self.k1PLUS_k2MINUSpumpedWavefunctions)


        self.k1MINUS_k2PLUS_k3PLUSpumpedWavefunctions = []
        self.k1PLUS_k2MINUS_k3PLUSpumpedWavefunctions = []

        pulse3tuples = []

        pumpLoop2StartTime = time.time()
        print "pump loop 1 time: ", pumpLoop2StartTime - pumpLoopStartTime

        for i, pulse2Tuple in enumerate(pulse2Tuples):
            pulse3tuple = (pulse2Tuple[0].pulseCopyJumpedForward(self.T), pulse2Tuple[1].pulseCopyJumpedForward(self.T), pulse2Tuple[2].pulseCopyJumpedForward(self.T))
            pulse3tuples.append(pulse3tuple)

            pulse3PLUStuple = (pulse3tuple[0].plusFunction(), pulse3tuple[1].plusFunction(), pulse3tuple[2].plusFunction())

            k3PlusInteractor = TimePerturbation.electricDipoleOneInteraction(space = self.mySpace,
                                                                           electronicHamiltonian = self.myElectronicHamiltonian,
                                                                           MuxMuyMuzElectronicOperatorTuple = (self.MuX, self.MuY, self.MuZ),
                                                                           ExEyEzTimeFunctionTuple = pulse3PLUStuple,
                                                                           maxTime = self.maximumTime)

            self.k1MINUS_k2PLUS_k3PLUSpumpedWavefunctions.append(interactionHelper(k3PlusInteractor, self.k1MINUS_k2PLUSpumpedWavefunctions[i]))
            self.k1PLUS_k2MINUS_k3PLUSpumpedWavefunctions.append(interactionHelper(k3PlusInteractor, self.k1PLUS_k2MINUSpumpedWavefunctions[i]))

        self.k1MINUS_k2PLUS_k3PLUSpumpedWavefunctions = myPool.map(BIG_RED_BUTTON, self.k1MINUS_k2PLUS_k3PLUSpumpedWavefunctions)
        self.k1PLUS_k2MINUS_k3PLUSpumpedWavefunctions = myPool.map(BIG_RED_BUTTON, self.k1PLUS_k2MINUS_k3PLUSpumpedWavefunctions)


        self.resphasingTimeSignals = np.zeros((len(self.pumpTimes), len(self.probeTimes)), dtype = np.complex)
        self.nonresphasingTimeSignals = np.zeros((len(self.pumpTimes), len(self.probeTimes)), dtype = np.complex)

        probeLoopStartTime = time.time()
        print "pumpLoop2 time: ", probeLoopStartTime - pumpLoop2StartTime
        intermediateTime = time.time()
        for i, pulse3Tuple in enumerate(pulse3tuples):
            print "past one inner loop...", time.time() - intermediateTime
            intermediateTime = time.time()
            for j, probeTime in enumerate(self.probeTimes):

                pulse4MINUS = (pulse3Tuple[0].pulseCopyJumpedForward(probeTime).minusFunction(), pulse3Tuple[1].pulseCopyJumpedForward(probeTime).minusFunction(), pulse3Tuple[2].pulseCopyJumpedForward(probeTime).minusFunction())
                rephasingSignal = (self.zeroOrderTimeWavefunction + self.k1MINUS_k2PLUS_k3PLUSpumpedWavefunctions[i]).timeExpectationValueOfPolarizationAndOverlapWithElectricField((self.MuX, self.MuY, self.MuZ), pulse4MINUS)
                nonrephasingSignal = (self.zeroOrderTimeWavefunction + self.k1PLUS_k2MINUS_k3PLUSpumpedWavefunctions[i]).timeExpectationValueOfPolarizationAndOverlapWithElectricField((self.MuX, self.MuY, self.MuZ), pulse4MINUS)
                self.resphasingTimeSignals[i, j] = rephasingSignal
                self.nonresphasingTimeSignals[i, j] = nonrephasingSignal

        probeLoopTime = time.time()
        print "probe loop time: ", probeLoopTime - probeLoopStartTime

        self.calculationTime = time.time() - overallStartTime

        self.rawResphasingFrequencySignals = scipy.fftpack.fftn(self.resphasingTimeSignals)
        self.rawNonresphasingFrequencySignals = scipy.fftpack.fftn(self.nonresphasingTimeSignals)

        self.pumpFrequencies = 2 * np.pi * scipy.fftpack.fftshift(scipy.fftpack.fftfreq(len(self.pumpTimes), d = self.pumpTimes[2] - self.pumpTimes[1]))
        self.probeFrequencies = 2 * np.pi * scipy.fftpack.fftshift(scipy.fftpack.fftfreq(len(self.probeTimes), d = self.probeTimes[2] - self.probeTimes[1]))

        self.pumpFrequenciesWavenumbers = self.mySpace.unitHandler.wavenumbersFromEnergyUnits(self.pumpFrequencies)
        self.probeFrequenciesWavenumbers = self.mySpace.unitHandler.wavenumbersFromEnergyUnits(self.probeFrequencies)

        return self

    def plotTime(self, newFig = True):
        if newFig:
            plt.figure()
        pumpTime = self.pumpTimes
        pumpTime = self.mySpace.unitHandler.femtosecondsFromTime(pumpTime)

        probeTime = self.probeTimes
        probeTime = self.mySpace.unitHandler.femtosecondsFromTime(probeTime)

        plt.contourf(pumpTime, probeTime, np.abs(self.nonresphasingTimeSignals + self.resphasingTimeSignals))
        plt.title("2D Spectrum (Untransformed)")
        plt.xlabel(r"$T_{pump}$")
        plt.ylabel(r"$T_{probe}$")


    def plotFrequency(self, newFig = True):
        if newFig:
            plt.figure()
        pumpFrequencies = self.pumpFrequencies
        pumpFrequencies = self.mySpace.unitHandler.wavenumbersFromEnergyUnits(pumpFrequencies)

        probeFrequencies = self.probeFrequencies
        probeFrequencies = self.mySpace.unitHandler.wavenumbersFromEnergyUnits(probeFrequencies)

        plt.contourf(self.pumpFrequenciesWavenumbers, self.probeFrequenciesWavenumbers, np.abs(self.rawResphasingFrequencySignals + self.rawNonresphasingFrequencySignals))
        plt.title("2D Spectrum")
        plt.xlabel(r"$\omega_{pump}$ (wavenumbers)")
        plt.ylabel(r"$\omega_{probe}$ (wavenumbers)")



def BIG_RED_BUTTON(ppObject):
    "just here to make embarassingly parallel calculations easier"
    return ppObject.calculate()


class interactionHelper(object):
    def __init__(self, interactor, wavefunction):
        self.interactor = interactor
        self.wavefunction = wavefunction

    def calculate(self):
        return self.interactor.goFromTimeWavefunction(self.wavefunction)

if __name__ == "__main__":
    print "heeeeey"
