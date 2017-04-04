# -*- coding: utf-8 -*-
"""
Created on Fri Jan 04 13:23:27 2013

@author: Joey
"""
import copy
import time
import pickle
import datetime
import warnings

import cmath

import numpy as np
import scipy.integrate

import multiprocessing
import matplotlib as mpl
import matplotlib.pyplot as plt

import spectroscopy.Spacetime as Spacetime
import spectroscopy.TimeElectronicWavefunction as TimeElectronicWavefunction
import spectroscopy.TimeFunction as TimeFunction
import spectroscopy.TimePerturbation as TimePerturbation

import experimentBase

try:
    import ClusterPool.ClusterPool as ClusterPool
    import ClusterPool.Dispatcher as Dispatcher
except ImportError:
    warnings.warn("ClusterPool Module Not Found, experiments will not work with use_clusterpool=True")

NUMBER_PROCESSES = 5
USE_CLUSTERPOOL = False

def BIG_RED_BUTTON(ppObject):
    "just here to make embarassingly parallel calculations easier"
    print "BOMBS AWAY"
    return ppObject.calculate()


class Base(experimentBase.experiment):
    "Base class for calculating pump probe spectra"
    experiment_type_string = "baseclass_Pump_Probe_"


    def __init__(self, space,
                 electronicHamiltonian,
                 MuxMuyMuzElectronicOperatorTuple,
                 initialElectronicWavefunction,
                 pumpBeamTuple,
                 maximumEvolutionTime,
                 numberOfTimePoints,
                 IDstring = "",
                 use_clusterpool = USE_CLUSTERPOOL):
        """
        An object to calculate the pump probe spectrum of a given system

        @param space: the SpaceTime object the system lives in
        @param electronicHamiltonian: Hamiltonian for the system
        @param MuxMuyMuzElectronicOperatorTuple: tuple of eletronic operators (x, y, z) which describe the transition dipole of the system
        @param initialElectronicWavefunction: the starting state of the system
        @param pumpBeamTuple: The first laser Beam.  will be translated in time to get the probe or second laser Beam.  is a tuple of (x, y, z) components

        @param maximumEvolutionTime; required maximum population time
        @param numberOfTimePoints: how many population time points to calcaulte

        @return: a Pump Probe object
        """
        #store all the variables
        self.mySpace = space

        self.use_clusterpool = use_clusterpool

        self.MuX, self.MuY, self.MuZ = MuxMuyMuzElectronicOperatorTuple
        self.pumpEx, self.pumpEy, self.pumpEz = pumpBeamTuple
        self.pumpBeamTuple = pumpBeamTuple

        self.intitialEWF = initialElectronicWavefunction
        self.myElectronicHamiltonian = electronicHamiltonian
        self.myFreePropagator = electronicHamiltonian.myPropagator()

        #We are given a target maximum waiting and a specific number of time points to take.
        #We then have to scale that to the closest multiple of mySpace.dt to make the propagation work properly

        self.Npp = int(numberOfTimePoints)
        targetMaxPopulationTime = float(maximumEvolutionTime)
        targetDtPP = targetMaxPopulationTime  / float(numberOfTimePoints)

        self.WFtimeStepsPerPPTimeSteps = int(np.round(targetDtPP / self.mySpace.dt, 0))
        #What if it's zero!?  We can't have that
        if self.WFtimeStepsPerPPTimeSteps == 0:
            self.WFtimeStepsPerPPTimeSteps = 1

        self.dtPp = self.WFtimeStepsPerPPTimeSteps * self.mySpace.dt

        self.TmaxPp =  self.dtPp * (self.Npp)

        self.timeSeriesPulseOverlaps = []


        self.firstProbeEx = copy.deepcopy(self.pumpEx)
        self.firstProbeEy = copy.deepcopy(self.pumpEy)
        self.firstProbeEz = copy.deepcopy(self.pumpEz)

        self.probePower = self.firstProbeEx.totalPower() + self.firstProbeEy.totalPower() + self.firstProbeEz.totalPower()

        self.calculated = False
        self.IDstring = IDstring

        self.__maxPumpTime = self.TmaxPp + 3.0 * self.firstProbeEx.timePillow


        self.pulse_overlap_ending_time = 6.0 * max(self.firstProbeEx.sigma, self.firstProbeEy.sigma, self.firstProbeEz.sigma)
        self.pulse_overlap_ending_index = self.pulse_overlap_ending_time / self.dtPp
        self.pulse_overlap_ending_index = int(self.pulse_overlap_ending_index)

        self.dipoleTuple = (self.MuX, self.MuY, self.MuZ)
        self.excitationMuTuple = (self.dipoleTuple[0].excitationOperator(), self.dipoleTuple[1].excitationOperator(), self.dipoleTuple[2].excitationOperator())
        self.relaxationMuTuple = (self.dipoleTuple[0].relaxationOperator(), self.dipoleTuple[1].relaxationOperator(), self.dipoleTuple[2].relaxationOperator())



        self.maximumTime = self.TmaxPp + 2.0 * 2.0 * max(map(lambda x: x.timePillow, self.pumpBeamTuple))
        self.maximumTimeSteps = int(np.round(self.maximumTime / self.mySpace.dt))

        self.ppTimeSeries = []


        self.monomer = (self.mySpace.electronicDimensionality == 2)

class Basic(Base):
    """
    This object simply calcualtes the Pump Probe spectrum in serial on a normal computer
    """
    experiment_type_string = "Basic_Pump_Probe_"

    def calculate(self):
        if self.calculated:
            print "You've already done the calculation, look at the results instead!"
            return
        overallStartTime = time.time()

        self.timeSeriesSignal_NON_RWA  = []
        self.timeSeriesSignal_RWA  = []

        self.ppSignals = []

        probeEx = copy.deepcopy(self.firstProbeEx)
        probeEy = copy.deepcopy(self.firstProbeEy)
        probeEz = copy.deepcopy(self.firstProbeEz)

        self.ppSignals = []

        zeroOrderTimeWavefunction = TimeElectronicWavefunction.timeElectronicWavefunction(self.mySpace)
        zeroOrderTimeWavefunction.applyOperatorsNTimesOnInitialWavefunction([self.myFreePropagator],N = self.maximumTimeSteps,initialWF = self.intitialEWF)

        pumpxPLUS = self.firstProbeEx.plusFunction()
        pumpxMINUS = self.firstProbeEx.minusFunction()

        pumpyPLUS = self.firstProbeEy.plusFunction()
        pumpyMINUS = self.firstProbeEy.minusFunction()

        pumpzPLUS = self.firstProbeEz.plusFunction()
        pumpzMINUS = self.firstProbeEz.minusFunction()

        pumpPLUStuple = (pumpxPLUS, pumpyPLUS, pumpzPLUS)
        pumpMINUStuple = (pumpxMINUS, pumpyMINUS, pumpzMINUS)

        pumpPlusInteractor = TimePerturbation.electricDipoleOneInteraction(space = self.mySpace, electronicHamiltonian = self.myElectronicHamiltonian, MuxMuyMuzElectronicOperatorTuple = self.excitationMuTuple, ExEyEzTimeFunctionTuple = pumpPLUStuple, maxTime = self.maximumTime)

        pumpMinusInteractor = TimePerturbation.electricDipoleOneInteraction(space = self.mySpace, electronicHamiltonian = self.myElectronicHamiltonian, MuxMuyMuzElectronicOperatorTuple = self.relaxationMuTuple, ExEyEzTimeFunctionTuple = pumpMINUStuple, maxTime = self.maximumTime)

        pumpPlus_Wavefunction = pumpPlusInteractor.goFromTimeWavefunction(zeroOrderTimeWavefunction)
#        pumpMinus_Wavefunction = pumpMinusInteractor.goFromTimeWavefunction(zeroOrderTimeWavefunction)
        self.pumpPlus_Wavefunction = pumpPlus_Wavefunction
        #wavefunctions named in order of interaction:: first interaction first
        pumpPlus_PumpMinus_Wavefunction = pumpMinusInteractor.goFromTimeWavefunction(pumpPlus_Wavefunction)
#        pumpMinus_PumpPlus_Wavefunction = pumpMinusInteractor.goFromTimeWavefunction(pumpPlus_Wavefunction)


        self.timeSeriesSignalExcitedStateAbsoprtion = []
        self.timeSeriesSignalGroundStateBleachA = []
        self.timeSeriesSignalGroundStateBleachB = []
        self.timeSeriesSignalGroundStateBleach = []
        self.timeSeriesSignalStimulatedEmission = []
        self.timeSeriesTotalSignal = []

        self.timeSeriesPulseOverlaps = []

        self.currentT = 0.0
        completedCalculations = 0

        self.probes = []

        for ii in range(self.Npp):
            self.probes.append(probeEx)
            overlapFunction = self.firstProbeEx * probeEx + self.firstProbeEy * probeEy + self.firstProbeEz * probeEz
            pulseOverlap = overlapFunction.integrateInSimulationSpace()
            self.timeSeriesPulseOverlaps.append(pulseOverlap)

            probeTuple = (probeEx, probeEy, probeEz)

            probePlusTuple = (probeEx.plusFunction(), probeEy.plusFunction(), probeEz.plusFunction())
            probeMinusTuple = (probeEx.minusFunction(), probeEy.minusFunction(), probeEz.minusFunction())


            probePlusInteractor = TimePerturbation.electricDipoleOneInteraction(space = self.mySpace, electronicHamiltonian = self.myElectronicHamiltonian, MuxMuyMuzElectronicOperatorTuple = self.excitationMuTuple, ExEyEzTimeFunctionTuple = probePlusTuple, maxTime = self.maximumTime)

            probeMinusInteractor = TimePerturbation.electricDipoleOneInteraction(space = self.mySpace, electronicHamiltonian = self.myElectronicHamiltonian, MuxMuyMuzElectronicOperatorTuple = self.relaxationMuTuple, ExEyEzTimeFunctionTuple = probeMinusTuple, maxTime = self.maximumTime)

            pumpPlus_PumpMinus_ProbePlus_Wavefunction = probePlusInteractor.goFromTimeWavefunction(pumpPlus_PumpMinus_Wavefunction)



            pumpPlus_ProbeMinus_Wavefunction = probeMinusInteractor.goFromTimeWavefunction(pumpPlus_Wavefunction)

            probePlus_Wavefunction = probePlusInteractor.goFromTimeWavefunction(zeroOrderTimeWavefunction)


            #NOTE WE ARE USING THE MINUS PUMP BEAM TO ACHIEVE THE NEEDED COMPLEX CONJUGATE OPERATION ON EITHER THE LASER BEAM OR THE EMISSION
            #NOTE THAT I MESSED THAT UP AND WE SHOULD BE USING THE FULL PROBE BEAM TO INTERFERE
            signal_SE = pumpPlus_Wavefunction.timeOverlapWithOtherBraEWFOfPolarizationAndOverlapWithElectricField(pumpPlus_ProbeMinus_Wavefunction, self.dipoleTuple, probePlusTuple)
            signal_SE = (2 * 1.0j * signal_SE).real

            if self.monomer ==False:
                pumpPlus_ProbePlus_Wavefunction = probePlusInteractor.goFromTimeWavefunction(pumpPlus_Wavefunction)
                signal_ESA = pumpPlus_ProbePlus_Wavefunction.timeOverlapWithOtherBraEWFOfPolarizationAndOverlapWithElectricField(pumpPlus_Wavefunction, self.dipoleTuple, probePlusTuple)
                signal_ESA = (2 * 1.0j * signal_ESA).real
            else:
                signal_ESA = 0.0

            signal_GSB_1 = pumpPlus_PumpMinus_ProbePlus_Wavefunction.timeOverlapWithOtherBraEWFOfPolarizationAndOverlapWithElectricField(zeroOrderTimeWavefunction, self.dipoleTuple, probePlusTuple)
            signal_GSB_1 = (2 * 1.0j * signal_GSB_1).real

            signal_GSB_2 = probePlus_Wavefunction.timeOverlapWithOtherBraEWFOfPolarizationAndOverlapWithElectricField(pumpPlus_PumpMinus_Wavefunction, self.dipoleTuple, probePlusTuple)
            signal_GSB_2 = (2 * 1.0j * signal_GSB_2).real


            self.timeSeriesSignalExcitedStateAbsoprtion.append(signal_ESA)
            self.timeSeriesSignalGroundStateBleachA.append(signal_GSB_1)
            self.timeSeriesSignalGroundStateBleachB.append(signal_GSB_2)
            self.timeSeriesSignalStimulatedEmission.append(signal_SE)

            self.timeSeriesTotalSignal.append(signal_ESA + signal_GSB_1 + signal_GSB_2 + signal_SE)

            self.ppTimeSeries.append(self.currentT)

            self.currentT = self.currentT + self.dtPp

            probeEx = probeEx.pulseCopyJumpedForward(self.dtPp)
            probeEy = probeEy.pulseCopyJumpedForward(self.dtPp)
            probeEz = probeEz.pulseCopyJumpedForward(self.dtPp)


            completedCalculations = completedCalculations + 1

#            print "Time Elapsed", -startTime + time.time()



        self.calculated = True
        print "Time for post-processing"

        self.timeSeriesSignalExcitedStateAbsoprtion = np.array(self.timeSeriesSignalExcitedStateAbsoprtion)
        self.timeSeriesSignalGroundStateBleach = np.array(self.timeSeriesSignalGroundStateBleach)
        self.timeSeriesSignalStimulatedEmission = np.array(self.timeSeriesSignalStimulatedEmission)
        self.timeSeriesTotalSignal = np.array(self.timeSeriesTotalSignal)
        self.timeSeriesPulseOverlaps = np.array(self.timeSeriesPulseOverlaps)

        self.ppTimeSeries = np.array(self.ppTimeSeries)
        self.ppTimeSeries_fs = self.mySpace.unitHandler.femtosecondsFromTime(self.ppTimeSeries)

        self.ppOmegaSeries, self.frequencySeriesTotalSignal = self.mySpace.genericOneDimensionalFourierTransformFromZero(self.ppTimeSeries, self.timeSeriesTotalSignal, gaussianFilter = False)


        self.timeElapsed = time.time() - overallStartTime
        print self.IDstring + ": time elapsed (s) for PP calculation", self.timeElapsed

        #perform calculations on the non-pulse-overlap region


        return self

class Basic_Overkill(Base):
    """
    This object calcualtes the Pump Probe spectrum in serial on a normal computer and includes laser interactions which are not time-ordered (probe before pump) to exactly calculate the pulse-overlap regime
    """
    experiment_type_string = "Basic_Overkill_Pump_Probe_"

    def calculate(self):
        if self.calculated:
            print "You've already done the calculation, look at the results instead!"
            return
        overallStartTime = time.time()

        self.timeSeriesSignal_NON_RWA  = []
        self.timeSeriesSignal_RWA  = []

        self.ppSignals = []

        probeEx = copy.deepcopy(self.firstProbeEx)
        probeEy = copy.deepcopy(self.firstProbeEy)
        probeEz = copy.deepcopy(self.firstProbeEz)

        self.ppSignals = []

        zeroOrderTimeWavefunction = TimeElectronicWavefunction.timeElectronicWavefunction(self.mySpace)
        zeroOrderTimeWavefunction.applyOperatorsNTimesOnInitialWavefunction([self.myFreePropagator],
                                                                            N = self.maximumTimeSteps,
                                                                            initialWF = self.intitialEWF)

        pumpxPLUS = self.firstProbeEx.plusFunction()
        pumpxMINUS = self.firstProbeEx.minusFunction()

        pumpyPLUS = self.firstProbeEy.plusFunction()
        pumpyMINUS = self.firstProbeEy.minusFunction()

        pumpzPLUS = self.firstProbeEz.plusFunction()
        pumpzMINUS = self.firstProbeEz.minusFunction()

        pumpPLUStuple = (pumpxPLUS, pumpyPLUS, pumpzPLUS)
        pumpMINUStuple = (pumpxMINUS, pumpyMINUS, pumpzMINUS)

        pumpPlusInteractor = TimePerturbation.electricDipoleOneInteraction(space = self.mySpace, electronicHamiltonian = self.myElectronicHamiltonian, MuxMuyMuzElectronicOperatorTuple = self.excitationMuTuple, ExEyEzTimeFunctionTuple = pumpPLUStuple, maxTime = self.maximumTime)

        pumpMinusInteractor = TimePerturbation.electricDipoleOneInteraction(space = self.mySpace, electronicHamiltonian = self.myElectronicHamiltonian, MuxMuyMuzElectronicOperatorTuple = self.relaxationMuTuple, ExEyEzTimeFunctionTuple = pumpMINUStuple, maxTime = self.maximumTime)

        pumpPlus_Wavefunction = pumpPlusInteractor.goFromTimeWavefunction(zeroOrderTimeWavefunction)
#        pumpMinus_Wavefunction = pumpMinusInteractor.goFromTimeWavefunction(zeroOrderTimeWavefunction)
        self.pumpPlus_Wavefunction = pumpPlus_Wavefunction
        #wavefunctions named in order of interaction:: first interaction first
        pumpPlus_PumpMinus_Wavefunction = pumpMinusInteractor.goFromTimeWavefunction(pumpPlus_Wavefunction)
#        pumpMinus_PumpPlus_Wavefunction = pumpMinusInteractor.goFromTimeWavefunction(pumpPlus_Wavefunction)


        self.timeSeriesSignalExcitedStateAbsoprtion = []
        self.timeSeriesSignalGroundStateBleachA = []
        self.timeSeriesSignalGroundStateBleachB = []
        self.timeSeriesSignalGroundStateBleach = []
        self.timeSeriesSignalStimulatedEmission = []
        self.timeSeriesSignal_overkill1 = []
        self.timeSeriesSignal_overkill2 = []
        self.timeSeriesSignal_overkill3 = []
        self.timeSeriesTotalSignal = []

        self.timeSeriesPulseOverlaps = []

        self.currentT = 0.0
        completedCalculations = 0

        self.probes = []

        for ii in range(self.Npp):
            self.probes.append(probeEx)
            overlapFunction = self.firstProbeEx * probeEx + self.firstProbeEy * probeEy + self.firstProbeEz * probeEz
            pulseOverlap = overlapFunction.integrateInSimulationSpace()
            self.timeSeriesPulseOverlaps.append(pulseOverlap)

            probeTuple = (probeEx, probeEy, probeEz)

            probePlusTuple = (probeEx.plusFunction(), probeEy.plusFunction(), probeEz.plusFunction())
            probeMinusTuple = (probeEx.minusFunction(), probeEy.minusFunction(), probeEz.minusFunction())


            probePlusInteractor = TimePerturbation.electricDipoleOneInteraction(space = self.mySpace, electronicHamiltonian = self.myElectronicHamiltonian, MuxMuyMuzElectronicOperatorTuple = self.excitationMuTuple, ExEyEzTimeFunctionTuple = probePlusTuple, maxTime = self.maximumTime)

            probeMinusInteractor = TimePerturbation.electricDipoleOneInteraction(space = self.mySpace, electronicHamiltonian = self.myElectronicHamiltonian, MuxMuyMuzElectronicOperatorTuple = self.relaxationMuTuple, ExEyEzTimeFunctionTuple = probeMinusTuple, maxTime = self.maximumTime)

            pumpPlus_PumpMinus_ProbePlus_Wavefunction = probePlusInteractor.goFromTimeWavefunction(pumpPlus_PumpMinus_Wavefunction)



            pumpPlus_ProbeMinus_Wavefunction = probeMinusInteractor.goFromTimeWavefunction(pumpPlus_Wavefunction)

            probePlus_Wavefunction = probePlusInteractor.goFromTimeWavefunction(zeroOrderTimeWavefunction)

            ####
            #NOT-PROPERLY-TIME-ORDERED WFs
            probePlus_pumpMinus_Wavefunction = pumpMinusInteractor.goFromTimeWavefunction(probePlus_Wavefunction)
            probePlus_pumpMinus_pumpPlus_Wavefunction = pumpPlusInteractor.goFromTimeWavefunction(probePlus_pumpMinus_Wavefunction)
            signal_overkill1 = probePlus_pumpMinus_pumpPlus_Wavefunction.timeOverlapWithOtherBraEWFOfPolarizationAndOverlapWithElectricField(zeroOrderTimeWavefunction, self.dipoleTuple, probePlusTuple)
            signal_overkill1 = (2 * 1.0j * signal_overkill1).real

            #NOTE WE ARE USING THE MINUS PUMP BEAM TO ACHIEVE THE NEEDED COMPLEX CONJUGATE OPERATION ON EITHER THE LASER BEAM OR THE EMISSION
            #NOTE THAT I MESSED THAT UP AND WE SHOULD BE USING THE FULL PROBE BEAM TO INTERFERE
            signal_SE = pumpPlus_Wavefunction.timeOverlapWithOtherBraEWFOfPolarizationAndOverlapWithElectricField(pumpPlus_ProbeMinus_Wavefunction, self.dipoleTuple, probePlusTuple)
            signal_SE = (2 * 1.0j * signal_SE).real

            if self.monomer ==False:
                pumpPlus_ProbePlus_Wavefunction = probePlusInteractor.goFromTimeWavefunction(pumpPlus_Wavefunction)
                signal_ESA = pumpPlus_ProbePlus_Wavefunction.timeOverlapWithOtherBraEWFOfPolarizationAndOverlapWithElectricField(pumpPlus_Wavefunction, self.dipoleTuple, probePlusTuple)
                signal_ESA = (2 * 1.0j * signal_ESA).real

                #NOT-PROPERLY-TIME-ORDERED WFs
                probePlus_pumpPlus_Wavefunction = pumpPlusInteractor.goFromTimeWavefunction(probePlus_Wavefunction)
                probePlus_pumpPlus_pumpMinus_Wavefunction = probeMinusInteractor.goFromTimeWavefunction(probePlus_pumpPlus_Wavefunction)
                signal_overkill2 = probePlus_pumpPlus_pumpMinus_Wavefunction.timeOverlapWithOtherBraEWFOfPolarizationAndOverlapWithElectricField(zeroOrderTimeWavefunction, self.dipoleTuple, probePlusTuple)
                signal_overkill2 = (2 * 1.0j * signal_overkill2).real
                signal_overkill3 = probePlus_pumpPlus_Wavefunction.timeOverlapWithOtherBraEWFOfPolarizationAndOverlapWithElectricField(pumpPlus_Wavefunction, self.dipoleTuple, probePlusTuple)
                signal_overkill3 = (2 * 1.0j * signal_overkill3).real
            else:
                signal_ESA = 0.0
                signal_overkill2 = 0.0
                signal_overkill3 = 0.0

            signal_GSB_1 = pumpPlus_PumpMinus_ProbePlus_Wavefunction.timeOverlapWithOtherBraEWFOfPolarizationAndOverlapWithElectricField(zeroOrderTimeWavefunction, self.dipoleTuple, probePlusTuple)
            signal_GSB_1 = (2 * 1.0j * signal_GSB_1).real

            signal_GSB_2 = probePlus_Wavefunction.timeOverlapWithOtherBraEWFOfPolarizationAndOverlapWithElectricField(pumpPlus_PumpMinus_Wavefunction, self.dipoleTuple, probePlusTuple)
            signal_GSB_2 = (2 * 1.0j * signal_GSB_2).real


            self.timeSeriesSignalExcitedStateAbsoprtion.append(signal_ESA)
            self.timeSeriesSignalGroundStateBleachA.append(signal_GSB_1)
            self.timeSeriesSignalGroundStateBleachB.append(signal_GSB_2)
            self.timeSeriesSignalStimulatedEmission.append(signal_SE)
            self.timeSeriesSignal_overkill1.append(signal_overkill1)
            self.timeSeriesSignal_overkill2.append(signal_overkill2)
            self.timeSeriesSignal_overkill3.append(signal_overkill3)

            self.timeSeriesTotalSignal.append(signal_ESA + signal_GSB_1 + signal_GSB_2 + signal_SE + signal_overkill1 + signal_overkill2 + signal_overkill3)

            self.ppTimeSeries.append(self.currentT)

            self.currentT = self.currentT + self.dtPp

            probeEx = probeEx.pulseCopyJumpedForward(self.dtPp)
            probeEy = probeEy.pulseCopyJumpedForward(self.dtPp)
            probeEz = probeEz.pulseCopyJumpedForward(self.dtPp)

            completedCalculations = completedCalculations + 1




        self.calculated = True
        print "Time for post-processing"

        self.timeSeriesSignalExcitedStateAbsoprtion = np.array(self.timeSeriesSignalExcitedStateAbsoprtion)
        self.timeSeriesSignalGroundStateBleach = np.array(self.timeSeriesSignalGroundStateBleach)
        self.timeSeriesSignalStimulatedEmission = np.array(self.timeSeriesSignalStimulatedEmission)
        self.timeSeriesTotalSignal = np.array(self.timeSeriesTotalSignal)
        self.timeSeriesPulseOverlaps = np.array(self.timeSeriesPulseOverlaps)

        self.ppTimeSeries = np.array(self.ppTimeSeries)
        self.ppTimeSeries_fs = self.mySpace.unitHandler.femtosecondsFromTime(self.ppTimeSeries)

        self.ppOmegaSeries, self.frequencySeriesTotalSignal = self.mySpace.genericOneDimensionalFourierTransformFromZero(self.ppTimeSeries, self.timeSeriesTotalSignal, gaussianFilter = False)


        self.timeElapsed = time.time() - overallStartTime
        print self.IDstring + ": time elapsed (s) for PP calculation", self.timeElapsed


        return self


class Basic_Cluster(Base):
    """
    This object calcualtes the Pump Probe spectrum in serial on a computational cluster using my ClusterPool package
    """
    experiment_type_string = "Cluster_Pump_Probe_"

    def calculate(self):
        if self.calculated:
            print "You've already done the calculation, look at the results instead!"
            return

        self.timeSeriesSignal_NON_RWA  = []
        self.timeSeriesSignal_RWA  = []

        self.ppSignals = []

        probeEx = copy.deepcopy(self.firstProbeEx)
        probeEy = copy.deepcopy(self.firstProbeEy)
        probeEz = copy.deepcopy(self.firstProbeEz)

        self.ppSignals = []

        zeroOrderTimeWavefunction = TimeElectronicWavefunction.timeElectronicWavefunction(self.mySpace)
        zeroOrderTimeWavefunction.applyOperatorsNTimesOnInitialWavefunction([self.myFreePropagator], N = self.maximumTimeSteps, initialWF = self.intitialEWF)

        zero_order_file_location = Dispatcher.Dispatcher.save_shared_object_return_filename(zeroOrderTimeWavefunction, "zero_order_ewf_pp" )

        pumpxPLUS = self.firstProbeEx.plusFunction()
        pumpxMINUS = self.firstProbeEx.minusFunction()

        pumpyPLUS = self.firstProbeEy.plusFunction()
        pumpyMINUS = self.firstProbeEy.minusFunction()

        pumpzPLUS = self.firstProbeEz.plusFunction()
        pumpzMINUS = self.firstProbeEz.minusFunction()

        pumpPLUStuple = (pumpxPLUS, pumpyPLUS, pumpzPLUS)
        pumpMINUStuple = (pumpxMINUS, pumpyMINUS, pumpzMINUS)

        pumpPlusInteractor = TimePerturbation.electricDipoleOneInteraction(space = self.mySpace, electronicHamiltonian = self.myElectronicHamiltonian, MuxMuyMuzElectronicOperatorTuple = self.excitationMuTuple, ExEyEzTimeFunctionTuple = pumpPLUStuple, maxTime = self.maximumTime)

        pumpMinusInteractor = TimePerturbation.electricDipoleOneInteraction(space = self.mySpace, electronicHamiltonian = self.myElectronicHamiltonian, MuxMuyMuzElectronicOperatorTuple = self.relaxationMuTuple, ExEyEzTimeFunctionTuple = pumpMINUStuple, maxTime = self.maximumTime)

        pumpPlus_Wavefunction = pumpPlusInteractor.goFromTimeWavefunction(zeroOrderTimeWavefunction)
        pumpPlus_file_location = Dispatcher.Dispatcher.save_shared_object_return_filename(pumpPlus_Wavefunction, "pumpPlus_ewf_pp")

        #wavefunctions named in order of interaction:: first interaction first
        pumpPlus_PumpMinus_Wavefunction = pumpMinusInteractor.goFromTimeWavefunction(pumpPlus_Wavefunction)

        pumpPlus_PumpMinus_file_location = Dispatcher.Dispatcher.save_shared_object_return_filename(pumpPlus_PumpMinus_Wavefunction, "pumpPlus_PumpMinus_ewf_pp")


        #delete wavefunctions from memory:
        zeroOrderTimeWavefunction = None
        pumpPlus_Wavefunction = None
        pumpPlus_PumpMinus_Wavefunction = None

        #create lists for the various signal types
        self.timeSeriesSignalExcitedStateAbsoprtion = np.zeros( [self.Npp], dtype= np.complex)
        self.timeSeriesSignalGroundStateBleachA = np.zeros( [self.Npp], dtype= np.complex)
        self.timeSeriesSignalGroundStateBleachB = np.zeros( [self.Npp], dtype= np.complex)
        self.timeSeriesSignalStimulatedEmission = np.zeros( [self.Npp], dtype= np.complex)
        self.timeSeriesTotalSignal = np.zeros( [self.Npp], dtype= np.complex)

        listOfCalculations = []

        self.currentT = 0.0


        stimulatedEmission_signalType_string = "se"
        excitedStateAbsorption_signalType_string = "esa"
        GroundStateBleach1_signalType_string = "gsb1"
        GroundStateBleach2_signalType_string = "gsb2"


        for ii in range(self.Npp):

            overlapFunction = self.firstProbeEx * probeEx + self.firstProbeEy * probeEy + self.firstProbeEz * probeEz
            pulseOverlap = overlapFunction.integrateInSimulationSpace()
            self.timeSeriesPulseOverlaps.append(pulseOverlap)

            probeTuple = (probeEx, probeEy, probeEz)
            probePlusTuple = (probeEx.plusFunction(), probeEy.plusFunction(), probeEz.plusFunction())
            probeMinusTuple = (probeEx.minusFunction(), probeEy.minusFunction(), probeEz.minusFunction())


            probePlusInteractor = TimePerturbation.electricDipoleOneInteraction(space = self.mySpace, electronicHamiltonian = self.myElectronicHamiltonian, MuxMuyMuzElectronicOperatorTuple = self.excitationMuTuple, ExEyEzTimeFunctionTuple = probePlusTuple, maxTime = self.maximumTime)

            probeMinusInteractor = TimePerturbation.electricDipoleOneInteraction(space = self.mySpace, electronicHamiltonian = self.myElectronicHamiltonian, MuxMuyMuzElectronicOperatorTuple = self.relaxationMuTuple, ExEyEzTimeFunctionTuple = probeMinusTuple, maxTime = self.maximumTime)

            #NOTE WE ARE USING THE MINUS PUMP BEAM TO ACHIEVE THE NEEDED COMPLEX CONJUGATE OPERATION ON EITHER THE LASER BEAM OR THE EMISSION
            bra = (probeMinusInteractor, pumpPlus_file_location) # pumpPlus_ProbeMinus_Wavefunction
            ket = pumpPlus_file_location # pumpPlus_Wavefunction
            cluster_helper_se = PP_Cluster_Helper(population_time_index = ii,
                                               muTuple = self.dipoleTuple,
                                               interfereing_pulse_tuple = probePlusTuple,
                                               bra = bra,
                                               ket = ket,
                                               signal_type = stimulatedEmission_signalType_string)

            listOfCalculations.append(cluster_helper_se)


#            signal_ESA
            if self.monomer == False:
                bra = pumpPlus_file_location # pumpPlus_Wavefunction
                ket = (probePlusInteractor,  pumpPlus_file_location) # pumpPlus_ProbePlus_Wavefunction
                cluster_helper_esa = PP_Cluster_Helper(population_time_index = ii,
                                                   muTuple = self.dipoleTuple,
                                                   interfereing_pulse_tuple = probePlusTuple,
                                                   bra = bra,
                                                   ket = ket,
                                                   signal_type = excitedStateAbsorption_signalType_string)
                listOfCalculations.append(cluster_helper_esa)


#            signal_GSB_1
            bra = zero_order_file_location# zeroOrderTimeWavefunction
            ket = (probePlusInteractor, pumpPlus_PumpMinus_file_location) # pumpPlus_PumpMinus_ProbePlus_Wavefunction
            cluster_helper_gsb1 = PP_Cluster_Helper(population_time_index = ii,
                                               muTuple = self.dipoleTuple,
                                               interfereing_pulse_tuple = probePlusTuple,
                                               bra = bra,
                                               ket = ket,
                                               signal_type = GroundStateBleach1_signalType_string)

            listOfCalculations.append(cluster_helper_gsb1)


#            signal_GSB_2
            bra = pumpPlus_PumpMinus_file_location  # pumpPlus_ProbeMinus_Wavefunction
            ket =  (probePlusInteractor, zero_order_file_location) # pumpPlus_Wavefunction
            cluster_helper_gsb2 = PP_Cluster_Helper(population_time_index = ii,
                                               muTuple = self.dipoleTuple,
                                               interfereing_pulse_tuple = probePlusTuple,
                                               bra = bra,
                                               ket = ket,
                                               signal_type = GroundStateBleach2_signalType_string)

            listOfCalculations.append(cluster_helper_gsb2)

            self.ppTimeSeries.append(self.currentT)

            self.currentT = self.currentT + self.dtPp

            probeEx = probeEx.pulseCopyJumpedForward(self.dtPp)
            probeEy = probeEy.pulseCopyJumpedForward(self.dtPp)
            probeEz = probeEz.pulseCopyJumpedForward(self.dtPp)

        if self.use_clusterpool:
            print "creating pool"
            myPool = ClusterPool.Pool()
            print "running subprocesses"
            listOfCalculations = myPool.map('calculate', listOfCalculations)
        else:
            print "creating pool"
            myPool = multiprocessing.Pool(NUMBER_PROCESSES)
            print "running subprocesses"
            listOfCalculations = myPool.map(lambda x: x.calculate(), listOfCalculations)

        for i, calc in enumerate(listOfCalculations):
            typeOfCalc = calc.signal_type
            T_index = calc.population_time_index
            sig = (2.0j * calc.signal).real
            if typeOfCalc == stimulatedEmission_signalType_string:
                self.timeSeriesSignalStimulatedEmission[T_index] = sig
                continue
            if typeOfCalc == GroundStateBleach1_signalType_string:
                self.timeSeriesSignalGroundStateBleachA[T_index] = sig
                continue
            if typeOfCalc == GroundStateBleach2_signalType_string:
                self.timeSeriesSignalGroundStateBleachB[T_index] = sig
                continue
            if typeOfCalc == excitedStateAbsorption_signalType_string:
                self.timeSeriesSignalExcitedStateAbsoprtion[T_index] = sig
                continue

        self.timeSeriesTotalSignal = self.timeSeriesSignalExcitedStateAbsoprtion + self.timeSeriesSignalGroundStateBleachA + self.timeSeriesSignalGroundStateBleachB + self.timeSeriesSignalStimulatedEmission
        self.ppTimeSeries = np.array(self.ppTimeSeries)
        self.ppTimeSeries_fs = self.mySpace.unitHandler.femtosecondsFromTime(self.ppTimeSeries)

        for filename in [zero_order_file_location, pumpPlus_file_location, pumpPlus_PumpMinus_file_location]:
            try:
                os.remove(filename)
            except:
                pass
        return self


class PP_Cluster_Helper(experimentBase.experiment):

    def load_object_from_file(self, file_location):
        object_file = open(file_location, "rb")
        loaded_object = pickle.load(object_file)
        object_file.close()
        return loaded_object


    def __init__(self, population_time_index, muTuple, interfereing_pulse_tuple, bra, ket, signal_type):
        "bra and ket are either filename strings or a tuple of an interactor object and a file string pointing to a wavefunction to be interacted with first"
        self.bra = bra
        self.ket = ket
        self.muTuple = muTuple
        self.interfereing_pulse_tuple  = interfereing_pulse_tuple

        self.population_time_index = population_time_index
        self.signal_type = signal_type

        self.calculated = False

    def calculate(self):
        if self.calculated:
            return self


        if isinstance(self.bra, tuple):
            interactor = self.bra[0]
            wavefunction_location = self.bra[1]
            wavefunction = self.load_object_from_file(wavefunction_location)
            calculated_bra = interactor.goFromTimeWavefunction(wavefunction)
        else:
            calculated_bra = self.load_object_from_file(self.bra)

        if isinstance(self.ket, tuple):
            interactor = self.ket[0]
            wavefunction_location = self.ket[1]
            wavefunction = self.load_object_from_file(wavefunction_location)
            calculated_ket = interactor.goFromTimeWavefunction(wavefunction)
        else:
            calculated_ket = self.load_object_from_file(self.ket)

        self.signal = calculated_ket.timeOverlapWithOtherBraEWFOfPolarizationAndOverlapWithElectricField(calculated_bra, self.muTuple, self.interfereing_pulse_tuple)
        return self


class WitnessExperiment(experimentBase.experiment):
    """
    This object facilitates the running of multiple pump probe experiments with different pulse widths on a supercomputer cluster
    """
    experiment_type_string = "Witness_Cluster_Pump_Probe_"
    def __init__(self, space,
                 electronicHamiltonian,
                 MuxMuyMuzElectronicOperatorTuple,
                 initialElectronicWavefunction,
                 maximumEvolutionTime,
                 numberOfTimePoints,
                 centerFrequency,
                 minimumPulseWidth,
                 maximumPulseWidth,
                 numberOfPulseWidthExperimentsToDo,
                 numberOfProcessorsToUse = None,
                 string_identity = "",
                 use_clusterpool = USE_CLUSTERPOOL):
        #store all the variables
        self.mySpace = space
        self.use_clusterpool = use_clusterpool
        self.string_identity = string_identity

        self.muTuple = MuxMuyMuzElectronicOperatorTuple

        self.intitialEWF = initialElectronicWavefunction
        self.myElectronicHamiltonian = electronicHamiltonian

        if minimumPulseWidth / self.mySpace.dt < 1.0:
            minimumPulseWidth = self.mySpace.dt
            warnings.warn("requested minimum pulse width smaller than time discretization!  changing minimum pulse width to dt")
        self.pulseWidthsToCalculate = np.linspace(minimumPulseWidth, maximumPulseWidth, numberOfPulseWidthExperimentsToDo)
        self.pulseWidthsToCalculate_femtoseconds = self.mySpace.unitHandler.femtosecondsFromTime(self.pulseWidthsToCalculate)
        self.pulseWidthsToCalculate_FWHM_femtoseconds = (2.0 *np.sqrt(2.0 * np.log(2.0))) * self.pulseWidthsToCalculate_femtoseconds

        pulseBeamTuples = []
        for width in self.pulseWidthsToCalculate:
            newPulseX = TimeFunction.GaussianCosinePulse(self.mySpace, centerOmega = centerFrequency, timeSpread = width)
            newPulseY = TimeFunction.GaussianCosinePulse(self.mySpace, centerOmega = centerFrequency, timeSpread = width)
            newPulseZ = TimeFunction.zeroTimeFunction(self.mySpace)
            pulseBeamTuples.append((newPulseX, newPulseY, newPulseZ))

        self.pumpBeamTuplesToCalculate = pulseBeamTuples

        self.TmaxPp = float(maximumEvolutionTime)
        self.dtPp = float(maximumEvolutionTime) / float(numberOfTimePoints)
        self.Npp = int(numberOfTimePoints)


        self.pulse_overlap_ending_time = 6.0 * maximumPulseWidth
        self.pulse_overlap_ending_index = self.pulse_overlap_ending_time / self.dtPp
        self.pulse_overlap_ending_index = int(self.pulse_overlap_ending_index)


        #now it's time to figure out how many processors to use

        if numberOfProcessorsToUse is None:
            #Give the computer some wiggle room to work whilst running the calculation
            self.numberOfProcessors = multiprocessing.cpu_count() - 1
        else:
            self.numberOfProcessors = numberOfProcessorsToUse

    def calculate(self):

        #first create all the objects:
        self.listOfPumpProbeExperiments = []

        self.W_list = []


        for ii, pumpTuple in enumerate(self.pumpBeamTuplesToCalculate):
            print "It is now", datetime.datetime.now()
            idNo =  self.string_identity + "_sigma= %s" % str(pumpTuple[0].sigma)
            newPP = Basic_Cluster(space = self.mySpace,
                                      electronicHamiltonian = self.myElectronicHamiltonian,
                                      MuxMuyMuzElectronicOperatorTuple = self.muTuple,
                                      initialElectronicWavefunction = self.intitialEWF,
                                      pumpBeamTuple = pumpTuple,
                                      maximumEvolutionTime = self.TmaxPp,
                                      numberOfTimePoints = self.Npp,
                                      use_clusterpool = self.use_clusterpool)

            newPP.calculate()
#            newPP.save(idNo)
            self.listOfPumpProbeExperiments.append(newPP)

#           Two-level clusterpool introduced too many errors, only one level for now
#        print "creating pool"
#        myPool = ClusterPool.Pool()
#        print "running subprocesses"
#        self.listOfPumpProbeExperiments = myPool.map('calculate', self.listOfPumpProbeExperiments)

        pp = self.listOfPumpProbeExperiments[1].timeSeriesTotalSignal
        n_ft_time_point = pp[self.pulse_overlap_ending_index:].shape[0]

        self.pp_oscillation_frequencies = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(n_ft_time_point, d = self.dtPp))

        self.pp_oscillation_frequencies_wavenumbers = self.mySpace.unitHandler.wavenumbersFromEnergyUnits(self.pp_oscillation_frequencies)

        total_signal_shape = (len(self.pumpBeamTuplesToCalculate), pp.shape[0])
        nonOverlap_signal_shape = (len(self.pumpBeamTuplesToCalculate), pp[self.pulse_overlap_ending_index:].shape[0])

        total_pp_signal = np.zeros(total_signal_shape, dtype = np.complex)

        total_ft_pp_signal = np.zeros(nonOverlap_signal_shape, dtype = np.complex)


        for i, pp in enumerate(self.listOfPumpProbeExperiments):
            total_pp_signal[i] = pp.timeSeriesTotalSignal
            pp_nonOverlap = pp.timeSeriesTotalSignal[self.pulse_overlap_ending_index:]
#            pp_nonOverlap = -pp_nonOverlap / np.max(np.abs(pp_nonOverlap))
            #average values
            pp_average = np.average(pp_nonOverlap)

            #signals to be used for W
            pp_forW = pp_nonOverlap - pp_average

            #FT values
            total_ft_pp_signal[i] = np.fft.fftshift(np.fft.fft(pp_forW))
            #W values
            W = scipy.integrate.simps(np.abs(pp_forW)**2) * self.dtPp

            self.W_list.append(W)


        self.total_pp_signal = total_pp_signal


        self.total_ft_pp_signal = total_ft_pp_signal


class WitnessExperiment_scanCenterFrequency(WitnessExperiment):
    """
    This object facilitates the running of multiple pump probe experiments with different pulse central frequencies on a supercomputer cluster
    """
    experiment_type_string = "Witness_Cluster_Pump_Probe_scan_central_frequency_"
    def __init__(self, space,
                 electronicHamiltonian,
                 MuxMuyMuzElectronicOperatorTuple,
                 initialElectronicWavefunction,
                 maximumEvolutionTime,
                 numberOfTimePoints,
                 centerFrequency_min,
                 centerFrequency_max,
                 pulseWidth,
                 numberOfPulseWidthExperimentsToDo,
                 numberOfProcessorsToUse = None,
                 string_identity = ""):
        #store all the variables
        self.mySpace = space

        self.string_identity = string_identity

        self.muTuple = MuxMuyMuzElectronicOperatorTuple

        self.intitialEWF = initialElectronicWavefunction
        self.myElectronicHamiltonian = electronicHamiltonian

        self.pulseWidth = pulseWidth
        self.pulseWidth_fs = self.mySpace.unitHandler.femtosecondsFromTime(self.pulseWidth)

        self.pulseCenterFrequenciesToCalculate = np.linspace(centerFrequency_min, centerFrequency_max, numberOfPulseWidthExperimentsToDo)
        self.pulseCenterFrequenciesToCalculate_wavenumbers = self.mySpace.unitHandler.wavenumbersFromEnergyUnits(self.pulseCenterFrequenciesToCalculate)

        pulseBeamTuples = []
        for wC in self.pulseCenterFrequenciesToCalculate:
            newPulseX = TimeFunction.GaussianCosinePulse(self.mySpace, centerOmega = wC, timeSpread = self.pulseWidth )
            newPulseY = TimeFunction.GaussianCosinePulse(self.mySpace, centerOmega = wC, timeSpread = self.pulseWidth )
            newPulseZ = TimeFunction.zeroTimeFunction(self.mySpace)
            pulseBeamTuples.append((newPulseX, newPulseY, newPulseZ))

        self.pumpBeamTuplesToCalculate = pulseBeamTuples

        self.TmaxPp = float(maximumEvolutionTime)
        self.dtPp = float(maximumEvolutionTime) / float(numberOfTimePoints)
        self.Npp = int(numberOfTimePoints)


        self.pulse_overlap_ending_time = 6.0 * self.pulseWidth
        self.pulse_overlap_ending_index = self.pulse_overlap_ending_time / self.dtPp
        self.pulse_overlap_ending_index = int(self.pulse_overlap_ending_index)


        #now it's time to figure out how many processors to use

        if numberOfProcessorsToUse is None:
            #Give the computer some wiggle room to work whilst running the calculation
            self.numberOfProcessors = multiprocessing.cpu_count() - 1
        else:
            self.numberOfProcessors = numberOfProcessorsToUse




if __name__ == "__main__":
    pass
