# -*- coding: utf-8 -*-
"""
Created on Fri Jan 04 13:23:27 2013

@author: Joey
"""
import copy
import time
import os
import math
import random
import hickle
import cPickle as pickle
import datetime
import warnings

import numpy as np
import scipy
import scipy.integrate

import scipy.fftpack
import multiprocessing
import matplotlib as mpl
import matplotlib.pyplot as plt



try:
    from matplotlib import animation
except:
    animation = object()

import spectroscopy.TimeElectronicWavefunction as TimeElectronicWavefunction
import spectroscopy.TimeFunction as TimeFunction
import spectroscopy.TimePerturbation as TimePerturbation

import experimentBase

USE_CLUSTERPOOL = False
NUMBER_PROCESSES = 8
try:
    import ClusterPool.ClusterPool as ClusterPool
    import ClusterPool.Dispatcher as Dispatcher
except ImportError:
    warnings.warn("ClusterPool Module Not Found, Witness experiments will not work")


class base_TG(experimentBase.experiment):
    "setup code for all Transient Grating style experiments"
    def __init__(self, space,
                 electronicHamiltonian,
                 MuxMuyMuzElectronicOperatorTuple,
                 initialElectronicWavefunction,
                 pumpBeamTuple,
                 maximumPopulationTime,
                 numberOfPopulationTimePoints,
                 maximumProbeTime,
                 numberOfProbeTimePoints,
                 opticalGap = 0.0,
                 IDstring = "",
                 numberOfProcessors = None,
                 use_clusterpool = USE_CLUSTERPOOL):
        #store all the variables
        self.mySpace = space
        self.use_clusterpool = use_clusterpool
        self.MuX, self.MuY, self.MuZ = MuxMuyMuzElectronicOperatorTuple
        self.dipoleTuple = MuxMuyMuzElectronicOperatorTuple
        self.pumpEx, self.pumpEy, self.pumpEz = pumpBeamTuple

        self.intitialEWF = initialElectronicWavefunction
        self.myElectronicHamiltonian = electronicHamiltonian
        self.myFreePropagator = electronicHamiltonian.myPropagator()

        #We are given a target maximum waiting and a specific number of time points to take.
        #We then have to scale that to the closest multiple of mySpace.dt to make the propagation work properly

        self.numberOfPopulationTimePoints = int(numberOfPopulationTimePoints)
        targetMaxPopulationTime = float(maximumPopulationTime)
        targetPopulationDT = targetMaxPopulationTime  / float(numberOfPopulationTimePoints)

        self.timeStepsPerPopulationStep = int(np.round(targetPopulationDT / self.mySpace.dt, 0))
        #What if it's zero!?  We can't have that
        if self.timeStepsPerPopulationStep == 0:
            self.timeStepsPerPopulationStep = 1

        self.populationDT = self.timeStepsPerPopulationStep * self.mySpace.dt

        self.maxPopulationTime =  self.populationDT * (self.numberOfPopulationTimePoints - 1)

        self.populationTimes = []


        self.minProbeTime = 0.0
        self.ProbeTimePoints = numberOfProbeTimePoints
#
        self.probeDT = self.mySpace.dt
        self.maxProbeTime = self.minProbeTime + self.probeDT * (self.ProbeTimePoints )
        self.probeTimes = np.arange(self.minProbeTime, self.maxProbeTime, self.probeDT)
#
        self.firstProbeEx = copy.deepcopy(self.pumpEx)
        self.firstProbeEy = copy.deepcopy(self.pumpEy)
        self.firstProbeEz = copy.deepcopy(self.pumpEz)

        self.probePower = self.firstProbeEx.totalPower() + self.firstProbeEy.totalPower() + self.firstProbeEz.totalPower()

        self.calculated = False
        self.IDstring = IDstring

        self.opticalGap = opticalGap

        #maximum time the experiment will go on in any case
        #the padding for all 3 (really 2) pulses
        total_pillow_padding = 4.0 * 3.0 * 2.0 * max(map(lambda x: x.timePillow, pumpBeamTuple))
        #plus the maxmimum probe spacing
        max_time_to_propagate_probe = self.maxProbeTime
        #plus the population time
        self.maximumTime = self.maxPopulationTime + total_pillow_padding + max_time_to_propagate_probe

        self.maximumTimeSteps = int(self.maximumTime / self.mySpace.dt)
        #this line will break the code for some unknown reason


        #TODO: find out WHAT THE HECK IS UP HERE with needing a signal divisible by four.  probably has something to do with the local oscillator

        #this line makes the code work
        self.maximumTimeSteps = self.maximumTimeSteps + (4 - self.maximumTimeSteps % 4) - 1


        self.maximumTime = self.maximumTimeSteps * (self.mySpace.dt)

        self.mySpace.unitHandler.femtosecondsFromTime(self.maximumTime)

        self.pulse_overlap_ending_time = 6.0 * max(self.firstProbeEx.sigma, self.firstProbeEy.sigma, self.firstProbeEz.sigma)
        self.pulse_overlap_ending_index = self.pulse_overlap_ending_time / self.populationDT
        self.pulse_overlap_ending_index = int(self.pulse_overlap_ending_index)


        if numberOfProcessors is None:
            self.numberOfProcessors = multiprocessing.cpu_count() - 2
        else:
            self.numberOfProcessors = numberOfProcessors

        self.totalSignalShapeTuple = (self.numberOfPopulationTimePoints, self.ProbeTimePoints)

        self.excitationMuTuple = (self.dipoleTuple[0].excitationOperator(), self.dipoleTuple[1].excitationOperator(), self.dipoleTuple[2].excitationOperator())
        self.relaxationMuTuple = (self.dipoleTuple[0].relaxationOperator(), self.dipoleTuple[1].relaxationOperator(), self.dipoleTuple[2].relaxationOperator())

        self.monomer = (self.mySpace.electronicDimensionality == 2)


    def calculate(self):
        Exception("USE NON-ABSTRACT CLASS")


class PopulationTimeScan(base_TG):
    "Transient Grating for multiple population evolution times.  RAM efficient"
    experiment_type_string = "Basic_Transient_Grating_"

    def calculate(self):
        overallStartTime = time.time()
        if self.calculated:
            print "You've already done the calculation, look at the results instead!"
            return self

        startTime = time.time()
        zeroOrderTimeWavefunction = TimeElectronicWavefunction.timeElectronicWavefunction(self.mySpace)
        zeroOrderTimeWavefunction.applyOperatorsNTimesOnInitialWavefunction([self.myFreePropagator], N = self.maximumTimeSteps, initialWF = self.intitialEWF)
        print "zero order done"
        print "elapsed Time: ",  time.time() - startTime

        pulse1xPLUS = self.firstProbeEx.plusFunction()
        pulse1yPLUS = self.firstProbeEy.plusFunction()
        pulse1zPLUS = self.firstProbeEz.plusFunction()



        pulse1PLUStuple = (pulse1xPLUS, pulse1yPLUS, pulse1zPLUS)

        pulse1xMINUS = self.firstProbeEx.minusFunction()
        pulse1yMINUS = self.firstProbeEy.minusFunction()
        pulse1zMINUS = self.firstProbeEz.minusFunction()

        pulse1MINUStuple = (pulse1xMINUS, pulse1yMINUS, pulse1zMINUS)

        k1PlusInteractor = TimePerturbation.electricDipoleOneInteraction(space = self.mySpace, electronicHamiltonian = self.myElectronicHamiltonian, MuxMuyMuzElectronicOperatorTuple = self.excitationMuTuple, ExEyEzTimeFunctionTuple = pulse1PLUStuple, maxTime = self.maximumTime)

        k1MinusInteractor = TimePerturbation.electricDipoleOneInteraction(space = self.mySpace, electronicHamiltonian = self.myElectronicHamiltonian, MuxMuyMuzElectronicOperatorTuple = self.relaxationMuTuple, ExEyEzTimeFunctionTuple = pulse1MINUStuple, maxTime = self.maximumTime)

        startTime = time.time()
        k1PlusWavefunction = k1PlusInteractor.goFromTimeWavefunction(zeroOrderTimeWavefunction)
        print "first order done"
        print "elapsed Time: ",  time.time() - startTime

        startTime = time.time()
        k2Plus_k1MinusWavefunction = k1MinusInteractor.goFromTimeWavefunction(k1PlusWavefunction)

        print "second order done"
        print "elapsed Time: ",  time.time() - startTime


        probeEx = copy.deepcopy(self.firstProbeEx)
        probeEy = copy.deepcopy(self.firstProbeEy)
        probeEz = copy.deepcopy(self.firstProbeEz)



        #important for filter length
        lowest_frequency_from_H = self.myElectronicHamiltonian.omega_low


        signal_size = (self.numberOfPopulationTimePoints , k1PlusWavefunction.length())


        signal_lo_size = signal_size
        self.totalSignal_lo_frequency = np.zeros(signal_lo_size, dtype = np.complex)

        #FREQUENCY SIGNALS
        self.totalSignal_xFrequency = np.zeros(signal_size, dtype = np.complex)
        self.totalSignal_yFrequency = np.zeros(signal_size, dtype = np.complex)
        self.totalSignal_zFrequency = np.zeros(signal_size, dtype = np.complex)


        current_T = 0.0

        tot = self.numberOfPopulationTimePoints
        max_T_index = tot - 1
        for population_time_index in range(tot):
            self.populationTimes.append(current_T)
            startTime = time.time()
            i = population_time_index

            pulse3tuple = (probeEx, probeEy, probeEz)
            pulse3PLUStuple = (probeEx.plusFunction(), probeEy.plusFunction(), probeEz.plusFunction())
            pulse3MINUStuple = (probeEx.minusFunction(), probeEy.minusFunction(), probeEz.minusFunction())

            k3PlusInteractor = TimePerturbation.electricDipoleOneInteraction(space = self.mySpace, electronicHamiltonian = self.myElectronicHamiltonian, MuxMuyMuzElectronicOperatorTuple = self.excitationMuTuple, ExEyEzTimeFunctionTuple = pulse3PLUStuple, maxTime = self.maximumTime)
            k3MinusInteractor = TimePerturbation.electricDipoleOneInteraction(space = self.mySpace, electronicHamiltonian = self.myElectronicHamiltonian, MuxMuyMuzElectronicOperatorTuple = self.relaxationMuTuple, ExEyEzTimeFunctionTuple = pulse3MINUStuple, maxTime = self.maximumTime)


            k3PLUS_Wavefunction = interactionHelper(k3PlusInteractor, zeroOrderTimeWavefunction)
            k3PLUS_Wavefunction = k3PLUS_Wavefunction.calculate()

            k1PLUS_k3MINUS_Wavefunction = interactionHelper(k3MinusInteractor, k1PlusWavefunction)
            k1PLUS_k3MINUS_Wavefunction = k1PLUS_k3MINUS_Wavefunction.calculate()

            if self.monomer == False:
                k1PLUS_k3PLUS_Wavefunction = interactionHelper(k3PlusInteractor, k1PlusWavefunction)
                k1PLUS_k3PLUS_Wavefunction = k1PLUS_k3PLUS_Wavefunction.calculate()
            else:
                k1PLUS_k3PLUS_Wavefunction = None

            k1PLUS_k2MINUS_k3PLUS_Wavefunction = interactionHelper(k3PlusInteractor, k2Plus_k1MinusWavefunction)
            k1PLUS_k2MINUS_k3PLUS_Wavefunction = k1PLUS_k2MINUS_k3PLUS_Wavefunction.calculate()



#            k1 - k2 + k3 // none
            signal1Helper = overlapHelper(zeroOrderTimeWavefunction, k1PLUS_k2MINUS_k3PLUS_Wavefunction, self.dipoleTuple, pulse3PLUStuple, lowest_frequency_from_H, population_time_index, max_T_index, n = self.timeStepsPerPopulationStep)
            signal1Helper = signal1Helper.calculate()
            obj = signal1Helper
            self.totalSignal_xFrequency[i] = self.totalSignal_xFrequency[i] + obj.xfrequency_spectrum
            self.totalSignal_yFrequency[i] = self.totalSignal_xFrequency[i] + obj.yfrequency_spectrum
            self.totalSignal_zFrequency[i] = self.totalSignal_xFrequency[i] + obj.zfrequency_spectrum
            self.totalSignal_lo_frequency[i] = self.totalSignal_lo_frequency[i] + obj.local_oscillator_signal

            signal1Helper = None
            k1PLUS_k2MINUS_k3PLUS_Wavefunction = None

#            k3 // k2-k1
            signal2Helper = overlapHelper(k2Plus_k1MinusWavefunction, k3PLUS_Wavefunction, self.dipoleTuple, pulse3PLUStuple, lowest_frequency_from_H, population_time_index, max_T_index, n = self.timeStepsPerPopulationStep)
            signal2Helper = signal2Helper.calculate()

            obj = signal2Helper
            self.totalSignal_xFrequency[i] = self.totalSignal_xFrequency[i] + obj.xfrequency_spectrum
            self.totalSignal_yFrequency[i] = self.totalSignal_xFrequency[i] + obj.yfrequency_spectrum
            self.totalSignal_zFrequency[i] = self.totalSignal_xFrequency[i] + obj.zfrequency_spectrum
            self.totalSignal_lo_frequency[i] = self.totalSignal_lo_frequency[i] + obj.local_oscillator_signal

            signal2Helper = None
            k3PLUS_Wavefunction = None

#            k1 // k2 - k3
            signal3Helper = overlapHelper(k1PLUS_k3MINUS_Wavefunction, k1PlusWavefunction, self.dipoleTuple, pulse3PLUStuple, lowest_frequency_from_H, population_time_index, max_T_index, n = self.timeStepsPerPopulationStep)
            signal3Helper = signal3Helper.calculate()

            obj = signal3Helper
            self.totalSignal_xFrequency[i] = self.totalSignal_xFrequency[i] + obj.xfrequency_spectrum
            self.totalSignal_yFrequency[i] = self.totalSignal_xFrequency[i] + obj.yfrequency_spectrum
            self.totalSignal_zFrequency[i] = self.totalSignal_xFrequency[i] + obj.zfrequency_spectrum
            self.totalSignal_lo_frequency[i] = self.totalSignal_lo_frequency[i] + obj.local_oscillator_signal

            signal3Helper = None
            k1PLUS_k3MINUS_Wavefunction = None
#            k1 + k3 // k2
            if self.monomer == False:
                signal5Helper = overlapHelper(k1PlusWavefunction, k1PLUS_k3PLUS_Wavefunction, self.dipoleTuple, pulse3PLUStuple,  lowest_frequency_from_H, population_time_index, max_T_index, n = self.timeStepsPerPopulationStep)
                signal5Helper = signal5Helper.calculate()

                obj = signal5Helper
                self.totalSignal_xFrequency[i] = self.totalSignal_xFrequency[i] + obj.xfrequency_spectrum
                self.totalSignal_yFrequency[i] = self.totalSignal_xFrequency[i] + obj.yfrequency_spectrum
                self.totalSignal_zFrequency[i] = self.totalSignal_xFrequency[i] + obj.zfrequency_spectrum
                self.totalSignal_lo_frequency[i] = self.totalSignal_lo_frequency[i] + obj.local_oscillator_signal
                signal5Helper = None
                k1PLUS_k3PLUS_Wavefunction = None
            else:
                pass

            probeEx = probeEx.pulseCopyJumpedForward(self.populationDT)
            probeEy = probeEy.pulseCopyJumpedForward(self.populationDT)
            probeEz = probeEz.pulseCopyJumpedForward(self.populationDT)

            current_T = current_T + self.populationDT

            print "elapsed Time: ",  time.time() - startTime


        self.populationTimes = np.array(self.populationTimes)

        self.frequency_values = obj.frequency_values
        self.frequency_values_wavenumbers = self.mySpace.unitHandler.wavenumbersFromEnergyUnits(self.frequency_values)


        self.populationTimes = np.array(self.populationTimes)
        self.populationTimes_fs = self.mySpace.unitHandler.femtosecondsFromTime(self.populationTimes)

        #No imaginary unit or factor of 2 is needed because the auxiliary object does this calculation
        self.pumpProbe_signal = -np.real(scipy.integrate.simps(self.totalSignal_lo_frequency, axis = 1) * (self.frequency_values[1] - self.frequency_values[0])) / (2.0 * np.pi)

        self.totalSignal_frequency_power = np.abs(self.totalSignal_xFrequency)**2 + np.abs(self.totalSignal_yFrequency)**2 + np.abs(self.totalSignal_zFrequency)**2

        self.totalSignal_frequency_abs = np.sqrt(self.totalSignal_frequency_power)

        self.calculationTime = time.time() - overallStartTime


        return self

class PopulationTimeScan_cluster(base_TG):
    "Transient Grating for multiple population evolution times.  RAM efficient"
    experiment_type_string = "Cluster_Transient_Grating_"

    def calculate(self):
        overallStartTime = time.time()
        if self.calculated:
            print "You've already done the calculation, look at the results instead!"
            return self

        startTime = time.time()
        zeroOrderTimeWavefunction = TimeElectronicWavefunction.timeElectronicWavefunction(self.mySpace)
        zeroOrderTimeWavefunction.applyOperatorsNTimesOnInitialWavefunction([self.myFreePropagator], N = self.maximumTimeSteps, initialWF = self.intitialEWF)

        print "zero order done"
        print "elapsed Time: ",  time.time() - startTime
        zero_order_file_location = Dispatcher.Dispatcher.save_shared_object_return_filename(zeroOrderTimeWavefunction, "zero_order_ewf")
        print "zero order located at: ", zero_order_file_location
        saved_shared_filenames = [zero_order_file_location]

        pulse1xPLUS = self.firstProbeEx.plusFunction()
        pulse1yPLUS = self.firstProbeEy.plusFunction()
        pulse1zPLUS = self.firstProbeEz.plusFunction()
        pulse1PLUStuple = (pulse1xPLUS, pulse1yPLUS, pulse1zPLUS)

        pulse1xMINUS = self.firstProbeEx.minusFunction()
        pulse1yMINUS = self.firstProbeEy.minusFunction()
        pulse1zMINUS = self.firstProbeEz.minusFunction()
        pulse1MINUStuple = (pulse1xMINUS, pulse1yMINUS, pulse1zMINUS)

        k1PlusInteractor = TimePerturbation.electricDipoleOneInteraction(space = self.mySpace, electronicHamiltonian = self.myElectronicHamiltonian,  MuxMuyMuzElectronicOperatorTuple = self.excitationMuTuple, ExEyEzTimeFunctionTuple = pulse1PLUStuple,  maxTime = self.maximumTime)

        k1MinusInteractor = TimePerturbation.electricDipoleOneInteraction(space = self.mySpace,  electronicHamiltonian = self.myElectronicHamiltonian, MuxMuyMuzElectronicOperatorTuple = self.relaxationMuTuple, ExEyEzTimeFunctionTuple = pulse1MINUStuple, maxTime = self.maximumTime)

        startTime = time.time()
        k1PlusWavefunction = k1PlusInteractor.goFromTimeWavefunction(zeroOrderTimeWavefunction)
        zeroOrderTimeWavefunction = None
        print "first order done"
        print "elapsed Time: ",  time.time() - startTime




        oneAndTwoPlus_file_location = Dispatcher.Dispatcher.save_shared_object_return_filename(k1PlusWavefunction, "1and2Plus_ewf" )

        saved_shared_filenames.append(oneAndTwoPlus_file_location)

        startTime = time.time()
        k2Plus_k1MinusWavefunction = k1MinusInteractor.goFromTimeWavefunction(k1PlusWavefunction)
        TwoPlus_OneMinus_file_location = Dispatcher.Dispatcher.save_shared_object_return_filename(k2Plus_k1MinusWavefunction, "2Plus_1Minus_ewf")

        saved_shared_filenames.append(TwoPlus_OneMinus_file_location)

        signal_size = (self.numberOfPopulationTimePoints , k1PlusWavefunction.length())

        k1PlusWavefunction = None
        k2Plus_k1MinusWavefunction = None

        print "second order done"
        print "elapsed Time: ",  time.time() - startTime


        probeEx = copy.deepcopy(self.firstProbeEx)
        probeEy = copy.deepcopy(self.firstProbeEy)
        probeEz = copy.deepcopy(self.firstProbeEz)

        #important for filter length
        lowest_frequency_from_H = self.myElectronicHamiltonian.omega_low


        signal_lo_size = signal_size
        self.totalSignal_lo_frequency = np.zeros(signal_lo_size, dtype = np.complex)

        #FREQUENCY SIGNALS
        self.totalSignal_xFrequency = np.zeros(signal_size, dtype = np.complex)
        self.totalSignal_yFrequency = np.zeros(signal_size, dtype = np.complex)
        self.totalSignal_zFrequency = np.zeros(signal_size, dtype = np.complex)



        my_tg_cluster_helpers = []

        current_T = 0.0
        total_T_points = self.numberOfPopulationTimePoints
        max_T_index = total_T_points - 1
        for population_time_index in range(total_T_points):
            self.populationTimes.append(current_T)
            startTime = time.time()
            i = population_time_index

            pulse3tuple = (probeEx, probeEy, probeEz)
            pulse3PLUStuple = (probeEx.plusFunction(), probeEy.plusFunction(), probeEz.plusFunction())
            pulse3MINUStuple = (probeEx.minusFunction(), probeEy.minusFunction(), probeEz.minusFunction())

            k3PlusInteractor = TimePerturbation.electricDipoleOneInteraction(space = self.mySpace, electronicHamiltonian = self.myElectronicHamiltonian, MuxMuyMuzElectronicOperatorTuple = self.excitationMuTuple, ExEyEzTimeFunctionTuple = pulse3PLUStuple, maxTime = self.maximumTime)

            k3MinusInteractor = TimePerturbation.electricDipoleOneInteraction(space = self.mySpace, electronicHamiltonian = self.myElectronicHamiltonian, MuxMuyMuzElectronicOperatorTuple = self.relaxationMuTuple, ExEyEzTimeFunctionTuple = pulse3MINUStuple, maxTime = self.maximumTime)


            pulse3PlusTuple = (probeEx.plusFunction(), probeEy.plusFunction(), probeEz.plusFunction())

            #cluster_TG_helper give it:
            #T_index
            #ket
            #bra
            #muTuple
            #pulse3Tuple

#            zero_order_file_location
#            oneAndTwoPlus_file_location
#            TwoPlus_OneMinus_file_location
#            k1 - k2 + k3 // none
            bra = zero_order_file_location
            ket = (k3PlusInteractor, TwoPlus_OneMinus_file_location)
            new_cluster_helper = TG_Cluster_Helper(population_time_index, self.dipoleTuple, pulse3PlusTuple, bra, ket, lowest_frequency_from_H, max_T_index = max_T_index, n = self.timeStepsPerPopulationStep)
            my_tg_cluster_helpers.append(new_cluster_helper)
#            k3 // k2-k1
            bra = TwoPlus_OneMinus_file_location
            ket = (k3PlusInteractor, zero_order_file_location)
            new_cluster_helper = TG_Cluster_Helper(population_time_index, self.dipoleTuple, pulse3PlusTuple, bra, ket, lowest_frequency_from_H, max_T_index = max_T_index, n = self.timeStepsPerPopulationStep)
            my_tg_cluster_helpers.append(new_cluster_helper)
#            k1 // k2 - k3
            bra = (k3MinusInteractor, oneAndTwoPlus_file_location)
            ket = oneAndTwoPlus_file_location
            new_cluster_helper = TG_Cluster_Helper(population_time_index, self.dipoleTuple, pulse3PlusTuple, bra, ket, lowest_frequency_from_H, max_T_index = max_T_index, n = self.timeStepsPerPopulationStep)
            my_tg_cluster_helpers.append(new_cluster_helper)
            if self.monomer == False:
                #k1 + k3 // k2
                bra = oneAndTwoPlus_file_location
                ket = (k3PlusInteractor, oneAndTwoPlus_file_location)
                new_cluster_helper = TG_Cluster_Helper(population_time_index, self.dipoleTuple, pulse3PlusTuple, bra, ket, lowest_frequency_from_H, max_T_index = max_T_index, n = self.timeStepsPerPopulationStep)
                my_tg_cluster_helpers.append(new_cluster_helper)
            else:
                pass



            probeEx = probeEx.pulseCopyJumpedForward(self.populationDT)
            probeEy = probeEy.pulseCopyJumpedForward(self.populationDT)
            probeEz = probeEz.pulseCopyJumpedForward(self.populationDT)
            current_T = current_T + self.populationDT

        if self.use_clusterpool:
            print "creating pool"
            myPool = ClusterPool.Pool()
            print "running subprocesses"
            calculated_tg_cluster_helpers = myPool.map('calculate', my_tg_cluster_helpers)
        else:
            print "creating pool"
            myPool = multiprocessing.Pool(NUMBER_PROCESSES)
            print "running subprocesses"
            calculated_tg_cluster_helpers = myPool.map(lambda x: x.calculate(), my_tg_cluster_helpers)

        for calculated_tg_cluster_helper in calculated_tg_cluster_helpers:
            i = calculated_tg_cluster_helper.population_time_index
            obj = calculated_tg_cluster_helper
            self.totalSignal_xFrequency[i] = self.totalSignal_xFrequency[i] + obj.xfrequency_spectrum
            self.totalSignal_yFrequency[i] = self.totalSignal_yFrequency[i] + obj.yfrequency_spectrum
            self.totalSignal_zFrequency[i] = self.totalSignal_zFrequency[i] + obj.zfrequency_spectrum
            self.totalSignal_lo_frequency[i] = self.totalSignal_lo_frequency[i] + obj.local_oscillator_signal


        self.frequency_values = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(obj.local_oscillator_signal.shape[0], d = self.mySpace.dt))
        self.frequency_values_wavenumbers = self.mySpace.unitHandler.wavenumbersFromEnergyUnits(self.frequency_values)

        self.populationTimes = np.array(self.populationTimes)
        self.populationTimes_fs = self.mySpace.unitHandler.femtosecondsFromTime(self.populationTimes)

        #No 2.0 or 1.0j is needed because the helper function does those multiplications
        self.pumpProbe_signal = -np.real( scipy.integrate.simps(self.totalSignal_lo_frequency, axis = 1) * (self.frequency_values[1] - self.frequency_values[0])) / (2.0 * np.pi)

        self.totalSignal_frequency_power = np.abs(self.totalSignal_xFrequency)**2 + np.abs(self.totalSignal_yFrequency)**2 + np.abs(self.totalSignal_zFrequency)**2

        self.totalSignal_frequency_abs = np.sqrt(self.totalSignal_frequency_power)

        self.calculationTime = time.time() - overallStartTime

        #delete shared files
        for filename in saved_shared_filenames:
            os.remove(filename)
        return self

class TG_Cluster_Helper(experimentBase.experiment):
    useFilter = True
    base_frequency_multiple = 2.0

    def load_object_from_file(self, file_location):
        object_file = open(file_location, "rb")
        loaded_object = pickle.load(object_file)
        object_file.close()
        return loaded_object


    def __init__(self, population_time_index, muTuple, pulse3PlusTuple, bra, ket, lowest_frequency, max_T_index, n):
        "bra and ket are either filename strings or a tuple of an interactor object and a file string pointing to a wavefunction to be interacted with first"
        self.bra = bra
        self.ket = ket
        self.muTuple = muTuple
        self.pulse3PlusTuple  = pulse3PlusTuple

        self.lowest_frequency = lowest_frequency

        self.population_time_index = population_time_index
        self.max_T_index = max_T_index

        self.n = n

        self.calculated = False

    def calculate(self):
        if self.calculated:
            return self
        dt = self.muTuple[0].mySpace.dt


        if isinstance(self.bra, tuple):
            interactor = self.bra[0]
            wavefunction_location = self.bra[1]
            wavefunction = self.load_object_from_file(wavefunction_location)
            calculated_bra = interactor.goFromTimeWavefunction(wavefunction)
        else:
            calculated_bra = self.load_object_from_file(self.bra)

        #print "calculated bra object: ", calculated_bra

        if isinstance(self.ket, tuple):
            interactor = self.ket[0]
            wavefunction_location = self.ket[1]
            wavefunction = self.load_object_from_file(wavefunction_location)
            calculated_ket = interactor.goFromTimeWavefunction(wavefunction)
        else:
            calculated_ket = self.load_object_from_file(self.ket)
        #print "calculated ket object: ", calculated_ket

        signalTuple = calculated_ket.timeOverlapWithOtherBraEWFOfPolarization(calculated_bra, self.muTuple)

        xSignal = 1.0j * signalTuple[0]
        ySignal = 1.0j * signalTuple[1]
        zSignal = 1.0j * signalTuple[2]

        #clear memory
        self.bra = None
        self.ket = None
        self.muTuple = None

        n_time_point = xSignal.shape[0]
        if TG_Cluster_Helper.useFilter:
            filter_length_time = TG_Cluster_Helper.base_frequency_multiple * (2.0 * np.pi / self.lowest_frequency )
            filter_length_steps = int(filter_length_time / dt)

            xf = n_time_point - (self.max_T_index - self.population_time_index) * self.n - 1
            x0 = xf - filter_length_steps
    #        Here we define a third order polynomial that follows the following equations:
    #        f(x0) = 1
    #        f(xf) = 0
    #        f'(x0) = 0
    #        f'(xf) = 0
            denominator = (x0 - xf)**3
            a = -2.0 / denominator
            b = 3.0 * (x0 + xf) / denominator
            c = -6.0 * x0 * xf / denominator
            d = (3.0 * x0 * xf**2 - xf**3) / denominator

            filterVals = np.ones(n_time_point, dtype = np.complex)
            for i in range(x0, xf+1):
                filterVals[i] = a * (i**3) + b * (i**2) + c * i + d
            for i in range(xf+1, n_time_point):
                filterVals[i] = 0.0
        else:
            filterVals = 1.0

        ft_constant = (dt) * n_time_point
        xfrequency_spectrum = np.fft.fftshift(np.fft.ifft(xSignal * filterVals)) * ft_constant
        yfrequency_spectrum = np.fft.fftshift(np.fft.ifft(ySignal * filterVals)) * ft_constant
        zfrequency_spectrum = np.fft.fftshift(np.fft.ifft(zSignal * filterVals)) * ft_constant


        self.xfrequency_spectrum = xfrequency_spectrum
        self.yfrequency_spectrum = yfrequency_spectrum
        self.zfrequency_spectrum = zfrequency_spectrum

        self.local_oscillator_signal = np.array([])

        self.frequency_values = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(n_time_point, d = dt))


        try:
            convoluterX = self.pulse3PlusTuple[0].myFourierTransformedFunction(self.frequency_values)

        except Exception:
            convoluterX = 0

        try:
            convoluterY = self.pulse3PlusTuple[1].myFourierTransformedFunction(self.frequency_values)
        except Exception:
            convoluterY = 0

        try:
            convoluterZ =  self.pulse3PlusTuple[2].myFourierTransformedFunction(self.frequency_values)
        except Exception:
            convoluterZ = 0

        local_oscillator_signal = convoluterX*np.conj(xfrequency_spectrum) + convoluterY*np.conj(yfrequency_spectrum) + convoluterZ*np.conj(zfrequency_spectrum)
        self.local_oscillator_signal = 2.0 * local_oscillator_signal


        self.calculated = True
        #clear memory
        self.pulse3PlusTuple  = None

        return self



class WitnessExperiment(experimentBase.experiment):
    experiment_type_string = "Witness_Cluster_Transient_Grating_"

    def __init__(self, space,
                 electronicHamiltonian,
                 MuxMuyMuzElectronicOperatorTuple,
                 initialElectronicWavefunction,
                 maximumEvolutionTime,
                 numberOfTimePoints,
                 centerFrequency,
                 minimumPulseWidth,
                 maximumPulseWidth,
                 maximumProbeTime,
                 numberOfProbeTimePoints,
                 numberOfPulseWidthExperimentsToDo,
                 numberOfProcessors = None,
                 id_string = "",
                 string_identity = ""):
        #store all the variables
        self.mySpace = space

        self.string_identity = string_identity

        self.muTuple = MuxMuyMuzElectronicOperatorTuple

        self.intitialEWF = initialElectronicWavefunction
        self.myElectronicHamiltonian = electronicHamiltonian

        self.pulseWidthsToCalculate = np.linspace(minimumPulseWidth, maximumPulseWidth, numberOfPulseWidthExperimentsToDo)
        self.pulseWidthsToCalculate_femtoseconds = self.mySpace.unitHandler.femtosecondsFromTime(self.pulseWidthsToCalculate)
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


        self.maximumProbeTime = maximumProbeTime
        self.numberOfProbeTimePoints = numberOfProbeTimePoints


        self.pulse_overlap_ending_time = 6.0 * maximumPulseWidth
        self.pulse_overlap_ending_index = self.pulse_overlap_ending_time / self.dtPp
        self.pulse_overlap_ending_index = int(self.pulse_overlap_ending_index)

        #now it's time to figure out how many processors to use

        if numberOfProcessors is None:
            #Give the computer some wiggle room to work whilst running the calculation
            self.numberOfProcessors = multiprocessing.cpu_count() - 1
        else:
            self.numberOfProcessors = numberOfProcessors

    def calculate(self):

        #first create all the objects:
        self.listOfTGExperiments = []

        self.W_list = []
        self.W_fascimile_n1_list = []
        self.W_fascimile_n2_list = []

        pp_signals = []
        pp_fascimile_n1_signals = []
        pp_fascimile_n2_signals = []

        for ii, pumpTuple in enumerate(self.pumpBeamTuplesToCalculate):
            print "It is now", datetime.datetime.now()
            idNo = self.string_identity + "_sigma= %s" % str(pumpTuple[0].sigma)
            newTG = PopulationTimeScan_cluster(space = self.mySpace,
                                      electronicHamiltonian = self.myElectronicHamiltonian,
                                      MuxMuyMuzElectronicOperatorTuple = self.muTuple,
                                      initialElectronicWavefunction = self.intitialEWF,
                                      pumpBeamTuple = pumpTuple,
                                      maximumPopulationTime = self.TmaxPp,
                                      numberOfPopulationTimePoints = self.Npp,
                                      maximumProbeTime = self.maximumProbeTime,
                                      numberOfProbeTimePoints = self.numberOfProbeTimePoints,
                                      IDstring = idNo,
                                      numberOfProcessors = self.numberOfProcessors)


            print idNo, "calculating...."
            newTG = newTG.calculate()

            newTG.save(idNo)

            #this saves a tremendous amount of memory, but makes data lookup harder
#            self.listOfTGExperiments.append(newTG)
            self.listOfTGExperiments.append(None)

            dOmega = newTG.frequency_values[1] - newTG.frequency_values[0]

            pp = newTG.pumpProbe_signal
            pp_fascimile_n1 = scipy.integrate.simps(newTG.totalSignal_frequency_abs, dx = dOmega, axis = 1)
            pp_fascimile_n2 = scipy.integrate.simps(newTG.totalSignal_frequency_abs**2, dx = dOmega, axis = 1)

            pp_signals.append(pp)
            pp_fascimile_n1_signals.append(pp_fascimile_n1)
            pp_fascimile_n2_signals.append(pp_fascimile_n2)
            #non-overlaphe just came in

        n_ft_time_point = pp[self.pulse_overlap_ending_index:].shape[0]

        self.pp_oscillation_frequencies = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(n_ft_time_point, d = self.dtPp))

        self.pp_oscillation_frequencies_wavenumbers = self.mySpace.unitHandler.wavenumbersFromEnergyUnits(self.pp_oscillation_frequencies)

        total_signal_shape = (len(pp_signals), pp.shape[0])
        nonOverlap_signal_shape = (len(pp_signals), pp[self.pulse_overlap_ending_index:].shape[0])

        total_pp_signal = np.zeros(total_signal_shape, dtype = np.complex)
        total_pp_fascimile_n1_signal = np.zeros(total_signal_shape, dtype = np.complex)
        total_pp_fascimile_n2_signal = np.zeros(total_signal_shape, dtype = np.complex)

        total_ft_pp_signal = np.zeros(nonOverlap_signal_shape, dtype = np.complex)

        for i, tg in enumerate(self.listOfTGExperiments):

            total_pp_signal[i] = pp_signals[i]
            total_pp_fascimile_n1_signal[i] = pp_fascimile_n1_signals[i]
            total_pp_fascimile_n2_signal[i] = pp_fascimile_n2_signals[i]



        for i, tg in enumerate(self.listOfTGExperiments):
            pp_nonOverlap = total_pp_signal[i, self.pulse_overlap_ending_index:]

            pp_fascimile_n1_nonOverlap = total_pp_fascimile_n1_signal[i, self.pulse_overlap_ending_index:]
            pp_fascimile_n2_nonOverlap = total_pp_fascimile_n2_signal[i, self.pulse_overlap_ending_index:]

            #average values
            pp_average = np.average(pp_nonOverlap)
            pp_fascimile_n1_average = np.average(pp_fascimile_n1_nonOverlap)
            pp_fascimile_n2_average = np.average(pp_fascimile_n2_nonOverlap)

            #signals to be used for W
            pp_forW = pp_nonOverlap - pp_average
            pp_fascimile_n1_forW = pp_fascimile_n1_nonOverlap - pp_fascimile_n1_average
            pp_fascimile_n2_forW = pp_fascimile_n2_nonOverlap - pp_fascimile_n2_average

            #FT values
            total_ft_pp_signal[i] = np.fft.fftshift(np.fft.fft(pp_forW))
            #W values
            W = scipy.integrate.simps(np.abs(pp_forW)**2) * self.dtPp
            W_fascimile_n1 = scipy.integrate.simps(np.abs(pp_fascimile_n1_forW)**2) * self.dtPp
            W_fascimile_n2 = scipy.integrate.simps(np.abs(pp_fascimile_n2_forW)**2) * self.dtPp

            self.W_list.append(W)
            self.W_fascimile_n1_list.append(W_fascimile_n1)
            self.W_fascimile_n2_list.append(W_fascimile_n2)


        self.total_pp_signal = total_pp_signal
        self.total_pp_fascimile_n1_signal = total_pp_fascimile_n1_signal
        self.total_pp_fascimile_n2_signal = total_pp_fascimile_n2_signal

        self.pulseWidthsToCalculate_FWHM_femtoseconds = (2.0 *np.sqrt(2.0 * np.log(2.0))) * self.pulseWidthsToCalculate_femtoseconds

        self.total_ft_pp_signal = total_ft_pp_signal


        print "ALL DONE! :D"




def BIG_RED_BUTTON(ppObject):
    "just here to make embarassingly parallel calculations easier"
    return ppObject.calculate()


class interactionHelper(object):
    def __init__(self, interactor, wavefunction):
        self.interactor = interactor
        self.wavefunction = wavefunction

    def calculate(self):
        output = self.interactor.goFromTimeWavefunction(self.wavefunction)
        self.interactor = None
        self.wavefunction = None
        return output


class overlapHelper(object):
    base_frequency_multiple = 2.0
    useFilter = True

    def __init__(self, bra, ket, muTuple, pulse3PlusTuple, lowest_frequency, population_time_index, max_T_index, n):
        self.bra = bra
        self.ket = ket
        self.muTuple = muTuple
        self.pulse3PlusTuple  = pulse3PlusTuple

        self.lowest_frequency = lowest_frequency


        self.population_time_index = population_time_index
        self.max_T_index = max_T_index

        self.n = n
        self.calculated = False

    def calculate(self):
        if self.calculated:
            return self
        dt = self.bra.mySpace.dt

        signalTuple = self.ket.timeOverlapWithOtherBraEWFOfPolarization(self.bra, self.muTuple)

        xSignal = 1.0j * signalTuple[0]
        ySignal = 1.0j * signalTuple[1]
        zSignal = 1.0j * signalTuple[2]

        #clear memory
        self.bra = None
        self.ket = None
        self.muTuple = None



        n_time_point = xSignal.shape[0]

        time_values = dt * np.array(range(0, n_time_point))

        if overlapHelper.useFilter:
            filter_length_time = overlapHelper.base_frequency_multiple * (2.0 * np.pi / self.lowest_frequency )
            filter_length_steps = int(filter_length_time / dt)

            xf = n_time_point - 1 - (self.max_T_index - self.population_time_index) * self.n
            x0 = xf - filter_length_steps

    #        Here we define a third order polynomial that follows the following equations:
    #        f(x0) = 1
    #        f(xf) = 0
    #        f'(x0) = 0
    #        f'(xf) = 0
            denominator = (x0 - xf)**3
            a = -2.0 / denominator
            b = 3.0 * (x0 + xf) / denominator
            c = -6.0 * x0 * xf / denominator
            d = (3.0 * x0 * xf**2 - xf**3) / denominator

            filterVals = np.ones(n_time_point, dtype = np.complex)
            for i in range(x0, xf+1):
                filterVals[i] = a * (i**3) + b * (i**2) + c * i + d
            for i in range(xf+1, n_time_point):
                filterVals[i] = 0.0
        else:
            filterVals = 1.0


        ft_constant = (time_values[1] - time_values[0]) * n_time_point
        xfrequency_spectrum = np.fft.fftshift(np.fft.ifft(xSignal * filterVals)) * ft_constant
        yfrequency_spectrum = np.fft.fftshift(np.fft.ifft(ySignal * filterVals)) * ft_constant
        zfrequency_spectrum = np.fft.fftshift(np.fft.ifft(zSignal * filterVals)) * ft_constant


        self.frequency_values = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(n_time_point, d = dt))

        self.xfrequency_spectrum = xfrequency_spectrum
        self.yfrequency_spectrum = yfrequency_spectrum
        self.zfrequency_spectrum = zfrequency_spectrum

        self.time_values = time_values


        self.local_oscillator_signal = np.array([])

        try:
            convoluterX = self.pulse3PlusTuple[0].myFourierTransformedFunction(self.frequency_values)

        except Exception:
            convoluterX = 0

        try:
            convoluterY = self.pulse3PlusTuple[1].myFourierTransformedFunction(self.frequency_values)
        except Exception:
            convoluterY = 0

        try:
            convoluterZ =  self.pulse3PlusTuple[2].myFourierTransformedFunction(self.frequency_values)
        except Exception:
            convoluterZ = 0

        local_oscillator_signal = convoluterX*np.conj(xfrequency_spectrum) + convoluterY*np.conj(yfrequency_spectrum) + convoluterZ*np.conj(zfrequency_spectrum)
        self.local_oscillator_signal = 2.0 * local_oscillator_signal


        self.calculated = True
        #clear memory
        self.pulse3PlusTuple  = None

        return self

if __name__ == "__main__":
    print "heeeeey"
