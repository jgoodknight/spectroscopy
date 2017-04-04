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
mpl.use('Agg')
import matplotlib.pyplot as plt

import spectroscopy.Spacetime as Spacetime
import spectroscopy.TimeElectronicWavefunction as TimeElectronicWavefunction
import spectroscopy.TimeFunction as TimeFunction
import spectroscopy.TimePerturbation as TimePerturbation

import experimentBase

import ClusterPool.ClusterPool as ClusterPool
import ClusterPool.Dispatcher as Dispatcher

NUMBER_PROCESSES = 5
USE_CLUSTERPOOL = True

def BIG_RED_BUTTON(ppObject):
    "just here to make embarassingly parallel calculations easier"
    print "BOMBS AWAY"
    return ppObject.calculate()


class Base(experimentBase.experiment):
    "Just calculates the Pump/probe signal of your chosen system: no frills"
    experiment_type_string = "Basic_Pump_Probe_"
    
    
    def __init__(self, space,
                 electronicHamiltonian,
                 MuxMuyMuzElectronicOperatorTuple,
                 pumpBeamTuple,
                 maximumEvolutionTime,
                 initial_state_quantum_numbers = None,
                 save_EWF_every_n_steps = 100,
                 override_pulse_overlap_time_interval = None,
                 IDstring = ""):
        #store all the variables
        self.mySpace = space

        self.dipoleTuple = MuxMuyMuzElectronicOperatorTuple
        self.excitationMuTuple = (self.dipoleTuple[0].excitationOperator(), self.dipoleTuple[1].excitationOperator(), self.dipoleTuple[2].excitationOperator())
        self.relaxationMuTuple = (self.dipoleTuple[0].relaxationOperator(), self.dipoleTuple[1].relaxationOperator(), self.dipoleTuple[2].relaxationOperator())
        
        self.pumpEx, self.pumpEy, self.pumpEz = pumpBeamTuple

        self.myElectronicHamiltonian = electronicHamiltonian
        self.myFreePropagator = electronicHamiltonian.myPropagator()
        
        
        self.intitialEWF = self.myElectronicHamiltonian.groundStateElectronicWavefunction()
        self.initial_energy = self.intitialEWF.initial_energy

        #We are given a target maximum waiting and a specific number of time points to take.
        #We then have to scale that to the closest multiple of mySpace.dt to make the propagation work properly

        self.save_EWF_every_n_steps = save_EWF_every_n_steps

        
        self.calculated = False
        self.IDstring = IDstring

        if override_pulse_overlap_time_interval == None:
            self.pulse_overlap_ending_time = 2.0 * 2.0 * max(map(lambda x: x.timePillow, pumpBeamTuple))
        else:
            self.pulse_overlap_ending_time = override_pulse_overlap_time_interval
            
        self.pulse_overlap_ending_index = self.pulse_overlap_ending_time / self.mySpace.dt
        self.pulse_overlap_ending_index = int(self.pulse_overlap_ending_index)

        self.maximumTime = self.pulse_overlap_ending_time + maximumEvolutionTime
        self.Number_of_evolution_points = int(self.maximumTime / self.mySpace.dt)

        self.monomer = (self.mySpace.electronicDimensionality == 2)
        self.delta_function_excitation = (self.dipoleTuple[0] + self.dipoleTuple[1] + self.dipoleTuple[2]) * copy.deepcopy(self.intitialEWF)
        
    def perturbationOperatorAtTime(self, T):
        return self.pumpEx.valueAtTime(T) * self.dipoleTuple[0] + self.pumpEy.valueAtTime(T) * self.dipoleTuple[1] + self.pumpEz.valueAtTime(T) * self.dipoleTuple[2]
        
     
class Basic_One_Gaussian_Excitation(Base):
    
    experiment_type_string = "Basic_Dynamics_"
        
    def calculate(self):
        if self.calculated:
            print "You've already done the calculation, look at the results instead!"
            return
        overallStartTime = time.time()
        
        tStep = 0.0
        t = 0.0
        
        c = -1.0j * self.mySpace.dt / ( 2.0 * self.mySpace.unitHandler.HBAR)
        
        old_application = self.perturbationOperatorAtTime(-self.mySpace.dt) * self.intitialEWF * cmath.exp(1.0j * self.initial_energy * self.mySpace.dt)  
        old_EWF = 0.0 * old_application
        
        #relevant_observables
        time_series = []
        electric_field_emission_series = []
        electronic_density_matrix_series = []
        total_population_series = []
        pulse_amplitude_series = []
        
        saved_ewf_series = []
        saved_ewf_times_series = []
        
        n = self.mySpace.electronicDimensionality
        self.n=n
        save_counter = 0
        
        for tStep in range(self.Number_of_evolution_points):
            t = tStep * self.mySpace.dt
            time_series.append(t)
            
            pulse_amp = np.sqrt(np.abs(self.pumpEx.valueAtTime(t))**2 + np.abs(self.pumpEy.valueAtTime(t))**2 + np.abs(self.pumpEz.valueAtTime(t))**2)
            pulse_amplitude_series.append(pulse_amp)
            
            phase_factor = cmath.exp(-1.0j * self.initial_energy * t)
            
            new_application = (self.perturbationOperatorAtTime(t) *  self.intitialEWF) * phase_factor
            
            thingToPropagate = old_EWF + c * old_application
            propagated_thing = self.myFreePropagator.APPLY(thingToPropagate)
            
            new_EWF = propagated_thing  + c * new_application
            
            #Save relevant observables            
            
            #ELECTRONIC DENSITY MATRIX CALCULATION
            
            total_population = 0.0
            new_density_matrix = np.zeros((n,n), dtype=np.complex)
            for i in range(0, n):
                for j in range(0, n):
                    new_density_matrix[i,j] = new_EWF[i].overlap( new_EWF[j])
                    if i == j:
                        total_population = total_population + new_density_matrix[i,j]
                        
            if tStep == self.pulse_overlap_ending_index:
                normalizer = total_population
                
            total_population_series.append(total_population)      
            electronic_density_matrix_series.append(new_density_matrix)
            #ELECTRIC FIELD CALCULATION
            emission = 1.0j * cmath.exp(1.0j * self.initial_energy * t) * self.delta_function_excitation.overlap(new_EWF)
            electric_field_emission_series.append(emission)
            
            #SAVE EWF?
            if save_counter == self.save_EWF_every_n_steps:
                #SAVE SHIT
                saved_ewf_series.append(new_EWF)
                saved_ewf_times_series.append(t)
                #RESET COUNTER
                save_counter = 0
            else:
                save_counter = save_counter + 1
            #RE-SET LOOP
            old_application = new_application
            old_EWF = new_EWF
            
        total_population_series = np.array(total_population_series)
        
        self.maximum_population = np.max(total_population_series)
        
        self.electronic_density_matrix_series = np.array(electronic_density_matrix_series) 
        self.electric_field_emission_series = np.array(electric_field_emission_series) 
        
        self.time_series = np.array(time_series)
        self.time_series_fs = self.mySpace.unitHandler.femtosecondsFromTime(self.time_series)
        
        self.pulse_amplitude_series = np.array(pulse_amplitude_series)
        self.pulse_amplitude_series = self.pulse_amplitude_series / np.max(self.pulse_amplitude_series)
        
        
            
        time_series_non_overlap = self.time_series[self.pulse_overlap_ending_index:]
#        density_matrix_series_non_overlap = self.electronic_density_matrix_series[self.pulse_overlap_ending_index]
        
#        density_matrix_frequency_series = np.fft.fftshift(np.fft.fft(density_matrix_series_non_overlap, axis = 0))
        self.emission_spectrum = np.fft.fftshift(np.fft.fft(self.electric_field_emission_series[self.pulse_overlap_ending_index:]))
        self.frequency_series = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(n = time_series_non_overlap.shape[0], d= self.mySpace.dt))
        
        self.pulse_spectrum = np.abs(self.pumpEx.myFourierTransformedFunction(self.frequency_series))
        self.pulse_spectrum += np.abs(self.pumpEy.myFourierTransformedFunction(self.frequency_series))
        self.pulse_spectrum += np.abs(self.pumpEz.myFourierTransformedFunction(self.frequency_series))
        
#        self.t_0 = time_series_non_overlap[0]
#        T_total = time_series_non_overlap[-1] - time_series_non_overlap[0]
#        self.density_matrix_frequency_series = T_total * density_matrix_frequency_series / np.sqrt(.02 * np.pi)
#        self.density_matrix_frequency_series = T_total * np.exp(-1.0j * self.frequency_series * t_0) * density_matrix_frequency_series / np.sqrt(.02 * np.pi)
        self.frequency_series_wavenumbers = self.mySpace.unitHandler.wavenumbersFromEnergyUnits(self.frequency_series)
        self.calculated = True
        
        self.saved_ewf_series = saved_ewf_series
        self.saved_ewf_time_series = np.array(saved_ewf_times_series)
        self.saved_ewf_time_series_fs = self.mySpace.unitHandler.femtosecondsFromTime(self.saved_ewf_time_series)

        self.timeElapsed_seconds = time.time() - overallStartTime
        print self.IDstring + ": time elapsed (min) for Dynamics calculation", self.timeElapsed_seconds / 60.0
        
        self.save(self.IDstring)
        
        return self
        
            
     

class ClusterScanPulseWidthPulseCenter(experimentBase.experiment):

    experiment_type_string = "Witness_Cluster_Pump_Probe_"
    def __init__(self, space,
                 electronicHamiltonian,
                 MuxMuyMuzElectronicOperatorTuple,
                 maximumEvolutionTime,
                 pulseCenterFrequencies,
                 minimumPulseWidth,
                 maximumPulseWidth,
                 numberOfPulseWidthExperimentsToDo,
                 initial_state_quantum_numbers = None,
                 save_EWF_every_n_steps = 100,
                 string_identity = "",
                 log_pulse_width_scale = False):
        #store all the variables
        self.mySpace = space
        
        self.string_identity = string_identity

        self.muTuple = MuxMuyMuzElectronicOperatorTuple

        self.myElectronicHamiltonian = electronicHamiltonian

        if minimumPulseWidth / self.mySpace.dt < 1.0:
            minimumPulseWidth = self.mySpace.dt
            warnings.warn("requested minimum pulse width smaller than time discretization!  changing minimum pulse width to dt")
        if log_pulse_width_scale:
            self.pulseWidthsToCalculate = np.logspace(np.log10(minimumPulseWidth), np.log10(maximumPulseWidth), numberOfPulseWidthExperimentsToDo)
        else:   
            self.pulseWidthsToCalculate = np.linspace(minimumPulseWidth, maximumPulseWidth, numberOfPulseWidthExperimentsToDo)
        
        self.pulseWidthsToCalculate_femtoseconds = self.mySpace.unitHandler.femtosecondsFromTime(self.pulseWidthsToCalculate)
        self.pulseWidthsToCalculate_FWHM_femtoseconds = (2.0 *np.sqrt(2.0 * np.log(2.0))) * self.pulseWidthsToCalculate_femtoseconds
        
        self.pulseCenterToCalculate = pulseCenterFrequencies
        
        pulseBeamTuples = []
        for width in self.pulseWidthsToCalculate:
            for center in self.pulseCenterToCalculate:
                newPulseX = TimeFunction.GaussianCosinePulse(self.mySpace, centerOmega = center, timeSpread = width)
                newPulseY = TimeFunction.GaussianCosinePulse(self.mySpace, centerOmega = center, timeSpread = width)
                newPulseZ = TimeFunction.zeroTimeFunction(self.mySpace)
                pulseBeamTuples.append((newPulseX.plusFunction(), newPulseY.plusFunction(), newPulseZ))

        self.pumpBeamTuplesToCalculate = pulseBeamTuples
        
        self.save_EWF_every_n_steps = save_EWF_every_n_steps
        self.initial_state_quantum_numbers = initial_state_quantum_numbers

        self.maximumEvolutionTime = maximumEvolutionTime

        self.pulse_overlap_ending_time = 6.0 * maximumPulseWidth
        self.pulse_overlap_ending_index = self.pulse_overlap_ending_time / self.mySpace.dt
        self.pulse_overlap_ending_index = int(self.pulse_overlap_ending_index)


        #now it's time to figure out how many processors to use


    def calculate(self):

        #first create all the objects:
        self.listOfDynamicsExperiments = []

        
        for ii, pumpTuple in enumerate(self.pumpBeamTuplesToCalculate):
            print "It is now", datetime.datetime.now()
            idNo =  self.string_identity + "_sigma= %s" % str(pumpTuple[0].sigma)+ "_center= %s" % str(pumpTuple[0].omega)
            newPP = Basic_One_Gaussian_Excitation(self.mySpace,
                                                 self.myElectronicHamiltonian,
                                                 self.muTuple,
                                                 pumpTuple,
                                                 self.maximumEvolutionTime,
                                                 initial_state_quantum_numbers = self.initial_state_quantum_numbers,
                                                 save_EWF_every_n_steps = self.save_EWF_every_n_steps,
                                                 override_pulse_overlap_time_interval = self.pulse_overlap_ending_time,
                                                 IDstring = idNo)
                          
            self.listOfDynamicsExperiments.append(newPP)

#        
        print "creating pool"
        myPool = ClusterPool.Pool()
        print "running subprocesses"
        self.listOfDynamicsExperiments = myPool.map('calculate', self.listOfDynamicsExperiments, index_of_slowest_calculation = -1)
        
            
        print "done!"
        



if __name__ == "__main__":
    pass
