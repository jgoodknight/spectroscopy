# -*- coding: utf-8 -*-
"""
Created on Fri Jan 04 13:23:27 2013
looked over for release 3 april
@author: Joey
"""
import copy

import numpy as np

import multiprocessing
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import spectroscopy.TimeFunction as TimeFunction
import spectroscopy.TimePerturbation as TimePerturbation
import spectroscopy.TimeElectronicWavefunction as TimeElectronicWavefunction
import spectroscopy.ElectronicWavefunction as ElectronicWavefunction

import experimentBase


class DipoleAbsorption(experimentBase.experiment):
    experiment_type_string = "Absorption"
    def __init__(self,
                 space,
                 electronicHamiltonian,
                 MuxMuyMuzElectronicOperatorTuple,
                 initialElectronicWavefunction,
                 dOmegaResolution,
                 omegaMax,
                 pauseCalculation = False):
        """
        An object to calculate the one-interaction absorption spectrum of a given system

        @param space: the SpaceTime object the system lives in
        @param electronicHamiltonian: Hamiltonian for the system
        @param MuxMuyMuzElectronicOperatorTuple: tuple of eletronic operators (x, y, z) which describe the transition dipole of the system
        @param initialElectronicWavefunction: the starting state of the system
        @param dOmegaResolution: required resolution in natural energy units
        @param omegaMax; required maximum ferquency (will set dt)
        @param pauseCalculation: whether to calculate the spectrum immediately

        @return: an Absortion object
        """
        #store all the variables
        self.mySpace = space


        Tneeded = 1.0 * np.pi / dOmegaResolution
        Nneeded = int(omegaMax / dOmegaResolution)

        self.oldDt = self.mySpace.dt

        self.dt = Tneeded / Nneeded

        self.mySpace.dt = self.dt

        self.muTuple  = MuxMuyMuzElectronicOperatorTuple

        EX = TimeFunction.deltaFunction(self.mySpace, location = self.dt)
        EY = TimeFunction.deltaFunction(self.mySpace, location = self.dt)
        EZ = TimeFunction.deltaFunction(self.mySpace, location = self.dt)


        self.laser_tuple = (EX, EY, EZ)

        self.myPerturbation = TimePerturbation.electricDipole(self.mySpace, electronicHamiltonian, MuxMuyMuzElectronicOperatorTuple, self.laser_tuple, overrideDtForPerturbation = self.dt)
        self.electronicHamiltonian = electronicHamiltonian


        self.initialElectronicWavefunction = initialElectronicWavefunction
        self.freeEvolutionPropagator = self.electronicHamiltonian.myPropagator(overrideDT = self.dt)

        self.Nneeded = Nneeded

        self.calculated = False
        if pauseCalculation:
            pass
        else:
            self.calculate()

    def calculate(self):
        if self.calculated:
            print "already calculated"
            return None


        self.zeroOrderTimeWavefunction = TimeElectronicWavefunction.timeElectronicWavefunction(self.mySpace)
        self.zeroOrderTimeWavefunction.applyOperatorsNTimesOnInitialWavefunction([self.freeEvolutionPropagator], N = self.Nneeded, initialWF = self.initialElectronicWavefunction)

        pulseInteractor = TimePerturbation.electricDipoleOneInteraction(space = self.mySpace, electronicHamiltonian = self.electronicHamiltonian, MuxMuyMuzElectronicOperatorTuple = self.muTuple, ExEyEzTimeFunctionTuple = self.laser_tuple, maxTime = 0.0)

        self.interacted_Wavefunction = pulseInteractor.goFromTimeWavefunction(self.zeroOrderTimeWavefunction)

        self.xEmission, self.yEmission, self.zEmission = self.interacted_Wavefunction.timeOverlapWithOtherBraEWFOfPolarization(self.zeroOrderTimeWavefunction, self.muTuple)

        self.timeIntensity = np.abs(self.xEmission)**2 + np.abs(self.yEmission)**2 + np.abs(self.zEmission)**2

        n_time_point = self.xEmission.shape[0]

        lowest_frequency_from_H = self.electronicHamiltonian.omega_low

        filter_length_time = 1.0 * (2.0 * np.pi / lowest_frequency_from_H )
        filter_length_steps = int(filter_length_time / self.mySpace.dt)

#        Create a filter to eliminate edge effects for the spectrum
        xf = n_time_point - 1
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

        gaussian_filter_stdev = float(n_time_point) / 8.0
        gaussian_filter_mean = float(n_time_point) / 2.0
        filter_indeces = np.array(range(n_time_point))
        filterVals = np.exp(-(filter_indeces - gaussian_filter_mean)**2 / (2.0 * gaussian_filter_stdev**2)) / np.sqrt(2.0 * np.pi * gaussian_filter_stdev**2)
#        filterVals = 1.0

        self.omegaSeries = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(self.xEmission.shape[0], d = self.mySpace.dt))
        self.omegaSeries_wavenumbers = self.mySpace.unitHandler.wavenumbersFromEnergyUnits(self.omegaSeries)

        ft_constant = (self.mySpace.dt) * n_time_point
        xspectrum = np.fft.fftshift(np.fft.ifft(self.xEmission * filterVals)) * ft_constant
        yspectrum = np.fft.fftshift(np.fft.ifft(self.yEmission * filterVals)) * ft_constant
        zspectrum = np.fft.fftshift(np.fft.ifft(self.zEmission * filterVals)) * ft_constant

        self.frequencyIntensity = np.abs(xspectrum)**2 + np.abs(yspectrum)**2 + np.abs(zspectrum)**2
        self.mySpace.dt = self.oldDt

        self.calculated = True
        print "One absorption done calculating!"

        return self

    def plot(self):
        plt.figure()
        w = self.omegaSeries
        w = self.mySpace.unitHandler.wavenumbersFromEnergyUnits(w)
        plt.plot(w, self.frequencyIntensity)
        plt.title("Absorption Spectrum")
        plt.xlabel(r"$\omega$")
        plt.ylabel(r"$|I(\omega)|^2$")

    def plotNormalized(self):
        plt.figure()
        w = self.omegaSeries
        w = self.mySpace.unitHandler.wavenumbersFromEnergyUnits(w)
        plt.plot(w, self.frequencyIntensity / np.max(self.frequencyIntensity))
        plt.title("Absorption Spectrum")
        plt.xlabel(r"$\omega$ (cm$^{-1}$)")
        plt.ylabel(r"$I(\omega)/I_max$")

    def plotLogScaleY(self):
        plt.figure()
        w = self.omegaSeries
        w = self.mySpace.unitHandler.wavenumbersFromEnergyUnits(w)
        plt.semilogy(w, self.frequencyIntensity / np.max(self.frequencyIntensity))
        plt.title("Absorption Spectrum")
        plt.xlabel(r"$\omega$ (cm$^{-1}$)")
        plt.ylabel(r"$I(\omega)/I_max$")

class ThermallyAveragedDipoleAbsorption(experimentBase.experiment):
    "Average over a bunch of thermal states and random disorder"
    def __init__(self,
                 space,
                 electronicHamiltonian,
                 MuxMuyMuzElectronicOperatorTuple,
                 groundElectronicStateNuclearHamiltonian,
                 dOmegaResolution,
                 omegaMax,
                 numberOfThermalAverages,
                 numberOfProcessorsToUse,
                 disorderCovarianceMatrix = None,
                 temperatureInKelvin = 273.0,
                 probeBeamTuple=None):
        #store all the variables
        self.mySpace = space

        self.muTuple = MuxMuyMuzElectronicOperatorTuple

        self.myElectronicHamiltonian = electronicHamiltonian

        #create the list of rotated mu vectors
        listofStartingWavefunctions = []
        for ii in range(numberOfThermalAverages ):
            newNuclearWF = groundElectronicStateNuclearHamiltonian.randomWavefunctionDrawnFromThermalDistributions(temperatureInKelvin)
            correspondingElectronicWF = ElectronicWavefunction.electronicWavefunction(self.mySpace, [newNuclearWF] + [0] * (self.mySpace.electronicDimensionality - 1))
            listofStartingWavefunctions.append(correspondingElectronicWF)


        self.startingEWFs = listofStartingWavefunctions

        if disorderCovarianceMatrix is not None:
            self.addDisorder = True
        else:
            self.addDisorder = False

        self.disorderCovarianceMatrix = disorderCovarianceMatrix

        self.numberOfProcessors = numberOfProcessorsToUse

        self.dOmegaResolution = dOmegaResolution
        self.omegaMax = omegaMax
        self.calculated = False



    def calculate(self):
        #first create all the objects:
        if self.calculated:
            return self
        self.listOfAbsorptionExperiments = []
        for ii, ewf in enumerate(self.startingEWFs):
            if self.addDisorder:
                electronicHamiltonian = self.myElectronicHamiltonian.addDiagonalDisorder(self.disorderCovarianceMatrix)
            else:
                electronicHamiltonian = self.myElectronicHamiltonian

            newAbs = DipoleAbsorption(self.mySpace,
                                     electronicHamiltonian,
                                     self.muTuple,
                                     ewf,
                                     self.dOmegaResolution,
                                     self.omegaMax,
                                     pauseCalculation = True)
            self.listOfAbsorptionExperiments.append(newAbs)

        workerPool = multiprocessing.Pool(self.numberOfProcessors)
        self.listOfCalculatedAbsorptionExperiments = workerPool.map(BIG_RED_BUTTON, self.listOfAbsorptionExperiments)
        print "Done with the calculation"
        self.calculated = True
        return self

    def plotAll(self):
        allSignals = map(lambda x: x.frequencyIntensity, self.listOfCalculatedAbsorptionExperiments)
        w = self.listOfCalculatedAbsorptionExperiments[0].omegaSeries
        w = self.mySpace.unitHandler.wavenumbersFromEnergyUnits(w)
        plt.figure()
        for sig in allSignals:
            plt.plot(w, sig / np.max(sig))
        plt.title("Absorption Spectrum")
        plt.xlabel(r"$\omega$ (cm$^{-1}$)")
        plt.ylabel(r"$I(\omega)/I_max$")


    def plotAverageNormalized(self):
        allSignals = map(lambda x: x.frequencyIntensity, self.listOfCalculatedAbsorptionExperiments)
        w = self.listOfCalculatedAbsorptionExperiments[0].omegaSeries
        w = self.mySpace.unitHandler.wavenumbersFromEnergyUnits(w)
        plt.figure()
        aveSignal = 0
        for sig in allSignals:
            aveSignal = aveSignal + sig
        N = len(allSignals)
        aveSignal = aveSignal / float(N)
        aveSignal = aveSignal / np.max(aveSignal)
        plt.plot(w, aveSignal)
        plt.title("Thermally Averaged Absorption Spectrum")
        plt.xlabel(r"$\omega$ (cm$^{-1}$)")
        plt.ylabel(r"$I(\omega)/I_max$")


def BIG_RED_BUTTON(exp):
    return exp.calculate()
if __name__ == "__main__":
    print "Hi!"
