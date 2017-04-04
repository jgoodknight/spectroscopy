# -*- coding: utf-8 -*-
"""
Created on Fri Jan 04 13:23:27 2013

@author: Joey
"""

import time
import multiprocessing
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy.integrate

import spectroscopy.TimeFunction as TimeFunction

import spectroscopy.experiments.TransientGrating as TG
import spectroscopy.experiments.PumpProbe as PP
import spectroscopy.experiments.experimentBase as experiment

import spectroscopy.NuclearOperator as NuclearOperator

#must access the systems folder
sys.path.append('..')
import systems.Juergen.one_fast_mode_monomer as monomer
# import systems.naturalNonCondon.coherentDimer as monomer
# import systems.naturalNonCondon.monomer_S_10 as monomer
# import systems.naturalNonCondon.monomer_S_0 as monomer


T_max  = monomer.mySpace.unitHandler.timeUnitsFromSeconds(50.0E-15)

nyquist_frequency = 2.0 * monomer.ElectronicHamiltonian.omega_high / (2.0 * np.pi)
dT_needed = 1.0 / nyquist_frequency

T_steps = int(T_max / dT_needed) + 1
T_steps = T_steps * 5.0
dT = T_max / T_steps


pulse_width_stdev_femtoseconds = 1.0 * 1.0E-15 #seconds

pulse_width_stdev = monomer.mySpace.unitHandler.timeUnitsFromSeconds(pulse_width_stdev_femtoseconds)

pulseCenterFrequency = monomer.pulse_carrier_frequency


xPump = TimeFunction.GaussianCosinePulse(monomer.mySpace, centerOmega = pulseCenterFrequency,  timeSpread = pulse_width_stdev)
yPump = TimeFunction.zeroTimeFunction(monomer.mySpace)
zPump = TimeFunction.zeroTimeFunction(monomer.mySpace)

pumpTuple = (xPump, yPump, zPump)


pp_test = PP.Basic(space = monomer.mySpace,
                                        electronicHamiltonian = monomer.ElectronicHamiltonian,
                                        MuxMuyMuzElectronicOperatorTuple = monomer.transitionDipoleTuple_FC,
                                        initialElectronicWavefunction = monomer.initialEWF,
                                        pumpBeamTuple = pumpTuple,
                                        maximumEvolutionTime = T_max,
                                        numberOfTimePoints = T_steps)
print "calculating Pump Probe Signals"
startTime = time.time()
pp_test.calculate()
pp_test.save("PP_object")
print "TOTAL Time Elapsed in seconds:", time.time() - startTime

plt.figure()
plt.plot(pp_test.ppTimeSeries_fs, pp_test.timeSeriesTotalSignal)
plt.title("Pump Probe Experiment")
plt.xlabel("T / fs")
plt.ylabel("Probe Beam Absorption")
plt.savefig("images/PP_test.png")
