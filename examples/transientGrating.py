# -*- coding: utf-8 -*-
"""
Created on Fri Jan 04 13:23:27 2013

@author: Joey
"""

import time
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


import spectroscopy.TimeFunction as TimeFunction
import spectroscopy.NuclearOperator as NuclearOperator

import spectroscopy.experiments.TransientGrating as TG


#must access the systems folder
sys.path.append('..')
import systems.Juergen.one_fast_mode_monomer as monomer
# import systems.naturalNonCondon.monomer as monomer
# import systems.naturalNonCondon.coherentDimer as monomer
#import systems.Juergen.two_mode_monomer as monomer

plt_xlims = -3000, 3000


T_max  = monomer.mySpace.unitHandler.timeUnitsFromSeconds(10.0E-15)

nyquist_frequency = 2.0 * monomer.ElectronicHamiltonian.omega_high / (2.0 * np.pi)
dT_needed = 1.0 / nyquist_frequency

T_steps = int(T_max / dT_needed) + 1
T_steps = T_steps * 5.0
dT = T_max / T_steps

dOmega_wavenumbers = 20.0 * np.sqrt(2.0)

maxFrequency = monomer.mySpace.unitHandler.wavenumbersFromEnergyUnits(0.5 / monomer.mySpace.dt)



pulse_width_stdev_femtoseconds = 2.0 * 1.0E-15 #seconds

pulse_width_stdev = monomer.mySpace.unitHandler.timeUnitsFromSeconds(pulse_width_stdev_femtoseconds)

pulseCenterFrequency = monomer.pulse_carrier_frequency


xPump = TimeFunction.GaussianCosinePulse(monomer.mySpace, centerOmega = pulseCenterFrequency,  timeSpread = pulse_width_stdev)
yPump = TimeFunction.zeroTimeFunction(monomer.mySpace)
zPump = TimeFunction.zeroTimeFunction(monomer.mySpace)

pumpTuple = (xPump, yPump, zPump)


dOmega = monomer.mySpace.unitHandler.energyUnitsFromWavenumbers(dOmega_wavenumbers)

numberOfPumpTerms =  int(np.ceil( 1.0 / ( monomer.mySpace.dt * dOmega)))

maxPumpTime = numberOfPumpTerms * monomer.mySpace.dt


tg_test = TG.PopulationTimeScan(space = monomer.mySpace,
                                         electronicHamiltonian = monomer.ElectronicHamiltonian,
                                         MuxMuyMuzElectronicOperatorTuple = monomer.transitionDipoleTuple_FC,
                                         initialElectronicWavefunction = monomer.initialEWF,
                                         pumpBeamTuple = pumpTuple,
                                         maximumPopulationTime = T_max,
                                         numberOfPopulationTimePoints = T_steps,
                                         maximumProbeTime = maxPumpTime,
                                         numberOfProbeTimePoints = numberOfPumpTerms,
                                         numberOfProcessors = 1,
                                         use_clusterpool = False)
print "calculating Transient Grating...."

startTime = time.time()
tg_test.calculate()
tg_test.save("_tg_test")

plt.figure()
plt.contourf(tg_test.frequency_values_wavenumbers, tg_test.populationTimes_fs, tg_test.totalSignal_frequency_abs)
plt.xlim(plt_xlims)
plt.title("Transient Grating Experiment")
plt.ylabel("T / fs")
plt.xlabel(r"$\omega_3$")
plt.savefig("images/TG_test.png")
