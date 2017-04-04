# -*- coding: utf-8 -*-
"""
Created on Fri Jan 04 13:23:27 2013

@author: Joey
"""
import matplotlib.pyplot as plt

import spectroscopy.experiments.experimentBase as experiment
plt_xlims = (-5000, 5000)

abs_test = experiment.experiment.openSavedExperiment("data/2017-04-04/AbsorptionAbs_test.pkl")
abs_test.plot()
plt.savefig("images/load_test_abs")


pp_test = experiment.experiment.openSavedExperiment("data/2017-04-04/Basic_Pump_Probe_PP_object.pkl")

plt.figure()
plt.plot(pp_test.ppTimeSeries_fs, pp_test.timeSeriesTotalSignal)
plt.title("Pump Probe Experiment")
plt.xlabel("T / fs")
plt.ylabel("Probe Beam Absorption")
plt.savefig("images/load_test_PP_test.png")


tg_test = experiment.experiment.openSavedExperiment("data/2017-04-04/Basic_Transient_Grating__tg_test.pkl")
plt.figure()
plt.contourf(tg_test.frequency_values_wavenumbers, tg_test.populationTimes_fs, tg_test.totalSignal_frequency_abs)
plt.xlim(plt_xlims)
plt.title("Transient Grating Experiment")
plt.ylabel("T / fs")
plt.xlabel(r"$\omega_3$")
plt.savefig("images/load_test_TG_test.png")
