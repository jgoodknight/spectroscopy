# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:40:19 2013

@author: joey
"""
import itertools
import sys
import time

import numpy as np
import matplotlib
import scipy.integrate
import matplotlib.pyplot as plt

from scipy import signal

import spectroscopy
import spectroscopy.experiments
from spectroscopy.experiments import Absorption

#must access the systems folder
sys.path.append('..')
#import systems.naturalNonCondon.monomer as monomer
import systems.naturalNonCondon.monomer_S_10 as monomer
# import systems.naturalNonCondon.monomer_S_0 as monomer


maxFrequency = monomer.mySpace.unitHandler.energyUnitsFromWavenumbers(6000.0)
dOmega = monomer.mySpace.unitHandler.energyUnitsFromWavenumbers(1.0)

startTime = time.time()
abs_test = Absorption.DipoleAbsorption(monomer.mySpace,
                                         electronicHamiltonian = monomer.ElectronicHamiltonian,
                                         MuxMuyMuzElectronicOperatorTuple = monomer.transitionDipoleTuple_FC,
                                         initialElectronicWavefunction = monomer.initialEWF,
                                         dOmegaResolution = dOmega,
                                         omegaMax = maxFrequency)

abs_test.save("Abs_test")
abs_test.plot()
plt.savefig("images/abs_test.png")

print "TOTAL Time Elapsed in seconds:", time.time() - startTime
