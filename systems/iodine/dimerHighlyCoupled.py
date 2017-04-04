# -*- coding: utf-8 -*-
"""
Created on Fri Jan 04 13:23:27 2013

@author: Joey
"""
import copy
import os
import time
import multiprocessing

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import scipy.integrate

import spectroscopy.Spacetime as Spacetime
import spectroscopy.TimeElectronicWavefunction as TimeElectronicWavefunction
import spectroscopy.ElectronicOperator as ElectronicOperator
import spectroscopy.TimeFunction as TimeFunction
import spectroscopy.NuclearWavefunction as NuclearWavefunction
import spectroscopy.NuclearOperator as NuclearOperator
import spectroscopy.ElectronicWavefunction as ElectronicWavefunction
import spectroscopy.TimePerturbation as TimePerturbation
#import spectroscopy.Experiments as Experiments

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 04 13:23:27 2013

@author: Joey
"""
import numpy as np

import spectroscopy.Spacetime as Spacetime
import spectroscopy.ElectronicOperator as ElectronicOperator
import spectroscopy.NuclearWavefunction as NuclearWavefunction
import spectroscopy.NuclearOperator as NuclearOperator
import spectroscopy.ElectronicWavefunction as ElectronicWavefunction


             
#FROM HERZBERG SPECTRA OF DIATOMIC MOLECULES
reducedMassIodineAMU = 63.466 #amu
reducedMassIodine = 1.0538E-25 #kg
betaCoefficient = 5.976917E21 #1/sqrt(kg * m)
betaCoefficient = betaCoefficient / 10.0 #1/sqrt(kg * cm)
#ground state
#looked up values
ground_omega_e = 214.57 #wavenumbers
ground_omega_e_chi_e = .6127 #wavenumbers
ground_T_e_ = 0.0 #wavenumbers
ground_radius_e = 2.6666E-10 #meters

#calculated values
ground_chi_e = ground_omega_e_chi_e / ground_omega_e #unitless
ground_D_e = ground_omega_e / (4.0 * ground_chi_e) #wavenumbers
ground_beta = 100 * betaCoefficient * np.sqrt(reducedMassIodine * ground_omega_e_chi_e) # 1/m
ground_zpe = ground_omega_e * .5 + ground_omega_e_chi_e * .25


#excited state
#looked up values
excited_omega_e = 128.0 #wavenumbers
excited_omega_e_chi_e = .834 #wavenumbers
excited_T_e = 15641.6 #wavenumbers, 0-->0 transition energy
excited_radius_e = 3.016E-10 #meters

#calculated values
excited_chi_e = excited_omega_e_chi_e / excited_omega_e #unitless
excited_D_e = excited_omega_e / (4.0 * excited_chi_e) #wavenumbers
excited_beta = 100 * betaCoefficient * np.sqrt(reducedMassIodine * excited_omega_e_chi_e) # 1/m
excited_zpe = excited_omega_e * .5 + excited_omega_e_chi_e * .25 #wavenumbers
excited_energy_gap = excited_T_e + ground_zpe - excited_zpe #wavenumbers
excited_center_gap = excited_radius_e - ground_radius_e #meters


##all units from here on out are defined by mySpace
mySpace = Spacetime.Spacetime(xMax = 15.0,
             numberOfNuclearDimenions = 2,
             numberOfElectronicDimensions = 4,
             numberOfSimulationSpacePointsPerNuclearDimension = 100,
             dt_SECONDS = .50000E-15, 
             UnityMassInElectronMasses = 10.0)
             
iodineGroundStateDeValue = mySpace.unitHandler.energyUnitsFromWavenumbers(ground_D_e)
iodineExcitedStateDeValue = mySpace.unitHandler.energyUnitsFromWavenumbers(excited_D_e)

iodineExcitedStateCenterOffset = mySpace.unitHandler.lengthUnitsFromMeters(excited_center_gap)
iodineExcitedStateEnergyOffset = mySpace.unitHandler.energyUnitsFromWavenumbers(excited_energy_gap)

iodineGroundStateBetaValue = 1.0 /  mySpace.unitHandler.lengthUnitsFromMeters(1.0 / ground_beta)
iodineExcitedStateBetaValue = 1.0 /  mySpace.unitHandler.lengthUnitsFromMeters(1.0 / excited_beta)

iodineReducedMass = mySpace.unitHandler.massUnitsFromAmu(reducedMassIodineAMU)

pulse_carrier_frequency = 2500.0 #wavenumbers
pulse_carrier_frequency = mySpace.unitHandler.energyUnitsFromWavenumbers(pulse_carrier_frequency)

#startingPoint = -mySpace.xMax + 10
groundCenter =  - iodineExcitedStateCenterOffset / 2
excitedCenter =  iodineExcitedStateCenterOffset / 2

opticalGap = iodineExcitedStateEnergyOffset
energyOffset = 0 #for more accurate calculation

omega_0_ground = iodineGroundStateBetaValue * np.sqrt(2 * iodineGroundStateDeValue / iodineReducedMass )
omega_0_ground_wavenumbers = mySpace.unitHandler.wavenumbersFromEnergyUnits(omega_0_ground)

omega_0_excited = iodineExcitedStateBetaValue * np.sqrt(2 * iodineExcitedStateDeValue / iodineReducedMass )
omega_0_excited_wavenumbers = mySpace.unitHandler.wavenumbersFromEnergyUnits(omega_0_excited)

HUANG_RHYS = iodineReducedMass * iodineExcitedStateCenterOffset**2 * omega_0_ground / ( 2* mySpace.hbar )
             
iodineGroundMorse = NuclearOperator.morsePotential(mySpace,
                                                     a= iodineGroundStateBetaValue, 
                                                     De=iodineGroundStateDeValue, 
                                                     mass=iodineReducedMass, 
                                                     center=groundCenter, 
                                                     energyOffset = 0.0 )
iodineExcitedMorse = NuclearOperator.morsePotential(mySpace,
                                                     a= iodineExcitedStateBetaValue, 
                                                     De= iodineExcitedStateDeValue, 
                                                     mass=iodineReducedMass, 
                                                     center=excitedCenter, 
                                                     energyOffset = energyOffset)

iodineGroundHO = iodineGroundMorse.correspondingHarmonicOscillator()
iodineExcitedHO = iodineExcitedMorse.correspondingHarmonicOscillator()        


lowestEnergyTransitionHO = iodineExcitedHO.energyEigenvalue(0) -  iodineGroundHO.energyEigenvalue(0)

mostImportantTransitionHO_wavenumbers = 1875.0
mostImportantTransitionHO = mySpace.unitHandler.energyUnitsFromWavenumbers(mostImportantTransitionHO_wavenumbers)

mostImportantTransitionMorse_wavenumbers = 3100.0
mostImportantTransitionMorse = mySpace.unitHandler.energyUnitsFromWavenumbers(  mostImportantTransitionMorse_wavenumbers  )

#HARMONIC OSCILLATOR   
groundGroundStateHO = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [iodineGroundHO, iodineGroundHO] )
excitedGroundStateHO = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [iodineExcitedHO, iodineGroundHO ] )
groundExcitedStateHO = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [iodineGroundHO, iodineExcitedHO ] )
excitedExcitedStateHO = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [iodineExcitedHO, iodineExcitedHO ] )

J = 3.2 * omega_0_ground
 
couplingOperator = NuclearOperator.constantPositionNuclearOperator(mySpace, J)


groundStateNuclearWFHO = NuclearWavefunction.nuclearWavefunction(mySpace, groundStateNuclearHamiltonian = groundGroundStateHO )


ElectronicHamiltonianHO = ElectronicOperator.ElectronicHamiltonian(mySpace, [(0,0, groundGroundStateHO), 
                                                            (1,1, excitedGroundStateHO),(2,2, groundExcitedStateHO), 
                                                            (1,2, couplingOperator), (2,1, couplingOperator),
                                                            (3,3, excitedExcitedStateHO),])
                                                            
                                                            
initialEWFHO = ElectronicWavefunction.electronicWavefunction(mySpace, 
                       listOfNuclearWavefunctions = [groundStateNuclearWFHO, 0, 0, 0],
                       Normalize=True)
                       
#MORSE   
#groundStateMorse = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [iodineGroundMorse] )
#
#excitedStateMorse = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [iodineExcitedMorse ] )
#
#
#groundStateNuclearWFMorse = NuclearWavefunction.nuclearWavefunction(mySpace, groundStateNuclearHamiltonian = groundStateMorse )
#
#
#ElectronicHamiltonianMorse = ElectronicOperator.ElectronicHamiltonian(mySpace, [(0,0, groundStateMorse), 
#                                                            (1,1, excitedStateMorse)])
#                                                            
#                                                            
#initialEWFMorse = ElectronicWavefunction.electronicWavefunction(mySpace, 
#                       listOfNuclearWavefunctions = [groundStateNuclearWFMorse, 0],
#                       Normalize=True)

                     

#calculated using def2-TZVP basis and PBE0 function
#Provided by Dimitrij Rappaport
muGtoEValueConstant = -.165270   #e * r_b
muGtoEValueConstant = mySpace.unitHandler.lengthUnitsFromBohrRadii(muGtoEValueConstant)
muGtoEValueLinearCo = -.0548235  #e * r_b / r_b


muEtoGValueConstant = -.132477   #e * r_b
muEtoGValueConstant = mySpace.unitHandler.lengthUnitsFromBohrRadii(muEtoGValueConstant)
muEtoGValueLinearCo = -.04555 #e * r_b / r_b

mu_0 = (muGtoEValueConstant + muEtoGValueConstant ) / 2.0


constantMu = NuclearOperator.constantPositionNuclearOperator(mySpace, mu_0)

mu_double_prime = muGtoEValueLinearCo + muEtoGValueLinearCo

mu_prime = mu_double_prime * np.sqrt(mySpace.hbar / ( 2.0 * iodineReducedMass * omega_0_ground))

constantMu = NuclearOperator.constantPositionNuclearOperator(mySpace, muEtoGValueConstant)

linear_mu_function = lambda x, y: mu_0 + mu_double_prime * (x + y) 

linearMu = NuclearOperator.functionalPositionNuclearOperator(mySpace, lambda x, y: linear_mu_function(x, y) )



                                                                 
xTransitionDipole_FC = ElectronicOperator.ElectronicPositionOperator(mySpace, [(0, 1, constantMu), (1, 0, constantMu), 
                                                                 (0, 2, constantMu), (2, 0, constantMu),  
                                                                 (1, 3, constantMu), (3, 1, constantMu),  
                                                                 (2, 3, constantMu), (3, 2, constantMu)])
yTransitionDipole_FC = ElectronicOperator.ElectronicPositionOperator(mySpace, [])
zTransitionDipole_FC = ElectronicOperator.ElectronicPositionOperator(mySpace, [])

transitionDipoleTuple_FC = (xTransitionDipole_FC, yTransitionDipole_FC, zTransitionDipole_FC)



xTransitionDipole_linear = ElectronicOperator.ElectronicPositionOperator(mySpace, [(0, 1, linearMu), (1, 0, linearMu), 
                                                                 (0, 2, linearMu), (2, 0, linearMu),  
                                                                 (1, 3, linearMu), (3, 1, linearMu),  
                                                                 (2, 3, linearMu), (3, 2, linearMu)])

yTransitionDipole_linear = ElectronicOperator.ElectronicPositionOperator(mySpace, [])
zTransitionDipole_linear = ElectronicOperator.ElectronicPositionOperator(mySpace, [])


transitionDipoleTuple_linear = (xTransitionDipole_linear, yTransitionDipole_linear, zTransitionDipole_linear)









