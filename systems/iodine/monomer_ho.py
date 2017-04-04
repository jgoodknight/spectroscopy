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
mySpace = Spacetime.Spacetime(xMax_tuple = (17.50,),
             numberOfNuclearDimenions = 1,
             numberOfElectronicDimensions = 2,
             numberOfSimulationSpacePointsPerNuclearDimension_tuple = (100,),
             dt_SECONDS = .500E-15,
             UnityMassInElectronMasses = 10.0)

             
             
iodineGroundStateDeValue = mySpace.unitHandler.energyUnitsFromWavenumbers(ground_D_e)
iodineExcitedStateDeValue = mySpace.unitHandler.energyUnitsFromWavenumbers(excited_D_e)

iodineExcitedStateCenterOffset = mySpace.unitHandler.lengthUnitsFromMeters(excited_center_gap)
iodineExcitedStateEnergyOffset = mySpace.unitHandler.energyUnitsFromWavenumbers(excited_energy_gap)

iodineGroundStateBetaValue = 1.0 /  mySpace.unitHandler.lengthUnitsFromMeters(1.0 / ground_beta)
iodineExcitedStateBetaValue = 1.0 /  mySpace.unitHandler.lengthUnitsFromMeters(1.0 / excited_beta)

iodineReducedMass = mySpace.unitHandler.massUnitsFromAmu(reducedMassIodineAMU)

pulse_carrier_frequency = 1875.0 #wavenumbers
pulse_carrier_frequency = mySpace.unitHandler.energyUnitsFromWavenumbers(pulse_carrier_frequency)

#startingPoint = -mySpace.xMax + 10
groundCenter =  0
excitedCenter =  iodineExcitedStateCenterOffset 

dx_1 = iodineExcitedStateCenterOffset


omega_0_ground = iodineGroundStateBetaValue * np.sqrt(2 * iodineGroundStateDeValue / iodineReducedMass )
omega_0_ground_wavenumbers = mySpace.unitHandler.wavenumbersFromEnergyUnits(omega_0_ground)

omega_0_excited = iodineExcitedStateBetaValue * np.sqrt(2 * iodineExcitedStateDeValue / iodineReducedMass )
omega_0_excited_wavenumbers = mySpace.unitHandler.wavenumbersFromEnergyUnits(omega_0_excited)

HUANG_RHYS = iodineReducedMass * iodineExcitedStateCenterOffset**2 * omega_0_ground / ( 2* mySpace.hbar )


##MATH PREP DONE, DEFINE OBJECTS NOW


nuclear_ground_1 = NuclearOperator.harmonicOscillator(mySpace, 
                                                 omega=omega_0_ground, 
                                                 mass=iodineReducedMass,
                                                 center= groundCenter)
nuclear_excited_1 = NuclearOperator.harmonicOscillator(mySpace, 
                                                 omega=omega_0_excited, 
                                                 mass=iodineReducedMass,
                                                 center= excitedCenter)     


x_max_needed = 2.0 * dx_1 + 4.0 *  nuclear_ground_1.sigma
print "xmax we need: ", x_max_needed, "  Versus what we got: ", mySpace.xMax_values[0]
dx_needed_2_interactions = (mySpace.hbar * np.pi) / (nuclear_ground_1.omega * nuclear_ground_1.mass * (2.0 * dx_1 + 4.0 *  nuclear_ground_1.sigma))
print "dx needed for 2 interactions: ", dx_needed_2_interactions, "but we have: ", mySpace.Dx_values[0]



lowestEnergyTransitionHO = nuclear_ground_1.energyEigenvalue(0) -  nuclear_ground_1.energyEigenvalue(0)

mostImportantTransitionHO_wavenumbers = 1875.0
mostImportantTransitionHO = mySpace.unitHandler.energyUnitsFromWavenumbers(mostImportantTransitionHO_wavenumbers)

mostImportantTransitionMorse_wavenumbers = 3100.0
mostImportantTransitionMorse = mySpace.unitHandler.energyUnitsFromWavenumbers(  mostImportantTransitionMorse_wavenumbers  )

#HARMONIC OSCILLATOR   
electronic_ground = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [nuclear_ground_1] )

electronic_excited = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [nuclear_excited_1 ] )


groundStateNuclearWF = NuclearWavefunction.nuclearWavefunction(mySpace, groundStateNuclearHamiltonian = electronic_ground )


ElectronicHamiltonian = ElectronicOperator.ElectronicHamiltonian(mySpace, [(0,0, electronic_ground), 
                                                            (1,1, electronic_excited)])
                                                            
                                                            
initialEWF = ElectronicWavefunction.electronicWavefunction(mySpace, 
                       listOfNuclearWavefunctions = [groundStateNuclearWF, 0],
                       Normalize=True)
                       
#MORSE   

#calculated using def2-TZVP basis and PBE0 function
#Provided by Dimitrij Rappaport
muGtoEValueConstant = -.165270   #e * r_b
muGtoEValueConstant = mySpace.unitHandler.lengthUnitsFromBohrRadii(muGtoEValueConstant)
muGtoEValueLinearCo = -.0548235  #e * r_b / r_b


muEtoGValueConstant = -.132477   #e * r_b
muEtoGValueConstant = mySpace.unitHandler.lengthUnitsFromBohrRadii(muEtoGValueConstant)
muEtoGValueLinearCo = -.04555 #e * r_b / r_b

mu_0 = (muGtoEValueConstant + muEtoGValueConstant ) / 2.0

test_dipole  = NuclearOperator.functionalPositionNuclearOperator(mySpace, lambda x: np.sin(10 * (x - groundCenter)  ))
test_op = ElectronicOperator.ElectronicPositionOperator(mySpace, [(0, 1, test_dipole), (1, 0, test_dipole)])



constantMu = NuclearOperator.constantPositionNuclearOperator(mySpace, mu_0)
linearMu = NuclearOperator.functionalPositionNuclearOperator(mySpace, lambda x: mu_0 + (x - excitedCenter) * muEtoGValueLinearCo + (x - groundCenter) * muGtoEValueLinearCo)


xTransitionDipole_FC = ElectronicOperator.ElectronicPositionOperator(mySpace, [(0, 1, constantMu), (1, 0, constantMu)])
yTransitionDipole_FC = ElectronicOperator.ElectronicPositionOperator(mySpace, [])
zTransitionDipole_FC = ElectronicOperator.ElectronicPositionOperator(mySpace, [])

transitionDipoleTuple_FC = (xTransitionDipole_FC, yTransitionDipole_FC, zTransitionDipole_FC)


xTransitionDipole_linear = ElectronicOperator.ElectronicPositionOperator(mySpace, [(0, 1, linearMu ), (1, 0, linearMu)])
yTransitionDipole_linear = ElectronicOperator.ElectronicPositionOperator(mySpace, [])
zTransitionDipole_linear = ElectronicOperator.ElectronicPositionOperator(mySpace, [])

transitionDipoleTuple_linear = (xTransitionDipole_linear, yTransitionDipole_linear, zTransitionDipole_linear)

    
