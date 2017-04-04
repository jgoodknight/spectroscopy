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


##advised to keep population time steps at oughly 2fs or less (20 steps per 100fs)
detuning = .9
omega_1_g_wavenumbers = 155.0
omega_1_e_wavenumbers = omega_1_g_wavenumbers * detuning
HR_1 = 0.33
mass_1_amu = 63.466

omega_2_g_wavenumbers = 1334.0
omega_2_e_wavenumbers = omega_2_g_wavenumbers * detuning
HR_2 = 0.44
mass_2_amu = 63.466

J_wavenumbers = (omega_2_e_wavenumbers + omega_1_e_wavenumbers ) / 3.0


#S = dx^2 m \omega / (2 \hbar)



##all units from here on out are defined by mySpace
mySpace = Spacetime.Spacetime(xMax_tuple = (65.0, 25.0),
             numberOfNuclearDimenions = 2,
             numberOfElectronicDimensions = 4,
             numberOfSimulationSpacePointsPerNuclearDimension_tuple = (36, 32),
             dt_SECONDS = .2000E-15,
             UnityMassInElectronMasses = 100.0)

#mySpace = Spacetime.Spacetime(xMax_tuple = (70.0, 27.0),
#             numberOfNuclearDimenions = 2,
#             numberOfElectronicDimensions = 2,
#             numberOfSimulationSpacePointsPerNuclearDimension_tuple = (32, 28),
#             dt_SECONDS = .100E-15,
#             UnityMassInElectronMasses = 100.0)

omega_1_g = mySpace.unitHandler.energyUnitsFromWavenumbers(omega_1_g_wavenumbers)
omega_1_e = mySpace.unitHandler.energyUnitsFromWavenumbers(omega_1_e_wavenumbers)

omega_2_g = mySpace.unitHandler.energyUnitsFromWavenumbers(omega_2_g_wavenumbers)
omega_2_e = mySpace.unitHandler.energyUnitsFromWavenumbers(omega_2_e_wavenumbers)

J = mySpace.unitHandler.energyUnitsFromWavenumbers(J_wavenumbers)


m_1 = mySpace.unitHandler.massUnitsFromAmu(mass_1_amu)
m_2 = mySpace.unitHandler.massUnitsFromAmu(mass_2_amu)

pulse_carrier_frequency_wavenumbers = 403.145944  ##mean of absorption spectrum
pulse_carrier_frequency = mySpace.unitHandler.energyUnitsFromWavenumbers(pulse_carrier_frequency_wavenumbers)


dx_1 = np.sqrt(HR_1 * 2.0 * mySpace.hbar / (m_1 * omega_1_e))
dx_2 = np.sqrt(HR_2 * 2.0 * mySpace.hbar / (m_2 * omega_2_e))


relevant_omega = min(omega_2_e, omega_1_e)
reelvant_dx = min(dx_1, dx_2)
fudge_factor = 4




x_0 = 0.0
groundCenter_1 = x_0
excitedCenter_1 = x_0 + dx_1

groundCenter_2 = x_0
excitedCenter_2 = x_0 + dx_2

nuclear_ground_1 = NuclearOperator.harmonicOscillator(mySpace,
                                                 omega = omega_1_g,
                                                 mass = m_1,
                                                 center = groundCenter_1)
nuclear_excited_1 = NuclearOperator.harmonicOscillator(mySpace,
                                                 omega = omega_1_e,
                                                 mass = m_1,
                                                 center = excitedCenter_1)

nuclear_ground_2 = NuclearOperator.harmonicOscillator(mySpace,
                                                 omega = omega_2_g,
                                                 mass = m_2,
                                                 center = groundCenter_2)
nuclear_excited_2 = NuclearOperator.harmonicOscillator(mySpace,
                                                 omega = omega_2_e,
                                                 mass = m_2,
                                                 center = excitedCenter_2)


x_max_needed_1 = 2 * dx_1 + 4.0 *  nuclear_ground_1.sigma
x_max_needed_2 = 2 * dx_2 + 4.0 *  nuclear_ground_2.sigma
print "xmax we need for 1: ", x_max_needed_1, " versus the actual: ", mySpace.xMax_values[0]
print " xmax needed for 2: ", x_max_needed_2, " versus the actual: ", mySpace.xMax_values[1]
print ""
dx_needed_2_interactions_1 = (mySpace.hbar * np.pi) / (omega_1_g * nuclear_ground_1.mass * (2.0 * dx_1 + 4.0 *  nuclear_ground_1.sigma))

dx_needed_2_interactions_2 = (mySpace.hbar * np.pi) / (omega_2_g * nuclear_ground_2.mass * (2.0 * dx_2 + 4.0 *  nuclear_ground_2.sigma))
print "dx needed for 2 interactions on 1: ", " versus the actual: ", dx_needed_2_interactions_1, mySpace.Dx_values[0]
print " dx needed for 2 on 2: ", dx_needed_2_interactions_2, " versus the actual: ", mySpace.Dx_values[1]




electronic_groundGround = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [nuclear_ground_1, nuclear_ground_2] )
electronic_excitedExcited = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [nuclear_excited_1, nuclear_excited_2] )

electronic_groundExcited = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [nuclear_excited_1, nuclear_ground_2] )
electronic_excitedGround = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [nuclear_excited_1, nuclear_ground_2] )


J_coupling = NuclearOperator.constantPositionNuclearOperator(mySpace, J)

groundStateNuclearWF = NuclearWavefunction.nuclearWavefunction(mySpace, groundStateNuclearHamiltonian = electronic_groundGround )



ElectronicHamiltonian = ElectronicOperator.ElectronicHamiltonian(mySpace, [(0,0, electronic_groundGround),
                                                            (1,1, electronic_groundExcited),
                                                            (2,2, electronic_excitedGround),
                                                            (1,2, J_coupling),
                                                            (2,1, J_coupling),
                                                            (3,3, electronic_excitedExcited)])


initialEWF = ElectronicWavefunction.electronicWavefunction(mySpace,
                       listOfNuclearWavefunctions = [groundStateNuclearWF, 0, 0, 0],
                       Normalize=True)


x_max = mySpace.xMax_values[0]


mu_0 = 1.0


constantMu_a = NuclearOperator.constantPositionNuclearOperator(mySpace, mu_0)
constantMu_b = NuclearOperator.constantPositionNuclearOperator(mySpace, mu_0 / 3.0 )

xTransitionDipole_FC = ElectronicOperator.ElectronicPositionOperator(mySpace, [(0, 1, constantMu_a), 
                                                                               (1, 0, constantMu_a), 
                                                                               (2, 0, constantMu_b), 
                                                                               (0, 2, constantMu_b), 
                                                                               (1, 3, constantMu_b), 
                                                                               (3, 1, constantMu_b), 
                                                                               (2, 3, constantMu_a), 
                                                                               (3, 2, constantMu_a)])
yTransitionDipole_FC = ElectronicOperator.ElectronicPositionOperator(mySpace, [])
zTransitionDipole_FC = ElectronicOperator.ElectronicPositionOperator(mySpace, [])

transitionDipoleTuple_FC = (xTransitionDipole_FC, yTransitionDipole_FC, zTransitionDipole_FC)


