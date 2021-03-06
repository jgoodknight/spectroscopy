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

omega_1_g_wavenumbers = 155.0
omega_1_e_wavenumbers = 155.0
HR_1 = 0.33
mass_1_amu = 63.466

omega_2_g_wavenumbers = 1334.0
omega_2_e_wavenumbers = 1334.0
HR_2 = 0.44
mass_2_amu = 63.466


#S = dx^2 m \omega / (2 \hbar)



##all units from here on out are defined by mySpace
mySpace = Spacetime.Spacetime(xMax_tuple = (65.0, 25.0),
             numberOfNuclearDimenions = 2,
             numberOfElectronicDimensions = 2,
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


m_1 = mySpace.unitHandler.massUnitsFromAmu(mass_1_amu)
m_2 = mySpace.unitHandler.massUnitsFromAmu(mass_2_amu)

pulse_carrier_frequency_wavenumbers = 0.0
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




electronic_ground = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [nuclear_ground_1, nuclear_ground_2] )

electronic_excited = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [nuclear_excited_1, nuclear_excited_2] )


groundStateNuclearWF = NuclearWavefunction.nuclearWavefunction(mySpace, groundStateNuclearHamiltonian = electronic_ground )



ElectronicHamiltonian = ElectronicOperator.ElectronicHamiltonian(mySpace, [(0,0, electronic_ground),
                                                            (1,1, electronic_excited)])


initialEWF = ElectronicWavefunction.electronicWavefunction(mySpace,
                       listOfNuclearWavefunctions = [groundStateNuclearWF, 0],
                       Normalize=True)




mu_0 = 1.0


constantMu = NuclearOperator.constantPositionNuclearOperator(mySpace, mu_0)

xTransitionDipole_FC = ElectronicOperator.ElectronicPositionOperator(mySpace, [(0, 1, constantMu), (1, 0, constantMu)])
yTransitionDipole_FC = ElectronicOperator.ElectronicPositionOperator(mySpace, [])
zTransitionDipole_FC = ElectronicOperator.ElectronicPositionOperator(mySpace, [])

transitionDipoleTuple_FC = (xTransitionDipole_FC, yTransitionDipole_FC, zTransitionDipole_FC)
