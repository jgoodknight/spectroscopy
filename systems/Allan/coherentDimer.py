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


phi = np.pi / 4.0

omega_0_wavenumbers = 640.0
omega_1_e_wavenumbers = 1.5 * omega_0_wavenumbers
omega_2_e_wavenumbers = 2.0 * omega_0_wavenumbers

J_wavenumbers = 1.0 * omega_0_wavenumbers

#Disp: [0.2000 0.1000]
HR_1 = 0.02
HR_2 = 0.005

constant_E_1_wavenumbers = -.50 * omega_0_wavenumbers
constant_E_2_wavenumbers = -1.231 * omega_0_wavenumbers

#constant_E_1_wavenumbers = -.0* omega_0_wavenumbers
#constant_E_2_wavenumbers = -.73 * omega_0_wavenumbers

#constant_E_1_wavenumbers = .0* omega_0_wavenumbers
#constant_E_2_wavenumbers = .731 * omega_0_wavenumbers

mass_1_amu = 63.466


pulse_carrier_frequency_wavenumbers = 1105.0
pulse_carrier_frequency_wavenumbers = 0.0

##all units from here on out are defined by mySpace
mySpace = Spacetime.Spacetime(xMax_tuple = (25.0,25.0),
             numberOfNuclearDimenions = 2,
             numberOfElectronicDimensions = 4,
             numberOfSimulationSpacePointsPerNuclearDimension_tuple = (16, 16),
             dt_SECONDS = .400E-15,
             UnityMassInElectronMasses = 100.0)

omega_0 = mySpace.unitHandler.energyUnitsFromWavenumbers(omega_0_wavenumbers)

omega_1_e = mySpace.unitHandler.energyUnitsFromWavenumbers(omega_1_e_wavenumbers)
omega_2_e = mySpace.unitHandler.energyUnitsFromWavenumbers(omega_2_e_wavenumbers)

constant_E_1 = mySpace.unitHandler.energyUnitsFromWavenumbers(constant_E_1_wavenumbers) 
constant_E_2 = mySpace.unitHandler.energyUnitsFromWavenumbers(constant_E_2_wavenumbers)

m_1 = mySpace.unitHandler.massUnitsFromAmu(mass_1_amu)

pulse_carrier_frequency = mySpace.unitHandler.energyUnitsFromWavenumbers(pulse_carrier_frequency_wavenumbers)
J = mySpace.unitHandler.energyUnitsFromWavenumbers(J_wavenumbers)

dx_1 = np.sqrt(HR_1 * 2.0 * mySpace.hbar / (m_1 * omega_0))
dx_2 = np.sqrt(HR_2 * 2.0 * mySpace.hbar / (m_1 * omega_0))

x_0 = 0.0
groundCenter = x_0
excitedCenter_1 = x_0 + dx_1
excitedCenter_2 = x_0 + dx_2


nuclear_ground_1 = NuclearOperator.harmonicOscillator(mySpace,
                                                 omega = omega_0,
                                                 mass = m_1,
                                                 center = groundCenter)
nuclear_ground_2 = NuclearOperator.harmonicOscillator(mySpace,
                                                 omega = omega_0,
                                                 mass = m_1,
                                                 center = groundCenter)
                                                 
nuclear_a_1 = NuclearOperator.harmonicOscillator(mySpace,
                                                 omega = omega_1_e,
                                                 mass = m_1,
                                                 center = excitedCenter_1,
                                                 energyOffset = constant_E_1)
nuclear_a_2 = NuclearOperator.harmonicOscillator(mySpace,
                                                 omega = omega_2_e,
                                                 mass = m_1,
                                                 center = groundCenter,
                                                 energyOffset = 0.0)
                                                 
                                                 
nuclear_b_1 = NuclearOperator.harmonicOscillator(mySpace,
                                                 omega = omega_1_e,
                                                 mass = m_1,
                                                 center = groundCenter,
                                                 energyOffset = 0.0)
nuclear_b_2 = NuclearOperator.harmonicOscillator(mySpace,
                                                 omega = omega_2_e,
                                                 mass = m_1,
                                                 center = excitedCenter_2,
                                                 energyOffset = constant_E_2)
                                                 
                                                 




x_max_needed_1 = 2 * dx_1 + 4.0 *  nuclear_ground_1.sigma
x_max_needed_2 = 2 * dx_2 + 4.0 *  nuclear_ground_2.sigma
print "xmaxes we need: ", (x_max_needed_1, x_max_needed_2), "  Versus what we got: ", mySpace.xMax_values
dx_needed_2_interactions_1 = (mySpace.hbar * np.pi) / (nuclear_ground_1.omega * nuclear_ground_1.mass * (2.0 * dx_1 + 4.0 *  nuclear_ground_1.sigma))
dx_needed_2_interactions_2 = (mySpace.hbar * np.pi) / (nuclear_ground_2.omega * nuclear_ground_2.mass * (2.0 * dx_2 + 4.0 *  nuclear_ground_2.sigma))
print "dx needed for 2 interactions: ", (dx_needed_2_interactions_1, dx_needed_2_interactions_2), "but we have: ", mySpace.Dx_values

electronic_ground_ground = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [nuclear_ground_1, nuclear_ground_2] )

electronic_excited_ground = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [nuclear_a_1, nuclear_a_2 ] )
electronic_ground_excited = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [nuclear_b_1, nuclear_b_2 ] )
electronic_excited_excited = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [nuclear_a_1, nuclear_b_2 ] )

J_coupling = NuclearOperator.constantPositionNuclearOperator(mySpace, J)


groundStateNuclearWF = NuclearWavefunction.nuclearWavefunction(mySpace, groundStateNuclearHamiltonian = electronic_ground_ground )


ElectronicHamiltonian = ElectronicOperator.ElectronicHamiltonian(mySpace, [(0,0, electronic_ground_ground),
                                                            (1,1, electronic_ground_excited),
                                                            (2,2, electronic_excited_ground),
                                                            (3,3, electronic_excited_excited),
                                                            (1,2, J_coupling),
                                                            (2,1, J_coupling)])


initialEWF = ElectronicWavefunction.electronicWavefunction(mySpace,
                       listOfNuclearWavefunctions = [groundStateNuclearWF, 0, 0, 0],
                       Normalize=True)



mu_0 = 1.0

x_max = mySpace.xMax_values[0]


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

#linearMu = NuclearOperator.functionalPositionNuclearOperator(mySpace, lambda x: mu_0 + c_natural * (x- x_max)/HR_1 )
#linearMu = NuclearOperator.functionalPositionNuclearOperator(mySpace, lambda x, y: mu_0 + c_natural * (x)/dx_1 + c_natural * (y)/dx_1 )
#
#
#xTransitionDipole_linear = ElectronicOperator.ElectronicPositionOperator(mySpace, [(0, 1, linearMu), 
#                                                                               (1, 0, linearMu), 
#                                                                               (2, 0, linearMu), 
#                                                                               (0, 2, linearMu), 
#                                                                               (1, 3, linearMu), 
#                                                                               (3, 1, linearMu), 
#                                                                               (2, 3, linearMu), 
#                                                                               (3, 2, linearMu)])
#yTransitionDipole_linear = ElectronicOperator.ElectronicPositionOperator(mySpace, [])
#zTransitionDipole_linear = ElectronicOperator.ElectronicPositionOperator(mySpace, [])
#
#transitionDipoleTuple_linear = (xTransitionDipole_linear, yTransitionDipole_linear, zTransitionDipole_linear)

#def transitionDipoleTuple_gaussian_noisy(center, stdev, seed = 42):
#    np.random.seed(seed)
#    noisyMu = NuclearOperator.constantPositionNuclearOperator(mySpace, center)
#    noisyMu.surface = noisyMu.surface + stdev * np.random.randn(*noisyMu.surface.shape)
#    x = ElectronicOperator.ElectronicPositionOperator(mySpace, [(0, 1, noisyMu ), (1, 0, noisyMu)])
#    y = ElectronicOperator.ElectronicPositionOperator(mySpace, [])
#    z = ElectronicOperator.ElectronicPositionOperator(mySpace, [])
#    return (x, y, z)
#    
#    
#def transitionDipoleTuple_sine_groundCenter(mu_0, amplitude, frequency):
#    sineMu = NuclearOperator.functionalPositionNuclearOperator(mySpace, lambda x: mu_0 + np.sin( frequency * (x - groundCenter) ) )
#    x = ElectronicOperator.ElectronicPositionOperator(mySpace, [(0, 1, sineMu ), (1, 0, sineMu)])
#    y = ElectronicOperator.ElectronicPositionOperator(mySpace, [])
#    z = ElectronicOperator.ElectronicPositionOperator(mySpace, [])
#    return (x, y, z)
#
#frequency_for_half_cycle_between_wells = np.pi / dx_1
