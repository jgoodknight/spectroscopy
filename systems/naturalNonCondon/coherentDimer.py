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


omega_1_g_wavenumbers = 640.0
omega_1_e_wavenumbers = 640.0

J_wavenumbers = 1.0 * omega_1_e_wavenumbers

HR_1 = 0.4
mass_1_amu = 63.466

c_natural = .20

pulse_carrier_frequency_wavenumbers = 761.74

##all units from here on out are defined by mySpace
mySpace = Spacetime.Spacetime(xMax_tuple = (35.0,35.0),
             numberOfNuclearDimenions = 2,
             numberOfElectronicDimensions = 4,
             numberOfSimulationSpacePointsPerNuclearDimension_tuple = (28, 28),
             dt_SECONDS = .400E-15,
             UnityMassInElectronMasses = 100.0)

omega_1_g = mySpace.unitHandler.energyUnitsFromWavenumbers(omega_1_g_wavenumbers)
omega_1_e = mySpace.unitHandler.energyUnitsFromWavenumbers(omega_1_e_wavenumbers)

m_1 = mySpace.unitHandler.massUnitsFromAmu(mass_1_amu)

pulse_carrier_frequency = mySpace.unitHandler.energyUnitsFromWavenumbers(pulse_carrier_frequency_wavenumbers)
J = mySpace.unitHandler.energyUnitsFromWavenumbers(J_wavenumbers)

dx_1 = np.sqrt(HR_1 * 2.0 * mySpace.hbar / (m_1 * omega_1_e))

x_0 = 0.0
groundCenter = x_0
excitedCenter = x_0 + dx_1


nuclear_ground_1 = NuclearOperator.harmonicOscillator(mySpace,
                                                 omega = omega_1_g,
                                                 mass = m_1,
                                                 center = groundCenter)
nuclear_excited_1 = NuclearOperator.harmonicOscillator(mySpace,
                                                 omega = omega_1_e,
                                                 mass = m_1,
                                                 center = excitedCenter)




x_max_needed = 2 * dx_1 + 4.0 *  nuclear_ground_1.sigma
print "xmax we need: ", x_max_needed, "  Versus what we got: ", mySpace.xMax_values[0]
dx_needed_2_interactions = (mySpace.hbar * np.pi) / (nuclear_ground_1.omega * nuclear_ground_1.mass * (2.0 * dx_1 + 4.0 *  nuclear_ground_1.sigma))
print "dx needed for 2 interactions: ", dx_needed_2_interactions, "but we have: ", mySpace.Dx_values[0]

electronic_ground_ground = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [nuclear_ground_1, nuclear_ground_1] )

electronic_excited_ground = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [nuclear_excited_1, nuclear_ground_1 ] )
electronic_ground_excited = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [nuclear_ground_1, nuclear_excited_1 ] )
electronic_excited_excited = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [nuclear_excited_1, nuclear_excited_1 ] )

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
linearMu = NuclearOperator.functionalPositionNuclearOperator(mySpace, lambda x, y: mu_0 + c_natural * (x)/dx_1 + c_natural * (y)/dx_1 )


xTransitionDipole_linear = ElectronicOperator.ElectronicPositionOperator(mySpace, [(0, 1, linearMu), 
                                                                               (1, 0, linearMu), 
                                                                               (2, 0, linearMu), 
                                                                               (0, 2, linearMu), 
                                                                               (1, 3, linearMu), 
                                                                               (3, 1, linearMu), 
                                                                               (2, 3, linearMu), 
                                                                               (3, 2, linearMu)])
yTransitionDipole_linear = ElectronicOperator.ElectronicPositionOperator(mySpace, [])
zTransitionDipole_linear = ElectronicOperator.ElectronicPositionOperator(mySpace, [])

transitionDipoleTuple_linear = (xTransitionDipole_linear, yTransitionDipole_linear, zTransitionDipole_linear)

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
