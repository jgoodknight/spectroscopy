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


omega_1_g_wavenumbers = 1334.0
omega_1_e_wavenumbers = 1334.0
HR_1 = 0.44
mass_1_amu = 63.466

#omega_2_g_wavenumbers = 1334.0
#omega_2_e_wavenumbers = 1334.0
#HR_2 = 0.44
#mass_2_amu = 63.466


#S = dx^2 m \omega / (2 \hbar)

##all units from here on out are defined by mySpace
mySpace = Spacetime.Spacetime(xMax_tuple = (23.0,),
             numberOfNuclearDimenions = 1,
             numberOfElectronicDimensions = 2,
             numberOfSimulationSpacePointsPerNuclearDimension_tuple = (24,),
             dt_SECONDS = .100E-15,
             UnityMassInElectronMasses = 100.0)
#the following parameters allow for a converged TG/PP signal.  Not tested with 2DES, though
#mySpace = Spacetime.Spacetime(xMax_tuple = (23.0,),
#             numberOfNuclearDimenions = 1,
#             numberOfElectronicDimensions = 2,
#             numberOfSimulationSpacePointsPerNuclearDimension_tuple = (24,),
#             dt_SECONDS = 1.00E-15,
#             UnityMassInElectronMasses = 100.0)

omega_1_g = mySpace.unitHandler.energyUnitsFromWavenumbers(omega_1_g_wavenumbers)
omega_1_e = mySpace.unitHandler.energyUnitsFromWavenumbers(omega_1_e_wavenumbers)

m_1 = mySpace.unitHandler.massUnitsFromAmu(mass_1_amu)

pulse_carrier_frequency_wavenumbers = 0.0
pulse_carrier_frequency = mySpace.unitHandler.energyUnitsFromWavenumbers(pulse_carrier_frequency_wavenumbers)


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

electronic_ground = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [nuclear_ground_1] )

electronic_excited = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [nuclear_excited_1 ] )


groundStateNuclearWF = NuclearWavefunction.nuclearWavefunction(mySpace, groundStateNuclearHamiltonian = electronic_ground )


ElectronicHamiltonian = ElectronicOperator.ElectronicHamiltonian(mySpace, [(0,0, electronic_ground),
                                                            (1,1, electronic_excited)])


initialEWF = ElectronicWavefunction.electronicWavefunction(mySpace,
                       listOfNuclearWavefunctions = [groundStateNuclearWF, 0],
                       Normalize=True)



mu_0 = 1.0

mu_0 = 1.0
x_max = mySpace.xMax_values[0]
c = 1E-3


constantMu = NuclearOperator.constantPositionNuclearOperator(mySpace, mu_0)

xTransitionDipole_FC = ElectronicOperator.ElectronicPositionOperator(mySpace, [(0, 1, constantMu), (1, 0, constantMu)])
yTransitionDipole_FC = ElectronicOperator.ElectronicPositionOperator(mySpace, [])
zTransitionDipole_FC = ElectronicOperator.ElectronicPositionOperator(mySpace, [])

transitionDipoleTuple_FC = (xTransitionDipole_FC, yTransitionDipole_FC, zTransitionDipole_FC)

linearMu = NuclearOperator.functionalPositionNuclearOperator(mySpace, lambda x: mu_0 + c * (x- x_max) )



xTransitionDipole_linear = ElectronicOperator.ElectronicPositionOperator(mySpace, [(0, 1, linearMu ), (1, 0, linearMu)])
yTransitionDipole_linear = ElectronicOperator.ElectronicPositionOperator(mySpace, [])
zTransitionDipole_linear = ElectronicOperator.ElectronicPositionOperator(mySpace, [])

transitionDipoleTuple_linear = (xTransitionDipole_linear, yTransitionDipole_linear, zTransitionDipole_linear)
