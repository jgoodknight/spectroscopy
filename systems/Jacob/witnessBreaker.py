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

omega_g_wavenumbers = 200.0
omega_e_wavenumbers = omega_g_wavenumbers * .75
D_e_wavenumbers = 2.0 * omega_e_wavenumbers

#HUANG_RHYS = .01
HUANG_RHYS = .40

mass_amu = reducedMassIodineAMU

#S = dx^2 m \omega / (2 \hbar)

##all units from here on out are defined by mySpace
mySpace = Spacetime.Spacetime(xMax = 300.0,
             numberOfNuclearDimenions = 1,
             numberOfElectronicDimensions = 2,
             numberOfSimulationSpacePointsPerNuclearDimension = 240,
             dt_SECONDS = .500E-15, 
             UnityMassInElectronMasses = 100.0)
             
omega_g = mySpace.unitHandler.energyUnitsFromWavenumbers(omega_g_wavenumbers)
omega_e = mySpace.unitHandler.energyUnitsFromWavenumbers(omega_e_wavenumbers)

D_e = mySpace.unitHandler.energyUnitsFromWavenumbers(D_e_wavenumbers)

m = mySpace.unitHandler.massUnitsFromAmu(mass_amu)

pulse_carrier_frequency_wavenumbers = 44.0
pulse_carrier_frequency = mySpace.unitHandler.energyUnitsFromWavenumbers(pulse_carrier_frequency_wavenumbers)

a = omega_e * np.sqrt(m / (2.0 * D_e))

dx = np.sqrt(HUANG_RHYS * 2.0 * mySpace.hbar / (m * omega_e))

x_0 = -100.0

S = dx**2 * m * omega_e / (2.0 * mySpace.hbar)


groundCenter = x_0
excitedCenter = x_0 - dx
ground_ho = NuclearOperator.harmonicOscillator(mySpace, 
                                                 omega=omega_g, 
                                                 mass=m,
                                                 center= groundCenter)
                                                 
excited_morse = NuclearOperator.morsePotential(mySpace,
                                                     a= a, 
                                                     De= D_e, 
                                                     mass=m, 
                                                     center=excitedCenter)
excited_ho = NuclearOperator.harmonicOscillator(mySpace,
                                                     omega = omega_e, 
                                                     mass=m, 
                                                     center=excitedCenter)
      


#HARMONIC OSCILLATOR   
electronic_ground = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [ground_ho] )

electronic_excited = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [excited_ho ] )


groundStateNuclearWF = NuclearWavefunction.nuclearWavefunction(mySpace, groundStateNuclearHamiltonian = electronic_ground )


ElectronicHamiltonian = ElectronicOperator.ElectronicHamiltonian(mySpace, [(0,0, electronic_ground), 
                                                            (1,1, electronic_excited)])
                                                            
                                                            
initialEWF = ElectronicWavefunction.electronicWavefunction(mySpace, 
                       listOfNuclearWavefunctions = [groundStateNuclearWF, 0],
                       Normalize=True)
                


mu_0 = 1.0

c = 100


constantMuGtoE = NuclearOperator.constantPositionNuclearOperator(mySpace, mu_0)
c100_muOperator = NuclearOperator.functionalPositionNuclearOperator(mySpace, lambda x: mu_0 + ( c / 2.0) * ( (x - groundCenter) / dx + (x - excitedCenter) / dx  ))
c100__n2_muOperator = NuclearOperator.functionalPositionNuclearOperator(mySpace, lambda x: mu_0 + ( c / 2.0) * ( (x - groundCenter)**2 / dx**2 + (x - excitedCenter)**2 / dx**2  ))


xTransitionDipole_FC = ElectronicOperator.ElectronicPositionOperator(mySpace, [(0, 1, constantMuGtoE), (1, 0, constantMuGtoE)])
yTransitionDipole_FC = ElectronicOperator.ElectronicPositionOperator(mySpace, [(0, 1, constantMuGtoE), (1, 0, constantMuGtoE)])
zTransitionDipole_FC = ElectronicOperator.ElectronicPositionOperator(mySpace, [])

transitionDipoleTuple_FC = (xTransitionDipole_FC, yTransitionDipole_FC, zTransitionDipole_FC)


xTransitionDipole_c100 = ElectronicOperator.ElectronicPositionOperator(mySpace, [(0, 1, c100_muOperator), (1, 0, c100_muOperator)])
yTransitionDipole_c100 = ElectronicOperator.ElectronicPositionOperator(mySpace, [(0, 1, c100_muOperator), (1, 0, c100_muOperator)])
zTransitionDipole_c100 = ElectronicOperator.ElectronicPositionOperator(mySpace, [])

transitionDipoleTuple_c100 = (xTransitionDipole_c100, yTransitionDipole_c100, zTransitionDipole_c100)


xTransitionDipole_c100_n2 = ElectronicOperator.ElectronicPositionOperator(mySpace, [(0, 1, c100__n2_muOperator), (1, 0, c100__n2_muOperator)])
yTransitionDipole_c100_n2 = ElectronicOperator.ElectronicPositionOperator(mySpace, [(0, 1, c100__n2_muOperator), (1, 0, c100__n2_muOperator)])
zTransitionDipole_c100_n2 = ElectronicOperator.ElectronicPositionOperator(mySpace, [])

transitionDipoleTuple_c100_n2 = (xTransitionDipole_c100_n2, yTransitionDipole_c100_n2, zTransitionDipole_c100_n2)