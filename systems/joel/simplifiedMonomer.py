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
import spectroscopy.TimeFunction as TimeFunction


energyScale = 100 #wavenumbers
timeScale   = 53.09 #femtoseconds

mySpace = Spacetime.Spacetime(xMax = 6.0,
             numberOfNuclearDimenions = 1,
             numberOfElectronicDimensions = 2,
             numberOfSimulationSpacePointsPerNuclearDimension = 100,
             dt = .03,
             wavenumberBasedUnits_scaling = 90.0)
             

E_00 = mySpace.unitHandler.energyUnitsFromWavenumbers(0.0)  #joel-energy-units 
omega_c = mySpace.unitHandler.energyUnitsFromWavenumbers(12500.0) #joel-energy-units 
E_10 = omega_c - mySpace.unitHandler.energyUnitsFromWavenumbers(125.0) #joel-energy-units 

#first coordinate
omega_00x = omega_00y =  mySpace.unitHandler.energyUnitsFromWavenumbers(100) #joel-energy-units 
omega_10x = omega_01x =  mySpace.unitHandler.energyUnitsFromWavenumbers(200) #joel-energy-units 
omega_10y = omega_01y =  mySpace.unitHandler.energyUnitsFromWavenumbers(150) #joel-energy-units 

delta_00x = -mySpace.unitHandler.energyUnitsFromWavenumbers(100) #joel-distance-units 
delta_00y = 0

delta_10x = 0.0
delta_10y = 0

#not used here
delta_01x = 0
delta_01y = 0

pulse_width_stdev_femtoseconds = 10.0 / (2.0 *np.sqrt(2.0 * np.log(2.0))) * 1.0E-15 #femtoseconds
pulse_width_stdev =  mySpace.unitHandler.timeUnitsFromSeconds(pulse_width_stdev_femtoseconds)  #joel-time-units

             
xOscillatorGround = NuclearOperator.harmonicOscillator(mySpace, 
                                                         omega = omega_00x, 
                                                         center = delta_00x, 
                                                         energyOffset = E_00)

xOscillatorExcited = NuclearOperator.harmonicOscillator(mySpace, 
                                                         omega = omega_10x, 
                                                         center = delta_10x, 
                                                         energyOffset = E_10) #make this energy offset 0 and give it all to the other

  
groundStateNuclearHamiltonian = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [xOscillatorGround] )

excitedStateNuclearHamiltonian = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [xOscillatorExcited] )


groundStateNuclearWavefunction = NuclearWavefunction.nuclearWavefunction(mySpace, groundStateNuclearHamiltonian = groundStateNuclearHamiltonian )


electronicHamiltonian = ElectronicOperator.ElectronicHamiltonian(mySpace, [(0,0, groundStateNuclearHamiltonian), 
                                                            (1,1, excitedStateNuclearHamiltonian)])
                                                            
                                                            
initialElectronicWavefunction = ElectronicWavefunction.electronicWavefunction(mySpace, 
                                                                               listOfNuclearWavefunctions = [groundStateNuclearWavefunction, 0],
                                                                               Normalize=True)



                     

xTransitionDipoleValue = 1.0
yTransitionDipoleValue = 3.0

xTransitionDipoleNuclearOperator = NuclearOperator.constantPositionNuclearOperator(mySpace, xTransitionDipoleValue)
yTransitionDipoleNuclearOperator = NuclearOperator.constantPositionNuclearOperator(mySpace, yTransitionDipoleValue)


xTransitionDipole = ElectronicOperator.ElectronicPositionOperator(mySpace, [(0, 1, xTransitionDipoleNuclearOperator), (1, 0, xTransitionDipoleNuclearOperator)])
yTransitionDipole = ElectronicOperator.ElectronicPositionOperator(mySpace, [(0, 1, yTransitionDipoleNuclearOperator), (1, 0, yTransitionDipoleNuclearOperator)])
zTransitionDipole = ElectronicOperator.ElectronicPositionOperator(mySpace, [])



xPump = TimeFunction.GaussianCosinePulse(mySpace, 
                                         centerOmega = omega_c, 
                                         timeSpread = pulse_width_stdev)

yPump = TimeFunction.GaussianCosinePulse(mySpace, 
                                         centerOmega = omega_c, 
                                         timeSpread = pulse_width_stdev)
                                         
zPump = TimeFunction.zeroTimeFunction(mySpace)
    
    
pumpBeamTuple = (xPump, yPump, zPump)

transitionDipoleTuple = (xTransitionDipole, yTransitionDipole, zTransitionDipole)

maxFrequency =  mySpace.unitHandler.energyUnitsFromWavenumbers(12500.0 + 3000.0)
frequencyResolution =  mySpace.unitHandler.energyUnitsFromWavenumbers(1.0)

pulseSpectrumFrequencies, pulseSpectrumValues = xPump.fourierTransform(maxFrequency, frequencyResolution)
pulseSpectrumFrequencies = mySpace.unitHandler.wavenumbersFromEnergyUnits(pulseSpectrumFrequencies)    


noiseSTDEV =  mySpace.unitHandler.energyUnitsFromWavenumbers(40.0)
noiseVariance = noiseSTDEV**2
noiseCorrelation = 0.8
noiseCovariance = noiseCorrelation * noiseVariance


noiseMean = np.array([[0.0]])
noiseCovarianceMatrix = np.array([[noiseVariance]])

