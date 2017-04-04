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

mySpace = Spacetime.Spacetime(xMax = 10.0,
             numberOfNuclearDimenions = 2,
             numberOfElectronicDimensions = 4,
             numberOfSimulationSpacePointsPerNuclearDimension = 100,
             dt = .05,
             wavenumberBasedUnits_scaling = 100.0)
             

E_00 = mySpace.unitHandler.energyUnitsFromWavenumbers(0.0)  #joel-energy-units 
omega_c = mySpace.unitHandler.energyUnitsFromWavenumbers(12500.0) #joel-energy-units 
E_10 = omega_c - mySpace.unitHandler.energyUnitsFromWavenumbers(300.0) #joel-energy-units 
E_01 = omega_c - mySpace.unitHandler.energyUnitsFromWavenumbers(200.0) #joel-energy-units 

J = mySpace.unitHandler.energyUnitsFromWavenumbers(100.0) #joel-energy-units

#first coordinate
omega_00x = omega_00y =  mySpace.unitHandler.energyUnitsFromWavenumbers(100.0) #joel-energy-units 
omega_10x = omega_01x =  mySpace.unitHandler.energyUnitsFromWavenumbers(200.0) #joel-energy-units 
omega_10y = omega_01y =  mySpace.unitHandler.energyUnitsFromWavenumbers(150.0) #joel-energy-units 

delta_00x = 0
delta_00y = 0

delta_10x = mySpace.unitHandler.energyUnitsFromWavenumbers(50.0) #joel-distance-units 
delta_10y = 0

#not used here
delta_01x = 0
delta_01y = mySpace.unitHandler.energyUnitsFromWavenumbers(50.0)

pulse_width_stdev = 18.7 / (2.0 *np.sqrt(2.0 * np.log(2.0))) * 1.0E-15 #femtoseconds
print "Coherent Dimer pulse stdev", pulse_width_stdev, "seconds"
pulse_width_stdev =  mySpace.unitHandler.timeUnitsFromSeconds(pulse_width_stdev)  #joel-time-units

             
xOscillatorGround = NuclearOperator.harmonicOscillator(mySpace, 
                                                         omega = omega_00x, 
                                                         center = delta_00x, 
                                                         energyOffset = E_00)
yOscillatorGround = NuclearOperator.harmonicOscillator(mySpace, 
                                                         omega = omega_00y, 
                                                         center = delta_00y, 
                                                         energyOffset = E_00)

xOscillatorExcited = NuclearOperator.harmonicOscillator(mySpace, 
                                                         omega = omega_10x, 
                                                         center = delta_10x, 
                                                         energyOffset = E_01) #make this energy offset 0 and give it all to the other
yOscillatorExcited = NuclearOperator.harmonicOscillator(mySpace, 
                                                         omega = omega_10y, 
                                                         center = delta_10y, 
                                                         energyOffset = E_10)


groundGroundStateNuclearHamiltonian = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [xOscillatorGround, yOscillatorGround] )
groundStateNuclearHamiltonian = groundGroundStateNuclearHamiltonian

excitedGroundStateNuclearHamiltonian = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [xOscillatorExcited, yOscillatorGround] )

groundExcitedStateNuclearHamiltonian = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [xOscillatorGround, yOscillatorExcited] )

exctiedExcitedStateNuclearHamiltonian = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [xOscillatorExcited, yOscillatorExcited] )

electronicCouplingNuclearOperator = NuclearOperator.constantPositionNuclearOperator(mySpace, J)

groundStateNuclearWavefunction = NuclearWavefunction.nuclearWavefunction(mySpace, groundStateNuclearHamiltonian = groundGroundStateNuclearHamiltonian )


electronicHamiltonian = ElectronicOperator.ElectronicHamiltonian(mySpace, [(0,0, groundGroundStateNuclearHamiltonian), 
                                                                            (1,1, excitedGroundStateNuclearHamiltonian),
                                                                            (1,2, electronicCouplingNuclearOperator),
                                                                            (2,1, electronicCouplingNuclearOperator), 
                                                                            (2,2, groundExcitedStateNuclearHamiltonian), 
                                                                            (3,3, exctiedExcitedStateNuclearHamiltonian)])
                                                            
                                                            
initialElectronicWavefunction = ElectronicWavefunction.electronicWavefunction(mySpace, 
                                                                               listOfNuclearWavefunctions = [groundStateNuclearWavefunction, 0, 0, 0],
                                                                               Normalize=True)



                     

xTransitionDipoleValue = 1.0
yTransitionDipoleValue = 3.0

xTransitionDipoleNuclearOperator = NuclearOperator.constantPositionNuclearOperator(mySpace, xTransitionDipoleValue)
yTransitionDipoleNuclearOperator = NuclearOperator.constantPositionNuclearOperator(mySpace, yTransitionDipoleValue)


xTransitionDipole = ElectronicOperator.ElectronicPositionOperator(mySpace, [(0, 1, xTransitionDipoleNuclearOperator), 
                                                                            (1, 0, xTransitionDipoleNuclearOperator), 
                                                                            (3, 2, xTransitionDipoleNuclearOperator), 
                                                                            (2, 3, xTransitionDipoleNuclearOperator)])
yTransitionDipole = ElectronicOperator.ElectronicPositionOperator(mySpace, [(0, 2, xTransitionDipoleNuclearOperator), 
                                                                            (2, 0, xTransitionDipoleNuclearOperator), 
                                                                            (3, 1, xTransitionDipoleNuclearOperator), 
                                                                            (1, 3, xTransitionDipoleNuclearOperator)])
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
noiseCovariance = noiseCorrelation * noiseSTDEV**2

noiseMean = np.array([0.0, 0.0])
noiseCovarianceMatrix = np.array([[noiseVariance, noiseCovariance],[noiseCovariance, noiseSTDEV ]])
