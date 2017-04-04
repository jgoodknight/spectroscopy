# -*- coding: utf-8 -*-
"""
Created on Fri Jan 04 13:23:27 2013

@author: Joey
"""
import copy
import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import numpy.random as random
import Spacetime
import TimeElectronicWavefunction
import ElectronicOperator
import TimeFunction
import NuclearWavefunction
import NuclearOperator
import ElectronicWavefunction

      
class electricDipoleOneInteraction(object):
    "Class which describes an electric Dipole Perturbation which interacts just once with the specified system"
    def __init__(self, space, electronicHamiltonian, MuxMuyMuzElectronicOperatorTuple, ExEyEzTimeFunctionTuple, maxTime, overrideDtForPerturbation = None):
        #store all the variables
        self.mySpace = space
        
        self.MuX, self.MuY, self.MuZ = MuxMuyMuzElectronicOperatorTuple
        self.EX, self.EY, self.EZ = ExEyEzTimeFunctionTuple
        
        self.myElectronicHamiltonian = electronicHamiltonian
        
        self.myPerturbationTimeDependentElectronicWavefunction = TimeElectronicWavefunction.timeElectronicWavefunction(self.mySpace)
        
        if overrideDtForPerturbation is None:
            self.dtPerturbation = self.mySpace.dt
        else:
            self.dtPerturbation = overrideDtForPerturbation
        
        #how many steps must be taken to get the perturbation over with
        self.numberOfTimeStepsToTakeForPerturbation = max(self.EX.maxTime(), self.EY.maxTime(), self.EZ.maxTime()) / float(self.dtPerturbation)
        
        if self.numberOfTimeStepsToTakeForPerturbation < 1:
            self.numberOfTimeStepsToTakeForPerturbation = 1
        else:
            self.numberOfTimeStepsToTakeForPerturbation = int(self.numberOfTimeStepsToTakeForPerturbation) + 1
          
        self.freeEvolutionPropagator = self.myElectronicHamiltonian.myPropagator(overrideDT = self.dtPerturbation)
        
        self.startingTime = min(self.EX.minTime(), self.EY.minTime(), self.EZ.minTime())
        self.startingIndex = int(self.startingTime / self.mySpace.dt)
        
        self.endingTime = max(self.EX.maxTime(), self.EY.maxTime(), self.EZ.maxTime())
        self.endingIndex = int(self.endingTime / self.mySpace.dt)
        
        self.timePerturbationIsOn = self.endingTime - self.startingTime
        
        self.numberOfStepsInPerturbation = np.ceil(self.timePerturbationIsOn / self.mySpace.dt) 
        
        self.maxTime = maxTime 
        self.maxTimeSteps = int(np.round(self.maxTime / self.mySpace.dt))
        
        
    def goFromTimeWavefunction(self, timeWavefunction):
        tStep = 0.0
        t = tStep * self.mySpace.dt
        
        timeWF = timeWavefunction
        perturbedWavefunction = timeWF.zero_copy()
        
        c = -1.0j * self.mySpace.dt / ( 2.0 * self.mySpace.unitHandler.HBAR)
        
        oldAddition = timeWF[tStep] * 0.0
        
        time_series = timeWavefunction.timeSeries
        
        for tStep, underlying_wf in enumerate(timeWavefunction):
            t = time_series[tStep]
            
            if t < self.startingTime:
                continue
            elif t < self.endingTime:
                newAddition = c * (self.perturbationOperatorAtTime(t) * timeWF[tStep])
                
                thingToPropagate = perturbedWavefunction[tStep - 1] + oldAddition
                propagatedAddition = self.freeEvolutionPropagator.APPLY(thingToPropagate)
                
                perturbedWavefunction[tStep] = propagatedAddition  + newAddition
                
                oldAddition = newAddition
            else:
                newEWF = perturbedWavefunction[tStep - 1]
                newEWF = self.freeEvolutionPropagator.APPLY(newEWF)
            
                perturbedWavefunction[tStep] = newEWF
                
        perturbedWavefunction.timeSeries = np.array(time_series)
        
        return perturbedWavefunction
            
    def perturbationOperatorAtTime(self, T):
        return self.EX.valueAtTime(T) * self.MuX + self.EY.valueAtTime(T) * self.MuY + self.EZ.valueAtTime(T) * self.MuZ


        
      
class electricDipole(object):
    """In case you would want it, this calculates the exact wavefunction
    for all possible number of interactions with a dipole moment and electric field
    this is not terribly useful as you would never be able to measure this object
    See chapter on phase matching in my thesis
    
    Accomplishes this by calculating a new propagator for every time step
    
    Actually is this even mathematically correct?"""
    def __init__(self, space, electronicHamiltonian, MuxMuyMuzElectronicOperatorTuple, ExEyEzTimeFunctionTuple, overrideDtForPerturbation = None):
        #store all the variables
        self.mySpace = space
        
        self.MuX, self.MuY, self.MuZ = MuxMuyMuzElectronicOperatorTuple
        self.EX, self.EY, self.EZ = ExEyEzTimeFunctionTuple
        
        self.myElectronicHamiltonian = electronicHamiltonian
        
        self.myPerturbationTimeDependentElectronicWavefunction = TimeElectronicWavefunction.timeElectronicWavefunction(self.mySpace)
        
        if overrideDtForPerturbation is None:
            self.dtPerturbation = self.mySpace.dt
        else:
            self.dtPerturbation = overrideDtForPerturbation
        
        #how many steps must be taken to get the perturbation over with
        self.numberOfTimeStepsToTakeForPerturbation = max(self.EX.maxTime(), self.EY.maxTime(), self.EZ.maxTime()) / float(self.dtPerturbation)
        
        if self.numberOfTimeStepsToTakeForPerturbation < 1:
            self.numberOfTimeStepsToTakeForPerturbation = 1
        else:
            self.numberOfTimeStepsToTakeForPerturbation = int(self.numberOfTimeStepsToTakeForPerturbation) + 1
          
        self.freeEvolutionPropagator = self.myElectronicHamiltonian.myPropagator(overrideDT = self.dtPerturbation)
        
    def propagateInitialWavefunctionToEndOfPerturbation(self, initialWavefunction):
        "Take the given perturbation and evolve out until the perturbation says it's gone"
        self.myPerturbationTimeDependentElectronicWavefunction.allocateSpace(self.numberOfTimeStepsToTakeForPerturbation + 3)
        self.myPerturbationTimeDependentElectronicWavefunction.setInitialWavefunction(initialWavefunction)
        
        electronicHamiltonianMomentumOperator = self.myElectronicHamiltonian.myMomentumOperator
        electronicHamiltonianPositionOperator = self.myElectronicHamiltonian.myPositionOperator
        
        #find the propagator for time zero
        t = self.myPerturbationTimeDependentElectronicWavefunction.currentTime()
        muOperator = self.MuX * self.EX.valueAtTime(t) + self.MuY * self.EY.valueAtTime(t) + self.MuZ * self.EZ.valueAtTime(t)
        newPositionOperator = muOperator + electronicHamiltonianPositionOperator
        formerPropagator = ElectronicOperator.ElectronicPropagator(self.mySpace, 
                                                                   newPositionOperator,  
                                                                   electronicHamiltonianMomentumOperator, 
                                                                   overrideDT= 0.5 * self.dtPerturbation)
        N = int(self.numberOfTimeStepsToTakeForPerturbation)
            
        for perturbationStepNumber in range(N):
            t = self.myPerturbationTimeDependentElectronicWavefunction.currentTime()
            tPlusDt = t + self.dtPerturbation
            
            muOperator = self.MuX * self.EX.valueAtTime(tPlusDt) + self.MuY * self.EY.valueAtTime(tPlusDt) + self.MuZ * self.EZ.valueAtTime(tPlusDt)
            newPositionOperator = muOperator + electronicHamiltonianPositionOperator
            newPropagator = ElectronicOperator.ElectronicPropagator(self.mySpace, 
                                                           newPositionOperator,  
                                                           electronicHamiltonianMomentumOperator, 
                                                           overrideDT= 0.5 * self.dtPerturbation)
            #PROPAGATE
            out = self.myPerturbationTimeDependentElectronicWavefunction.applyOperatorsAndAdvance([formerPropagator, newPropagator])
            
            formerPropagator = newPropagator
            
            self.timeSeries = self.myPerturbationTimeDependentElectronicWavefunction.timeSeries
        return out
            
    def freeEvolution(self, Nsteps):
        "Once the perturbation is turned off, evolve the resultant wavefunction for Nsteps"
        self.myPerturbationTimeDependentElectronicWavefunction.allocateMoreSpace(Nsteps)
        out = self.myPerturbationTimeDependentElectronicWavefunction.applyOperatorsNTimes([self.freeEvolutionPropagator], Nsteps, overrideDT = self.dtPerturbation)
        return out

        
        
    def xEmission(self):
        return self.myPerturbationTimeDependentElectronicWavefunction.timeExpectationValue(self.MuX) * -1.0j
    def yEmission(self):
        return self.myPerturbationTimeDependentElectronicWavefunction.timeExpectationValue(self.MuY) * -1.0j
    def zEmission(self):
        return self.myPerturbationTimeDependentElectronicWavefunction.timeExpectationValue(self.MuZ) * -1.0j
        
        
    def xSpectrum(self):
        return self.mySpace.genericOneDimensionalFourierTransformFromZero(np.array(self.myPerturbationTimeDependentElectronicWavefunction.timeSeries), 
                                                                          -1.0j * self.myPerturbationTimeDependentElectronicWavefunction.timeExpectationValue(self.MuX))
    def ySpectrum(self):
        return self.mySpace.genericOneDimensionalFourierTransformFromZero(np.array(self.myPerturbationTimeDependentElectronicWavefunction.timeSeries), 
                                                                          -1.0j * self.myPerturbationTimeDependentElectronicWavefunction.timeExpectationValue(self.MuY))
    def zSpectrum(self):
        return self.mySpace.genericOneDimensionalFourierTransformFromZero(np.array(self.myPerturbationTimeDependentElectronicWavefunction.timeSeries), 
                                                                          -1.0j * self.myPerturbationTimeDependentElectronicWavefunction.timeExpectationValue(self.MuZ))

    def ElectricFieldOverlapWithOtherTimeFunction(self, timeFunctionTuple, phase = 0.0):
        phaseFactor = np.exp(1.0j * phase) 
        timeSeries = self.myPerturbationTimeDependentElectronicWavefunction.timeSeries
        xEmit = self.xEmission() * phaseFactor
        yEmit = self.yEmission() * phaseFactor
        zEmit = self.zEmission() * phaseFactor
        
        xFunc, yFunc, zFunc = timeFunctionTuple
        overlaps = []
        for timeIndex, tValue in enumerate(timeSeries):
            newOverlap = xEmit[timeIndex] * xFunc.valueAtTime(tValue)
            newOverlap = newOverlap + yEmit[timeIndex] * yFunc.valueAtTime(tValue)
            newOverlap = newOverlap + zEmit[timeIndex] * zFunc.valueAtTime(tValue)
            newOverlap = 2.0 * np.real(newOverlap)
            overlaps.append(newOverlap)
        
        overlaps = np.array(overlaps)
        
        return scipy.integrate.simps(y=overlaps, x=timeSeries)
        
        
class spaceSavingElectricDipole(object):
    
    def __init__(self, space, electronicHamiltonian, MuxMuyMuzElectronicOperatorTuple, ExEyEzTimeFunctionTuple, overrideDtForPerturbation = None):
        #store all the variables
        self.mySpace = space
        
        self.MuX, self.MuY, self.MuZ = MuxMuyMuzElectronicOperatorTuple
        self.EX, self.EY, self.EZ = ExEyEzTimeFunctionTuple
        
        self.myElectronicHamiltonian = electronicHamiltonian
        
        if overrideDtForPerturbation is None:
            self.dtPerturbation = self.mySpace.dt
        else:
            self.dtPerturbation = overrideDtForPerturbation
        
        #how many steps must be taken to get the perturbation over with
        self.numberOfTimeStepsToTakeForPerturbation = max(self.EX.maxTime(), self.EY.maxTime(), self.EZ.maxTime()) / float(self.dtPerturbation)
        
        if self.numberOfTimeStepsToTakeForPerturbation < 1:
            self.numberOfTimeStepsToTakeForPerturbation = 1
        else:
            self.numberOfTimeStepsToTakeForPerturbation = int(self.numberOfTimeStepsToTakeForPerturbation) + 1
          
        self.freeEvolutionPropagator = self.myElectronicHamiltonian.myPropagator()
        
    def propagateInitialWavefunctionToEndOfPerturbation(self, initialWavefunction):
        "Take the given perturbation and evolve out until the perturbation says it's gone"
        
        electronicHamiltonianMomentumOperator = self.myElectronicHamiltonian.myMomentumOperator
        electronicHamiltonianPositionOperator = self.myElectronicHamiltonian.myPositionOperator
        
        #find the propagator for time zero
        t = 0.0
        muOperator = self.MuX * self.EX.valueAtTime(t) + self.MuY * self.EY.valueAtTime(t) + self.MuZ * self.EZ.valueAtTime(t)
        newPositionOperator = muOperator + electronicHamiltonianPositionOperator
        formerPropagator = ElectronicOperator.ElectronicPropagator(self.mySpace, 
                                                                   newPositionOperator,  
                                                                   electronicHamiltonianMomentumOperator, 
                                                                   overrideDT= 0.5 * self.dtPerturbation)
        N = int(self.numberOfTimeStepsToTakeForPerturbation)
        
        self.currentWF = initialWavefunction     
        self.__xEmission = [self.currentWF.overlap(self.MuX * self.currentWF)]
        self.__yEmission = [self.currentWF.overlap(self.MuY * self.currentWF)]
        self.__zEmission = [self.currentWF.overlap(self.MuZ * self.currentWF)]
        
        self.timeSeries = [t]
        
        
        for perturbationStepNumber in range(N):
            t = self.timeSeries[-1]
            tPlusDt = t + self.dtPerturbation
            self.timeSeries.append(tPlusDt)
            
            muOperator = self.MuX * self.EX.valueAtTime(tPlusDt) + self.MuY * self.EY.valueAtTime(tPlusDt) + self.MuZ * self.EZ.valueAtTime(tPlusDt)
            newPositionOperator = muOperator + electronicHamiltonianPositionOperator
            newPropagator = ElectronicOperator.ElectronicPropagator(self.mySpace, 
                                                           newPositionOperator,  
                                                           electronicHamiltonianMomentumOperator, 
                                                           overrideDT= 0.5 * self.dtPerturbation)
            #PROPAGATE
            self.currentWF = formerPropagator.APPLY(self.currentWF)
            self.currentWF = newPropagator.APPLY(self.currentWF)
            
            xEmissionCurrent = self.currentWF.overlap(self.MuX * self.currentWF)
            self.__xEmission.append(xEmissionCurrent)
            yEmissionCurrent = self.currentWF.overlap(self.MuY * self.currentWF)
            self.__yEmission.append(yEmissionCurrent)
            zEmissionCurrent = self.currentWF.overlap(self.MuZ * self.currentWF)
            self.__zEmission.append(zEmissionCurrent)
            
            formerPropagator = newPropagator
        return self.currentWF
            
    def freeEvolution(self, Nsteps):
        "Once the perturbation is turned off, evolve the resultant wavefunction for Nsteps"
        for ii in range(Nsteps):
            t = self.timeSeries[-1]
            tPlusDt = t + self.dtPerturbation
            self.timeSeries.append(tPlusDt)
            
            self.currentWF = self.freeEvolutionPropagator.APPLY(self.currentWF)
            
            xEmissionCurrent = self.currentWF.overlap(self.MuX * self.currentWF)
            self.__xEmission.append(xEmissionCurrent)
            yEmissionCurrent = self.currentWF.overlap(self.MuY * self.currentWF)
            self.__yEmission.append(yEmissionCurrent)
            zEmissionCurrent = self.currentWF.overlap(self.MuZ * self.currentWF)
            self.__zEmission.append(zEmissionCurrent)
        return self.currentWF

        
        
    def xEmission(self):
        return np.array(self.__xEmission, dtype=np.complex) * -1.0j
    def yEmission(self):
        return np.array(self.__yEmission, dtype=np.complex) * -1.0j
    def zEmission(self):
        return np.array(self.__zEmission, dtype=np.complex) * -1.0j
        
        
    def xSpectrum(self):
        return self.mySpace.genericOneDimensionalFourierTransformFromZero(np.array(self.timeSeries), 
                                                                          self.xEmission())
    def ySpectrum(self):
        return self.mySpace.genericOneDimensionalFourierTransformFromZero(np.array(self.timeSeries), 
                                                                          self.yEmission())
    def zSpectrum(self):
        return self.mySpace.genericOneDimensionalFourierTransformFromZero(np.array(self.timeSeries), 
                                                                          self.zEmission())
    def ElectricFieldOverlapWithOtherTimeFunction(self, timeFunctionTuple, phase = 0.0):
        phaseFactor = np.exp(1.0j * phase) 
        
        timeSeries = self.timeSeries
        xEmit = self.xEmission() * phaseFactor
        yEmit = self.yEmission() * phaseFactor
        zEmit = self.zEmission() * phaseFactor
        
        
        xFunc, yFunc, zFunc = timeFunctionTuple
        
        overlaps = []
        
        for timeIndex, tValue in enumerate(timeSeries):
            newOverlap = xEmit[timeIndex] * xFunc.valueAtTime(tValue)
            newOverlap = newOverlap + yEmit[timeIndex] * yFunc.valueAtTime(tValue)
            newOverlap = newOverlap + zEmit[timeIndex] * zFunc.valueAtTime(tValue)
            newOverlap = 2.0 * np.real(newOverlap)
            
            
            overlaps.append(newOverlap)
        
        overlaps = np.array(overlaps, dtype=np.complex)
        
        return scipy.integrate.simps(y=overlaps, x=timeSeries)
        
        
if __name__ == "__main__":
    #Some useful test functions
    startTime = time.time()
    mySpace = Spacetime.Spacetime(xMax = 10,
                 numberOfNuclearDimenions = 2,
                 numberOfElectronicDimensions = 2,
                 numberOfSimulationSpacePointsPerNuclearDimension = 100,
                 dt = .01)
    
    
    groundStateHO = NuclearOperator.harmonicOscillator(mySpace, 
                 omega=1.0, 
                 mass=1, 
                 center=0, 
                 energyOffset = 0)
    excitedStateHO = NuclearOperator.harmonicOscillator(mySpace, 
                 omega= 2.0, 
                 mass=1, 
                 center=2.0, 
                 energyOffset = 1)
                 
                 
    testNuclearHamiltonian1 = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [groundStateHO, groundStateHO] )
    testNuclearHamiltonian2 = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [excitedStateHO, excitedStateHO ] )
    
    
    
    testNuclearWavefunction1 = NuclearWavefunction.nuclearWavefunction(mySpace, groundStateNuclearHamiltonian = testNuclearHamiltonian1 )
    
    
    testElectronicHamiltonian = ElectronicOperator.ElectronicHamiltonian(mySpace, [(0,0, testNuclearHamiltonian1), 
                                                                (1,1, testNuclearHamiltonian2)])
    testEWF = ElectronicWavefunction.electronicWavefunction(mySpace, 
                           listOfNuclearWavefunctions = [testNuclearWavefunction1, 0],
                           Normalize=True)
                           
                 
    Ey = TimeFunction.GaussianCosinePulse(mySpace, 
                            centerOmega=2.0, 
                            timeSpread=.5, 
                            centerTime=2, 
                            amplitude=1.)
                                         
    Ex = TimeFunction.zeroTimeFunction(mySpace)
    Ez = TimeFunction.zeroTimeFunction(mySpace)
    
    constantMu1 = NuclearOperator.constantPositionNuclearOperator(mySpace, .3)
    constantMu2 = NuclearOperator.constantPositionNuclearOperator(mySpace, .5)
    
    MuX = ElectronicOperator.ElectronicPositionOperator(mySpace, [])
    MuY = ElectronicOperator.ElectronicPositionOperator(mySpace, [(0, 1, constantMu2), (1, 0, constantMu2)])
    MuZ = ElectronicOperator.ElectronicPositionOperator(mySpace, [])
    
    testPerturbation = electricDipole(mySpace, testElectronicHamiltonian, (MuX, MuY, MuZ), (Ex, Ey, Ez))
    
    testPerturbation.propagateInitialWavefunctionToEndOfPerturbation(testEWF)
    testPerturbation.freeEvolution(2000)
    
    #testPerturbation.myPerturbationTimeDependentElectronicWavefunction.animate1D('perturbationTest.mp4', 450)

       
    
    w, sig = testPerturbation.ySpectrum()
    plt.figure()
    plt.plot(w, np.abs(sig))
    plt.title(r"$E_y(\omega)$")    
    plt.xlim((0, 10))
    
    print testPerturbation.ElectricFieldOverlapWithOtherTimeFunction((Ex, Ey, Ez))
    
    print "time elapsed", time.time() - startTime
    
    
    
    