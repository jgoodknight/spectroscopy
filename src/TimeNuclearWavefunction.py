# -*- coding: utf-8 -*-
#Checked for release 10 Feb 2017
import copy
import time

import matplotlib.pyplot as plt
try:
    from matplotlib import animation
except:
    animation = object()
import numpy as np
import scipy
import scipy.integrate

import Spacetime
import NuclearOperator
import NuclearWavefunction


class timeNuclearWavefunction(object):
    "Defines a nuclear wavefunction and the operations which can occur on it.  Not terribly utilized in this package"
    
    def __init__(self, SpaceToExistIn):
        
        self.mySpace = SpaceToExistIn
        
        self.timePositionAmplitude = None
        
        self.timeSeries = []
    
    def applyOperatorNTimesOnInitialWavefunction(self, nuclearOperator, N, initialWF, overrideDT = None):
        self.initialWF = initialWF
        initialAmplitude = initialWF.xAmplitude
        self.timePositionAmplitude = self.mySpace.functionSpacetimeZero(N)
        self.timePositionAmplitude[0] = initialAmplitude
        
        #check that dt is the same
        if overrideDT is not None:
            dt = overrideDT
        else:
            dt = self.mySpace.dt
        self.timeSeries.append(0.0)
        #now to apply the operator
        for i in range(1, N):
            newWF = nuclearOperator.APPLY(self[i-1])
            newAmplitude = newWF.xAmplitude
            self.timePositionAmplitude[i] = newAmplitude
            #update the time series
            self.timeSeries.append(self.timeSeries[-1] + dt)
        self.timeSeries = np.array(self.timeSeries)
        
        self.__autocorrelation = None
        self.__autocorrelationTimeSeries = None
        
        
    def allocateSpace(self, n):
        self.timePositionAmplitude = self.mySpace.functionSpacetimeZero(n)
    
    def setInitialWavefunction(self, initialWF):
        self.initialWF = initialWF
        initialAmplitude = initialWF.xAmplitude
        self.timePositionAmplitude = self.mySpace.functionSpacetimeZero(1)
        self.timePositionAmplitude = initialAmplitude
    
    def applyAndExtendOnce(self, operator, updateTime=True, overrideDT = None):
        if updateTime:
            if overrideDT is not None:
                dt = overrideDT
            else:
                dt = self.mySpace.dt
            self.timeSeries.append(self.timeSeries[-1] + dt)
        
        newLength = self.timePositionAmplitude.shape[0] + 1
        
        newAmplitude = self.mySpace.functionSpacetimeZero(newLength)
        newAmplitude[0:newLength-1] = self.timePositionAmplitude
        
        newWF = operator.APPLY(self[newLength-1])
        newFinalAmplitude = newWF.xAmplitude
        
        newAmplitude[-1] = newFinalAmplitude
        
        self.timePositionAmplitude = newAmplitude
        
    def normSeries(self):
        "For error-checking: how does the norm of the wavefunction change as a function of time"
        norms = []
        for WF in self:
            norms.append(WF.norm())
        return np.array(norms)
        
    def timeOverlap(self, otherTimeNuclearWF):
        "takes another nuclear wavefunction as input, conjugates the other and then outputs their time-series overlap"
        outputTimeSeries = []
        for timeIndex, nuclearWF in enumerate(self):
            temp = otherTimeNuclearWF[timeIndex].returnComplexConjugate() * nuclearWF
            temp = temp.integratedAmplitude()
            outputTimeSeries.append(temp)
        return np.array(outputTimeSeries)
        
    def integratedTimeOverlap(self, otherTimeNuclearWF):
        return scipy.integrate.simps(self.timeOverlap(otherTimeNuclearWF), dx = self.mySpace.dt)
        
    def autocorrelation(self):
        "Autocorrelation as a function of time"
        if self.__autocorrelation is not None:
            return self.__autocorrelationTimeSeries, self.__autocorrelation
        
        negT = -np.flipud(self.timeSeries[1:])
        autocorrelationTime = np.hstack((negT, self.timeSeries))
        self.__autocorrelationTimeSeries = autocorrelationTime
        
        initialWF = self[0]
        ACF = []
        for WF in self:
            ACF.append(WF.overlap(initialWF))
        ACF = np.array(ACF)
        negACF = np.conj(np.flipud(ACF[1:]))
        totalACF = np.hstack((negACF, ACF))
        self.__autocorrelation = totalACF
        return self.__autocorrelationTimeSeries, self.__autocorrelation
        
    def autocorrelationInFrequencySpace(self):
        "Autocorrelation as a function of frequency"
        t, ACF = self.autocorrelation()
        return self.mySpace.genericOneDimensionalFourierTransformFromZero(t, ACF)

                
    ##DEFINE ITERATION METHODS OVER TIME
    def __iter__(self):
        self.counter = 0
        return self

    def next(self):
        try:
            self.counter = self.counter + 1
            return self[self.counter - 1]
        except IndexError:
            raise StopIteration
        
    def __getitem__(self,index):
        "outputs the spatial wavefunction at time index index"
        out = NuclearWavefunction.nuclearWavefunction(self.mySpace)
        out.xAmplitude = self.timePositionAmplitude[index]
        return out
        
    def __setitem__(self,index, nucWF):
        "outputs the spatial wavefunction at time index index"
        self.timePositionAmplitude[index] = nucWF.xAmplitude
    
    def animate1D(self, fileName, numberOfFrames):   
        "Animate a 1D nuclear wavefunction as it evolves in time"             
        d = self.mySpace.nuclearDimensionality
        if d != 1:
            raise NuclearWavefunction.unplotableNuclearWavefunction()
            
        plottingAmplitude = np.abs(self.timePositionAmplitude)
        yMin = np.min(plottingAmplitude)
        yMax = np.max(plottingAmplitude)
        
        fig = plt.figure()
        ax = plt.axes(xlim=(-self.mySpace.xMax, self.mySpace.xMax), ylim = (yMin, yMax))
        line, = ax.plot([], [])
        def init():
            line.set_data([], [])
            return line, 
        
        def animate(i):
            line.set_data(self.mySpace.xValues, plottingAmplitude[i])
            return line, 
        
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames = numberOfFrames, interval=20, blit=True)
        anim.save(fileName, fps=20)
        
        
        
    def animate2D(self, fileName, numberOfFrames):   
        "Animate a 2D nuclear wavefunction as it evolves in time"             
        d = self.mySpace.nuclearDimensionality
        if d != 2:
            raise NuclearWavefunction.unplotableNuclearWavefunction()
            
        plottingAmplitude = np.abs(self.timePositionAmplitude)
        zMin = np.min(plottingAmplitude)
        zMax = np.max(plottingAmplitude)
        
        contourLevels = 100
        
        contourSpacings = np.linspace(zMin, zMax, contourLevels)
        
        xVals = self.mySpace.xValues
        yVals = self.mySpace.xValues
        
        fig = plt.figure()
        im = plt.contourf(xVals, yVals, plottingAmplitude[0], contourSpacings)
        ax = fig.gca()
        
        def animate(i, data,  ax, fig):
            ax.cla()
            im = ax.contourf(xVals, yVals, data[i], contourSpacings)
            plt.title(str(i)) 
            return im,
        
        anim = animation.FuncAnimation(fig, animate,
                               frames = numberOfFrames, interval=20, blit=True, fargs=(plottingAmplitude, ax, fig))
        anim.save(fileName, fps=20)
            
    
    
    #override arithmetic http://www.rafekettler.com/magicmethods.html
    def __mul__(self, other):
        output = copy.copy(self)
        output.timePositionAmplitude = output.timePositionAmplitude * other
        return output
        
    def __neg__(self):
        output = copy.copy(self)
        output.timePositionAmplitude = -output.timePositionAmplitude
        return output
        
    def __add__(self, other):
        output = copy.copy(self)
        output.timePositionAmplitude = output.timePositionAmplitude + other.timePositionAmplitude
        return output


if __name__ == "__main__":
    #Some useful test functions
    ##1 dimensional test of timeWavefunction
    DT = .01
    mySpace = Spacetime.Spacetime(xMax = 10,
                 numberOfNuclearDimenions = 1,
                 numberOfElectronicDimensions = 2,
                 numberOfSimulationSpacePointsPerNuclearDimension = 100,
                 dt = DT)
                 
    omega0 = 2.0
    omegaOff = 1.0
    testHarmonicOscillator = NuclearOperator.harmonicOscillator(mySpace, 
                 omega=omega0, 
                 mass=1, 
                 center=0, 
                 energyOffset = 0)
    testOffsetHarmonicOscillator = NuclearOperator.harmonicOscillator(mySpace, 
                 omega=omegaOff, 
                 mass=1, 
                 center=3, 
                 energyOffset = 0)
    T = 2.0 * np.pi / (omegaOff )
    nSteps = 6 * int( T / DT )
    #wavefunction with offset center
    testNuclearOffsetHamiltonian = NuclearOperator.nuclearHamiltonian(mySpace, 
                 listOfOneDimensionalHamiltonians = [testOffsetHarmonicOscillator] )
    testNuclearWF = NuclearWavefunction.nuclearWavefunction(mySpace, 
                                        groundStateNuclearHamiltonian = testNuclearOffsetHamiltonian)
    testWFE0 = testNuclearOffsetHamiltonian.energyEigenvalue([0])
    testWFomega0 = testWFE0 / mySpace.hbar
    
    #propagator for HO in center of space                                
    testNuclearHamiltonian = NuclearOperator.nuclearHamiltonian(mySpace, 
                 listOfOneDimensionalHamiltonians = [testHarmonicOscillator] )
    testNuclearPropagator = testNuclearHamiltonian.myNuclearPropagator(HIGH_ACCURACY=False, LOW_ACCURACY=False)
    

    #put the offset wavefunction in the propagator of the centered potential
    testTimeFunction = timeNuclearWavefunction(mySpace)
    startTime = time.time()
    testTimeFunction.applyOperatorNTimesOnInitialWavefunction(testNuclearPropagator, nSteps, testNuclearWF)
    
    
    omega, spec = testTimeFunction.autocorrelationInFrequencySpace()
    omega = omega - testWFomega0 #shift so it's an absorption spectrum from the initial energy state
    normSpec = np.abs(spec) 
    normSpec = normSpec / np.max(normSpec)
    
    print "Elapsed time", time.time() - startTime
    
    #find the peaks
    peakFrequencies, peakHeights = Spacetime.Spacetime.peakFinder(omega, normSpec, threshold=.2)
    
    nPeaks = 10 #len(peakHeights)
    
    #find what the spectrum shoudl look like
    omegaFC, specFC = testNuclearOffsetHamiltonian.frankCondonOverlapSpectrum( testNuclearHamiltonian, nPeaks)
    normspecFC = specFC / np.max(specFC)

    
    plt.figure()
    plt.plot(omega, np.abs(normSpec), label="propagation spectrum")
    plt.plot(omegaFC, normspecFC, 'rx', label="FC spectrum")
    plt.plot(peakFrequencies, peakHeights, 'g+', label="peaks found")
    plt.legend()
    plt.title("spectra")
    plt.xlim((-1, 50))
    
    print "Actual Peaks at:", peakFrequencies
    print "\nFC peaks at:", omegaFC


    
    

    ##2 dimensional test of timeWavefunction
    DT = .01
    mySpace = Spacetime.Spacetime(xMax = 10,
                 numberOfNuclearDimenions = 2,
                 numberOfElectronicDimensions = 2,
                 numberOfSimulationSpacePointsPerNuclearDimension = 200,
                 dt = DT)
                 
    omega0 = 2.0
    omegaOff = 1.0
    testHarmonicOscillator1 = NuclearOperator.harmonicOscillator(mySpace, 
                 omega=omega0, 
                 mass=1, 
                 center=0, 
                 energyOffset = 0)
    testHarmonicOscillator2 = NuclearOperator.harmonicOscillator(mySpace, 
                 omega= 2 * omega0, 
                 mass=1, 
                 center=0, 
                 energyOffset = 0)
                 
            
    testOffsetHarmonicOscillator1 = NuclearOperator.harmonicOscillator(mySpace, 
                 omega=omegaOff, 
                 mass=1, 
                 center=3, 
                 energyOffset = 0)
    testOffsetHarmonicOscillator2 = NuclearOperator.harmonicOscillator(mySpace, 
                 omega= .5*omegaOff, 
                 mass=1, 
                 center=2, 
                 energyOffset = 0)
                 
    T = 2.0 * np.pi / (omegaOff )
    nSteps = 6 * int( T / DT )
    #wavefunction with offset center
    testNuclearOffsetHamiltonian = NuclearOperator.nuclearHamiltonian(mySpace, 
                 listOfOneDimensionalHamiltonians = [testOffsetHarmonicOscillator1, testOffsetHarmonicOscillator2] )
    testNuclearWF = NuclearWavefunction.nuclearWavefunction(mySpace, 
                                        groundStateNuclearHamiltonian = testNuclearOffsetHamiltonian)
                                        
    testWFE0 = testNuclearOffsetHamiltonian.energyEigenvalue([0, 0])
    testWFomega0 = testWFE0 / mySpace.hbar
    
    #propagator for HO in center of space                                
    testNuclearHamiltonian = NuclearOperator.nuclearHamiltonian(mySpace, 
                 listOfOneDimensionalHamiltonians = [testHarmonicOscillator1, testHarmonicOscillator2] )
    testNuclearPropagator = testNuclearHamiltonian.myNuclearPropagator(HIGH_ACCURACY=False, LOW_ACCURACY=False)
    

    #put the offset wavefunction in the propagator of the centered potential
    testTimeFunction = timeNuclearWavefunction(mySpace)
    startTime = time.time()
    testTimeFunction.applyOperatorNTimesOnInitialWavefunction(testNuclearPropagator, nSteps, testNuclearWF)
    
    
    omega, spec = testTimeFunction.autocorrelationInFrequencySpace()
    omega = omega - testWFomega0 #shift so it's an absorption spectrum from the initial energy state
    normSpec = np.abs(spec) 
    normSpec = normSpec / np.max(normSpec)
    
    print "Elapsed time", time.time() - startTime
    
    #find the peaks
    peakFrequencies, peakHeights = Spacetime.Spacetime.peakFinder(omega, normSpec, threshold=.2)
    
    plt.figure()
    plt.plot(omega, np.abs(normSpec), label="propagation spectrum")
    #plt.plot(omegaFC, normspecFC, 'rx', label="FC spectrum")
    plt.plot(peakFrequencies, peakHeights, 'g+', label="peaks found")
    #plt.legend()
    plt.title("spectra")
    plt.xlim((-1, 50))
    
    print "Actual Peaks at:", peakFrequencies
    #print "\nFC peaks at:", omegaFC

    testTimeFunction.animate2D("2dPropagationTest.mp4", 200)
    
  
    
    