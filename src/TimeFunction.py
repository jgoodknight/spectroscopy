# -*- coding: utf-8 -*-
"""
Created on Fri Jan 04 13:23:27 2013
Gone through for release 10 Feb 2017
@author: Joey
"""
import copy
import math

import numpy as np
import matplotlib.pyplot as plt

import scipy.integrate
import scipy.interpolate

import Spacetime


class timeFunction(object):
    """ 
    Abstract class to keep track of useful items for a function in time.
    Must have a defined beginning and end, before and after which the function
    is identically zero
    """
    
    #Override arithmetic functions
    def __mul__(self, other):
        if isinstance(other, zeroTimeFunction):
            return other
        out = copy.copy(self)
        out.myFunction = lambda t: self.myFunction(t) * other.myFunction(t)
        return out
        
    def __add__(self, other):
        if isinstance(other, zeroTimeFunction):
            return self
        out = copy.copy(self)
        out.myFunction = lambda t: self.myFunction(t) + other.myFunction(t)
        newMin = min(out.timeRangeTuple[0], other.timeRangeTuple[0])
        newMax = max(out.timeRangeTuple[1], other.timeRangeTuple[1])
        out.timeRangeTuple = (newMin, newMax)
        return out
        
    def plot(self, nPoints = 2000):
        "Plots from user-defined beginning to end of function for specified number of points"
        plt.figure()
        tVals = np.linspace(self.timeRangeTuple[0], self.timeRangeTuple[1], nPoints)
        t = self.mySpace.unitHandler.femtosecondsFromTime(tVals)
        fValues = self.myFunction(tVals)
        plt.plot(t, fValues)
        plt.xlabel(r"$t$ (fs)")
        
        
    def plotInSimulationSpace(self): 
        "Plots using beginning/end and dt from mySpace"
        plt.figure()
        tVals = np.arange(self.timeRangeTuple[0], self.timeRangeTuple[1] + self.mySpace.dt, self.mySpace.dt)
        t = self.mySpace.unitHandler.femtosecondsFromTime(tVals)
        fValues = self.myFunction(tVals)
        plt.plot(t, fValues)
        plt.xlabel(r"$t$ (fs)")
        
    def plotFT(self,  maxFrequency, resolution):
        "Plots the Fourier Transform of the function"
        plt.figure()
        w, s = self.fourierTransform(maxFrequency, resolution)
        w = self.mySpace.unitHandler.wavenumbersFromEnergyUnits(w)
        plt.plot(w, np.abs(s))
        plt.xlabel(r"$\omega$ (cm$^{-1}$)")
        
    def FT(self,  maxFrequency, resolution):
        "returns the Fourier Transform frewuency in wavenumbers and the amplitude"
        w, s = self.fourierTransform(maxFrequency, resolution)
        w = self.mySpace.unitHandler.wavenumbersFromEnergyUnits(w)
        return w, s
        
    def fourierTransform(self, maxFrequency, resolution):
        tMax = 1.0 /  resolution
        N = maxFrequency / resolution
        tValues = np.linspace(-tMax, tMax, N)
        fValues = self.myFunction(tValues)
        return self.mySpace.genericOneDimensionalFourierTransformFromZero(tValues, fValues, gaussianFilter=False)
        
    def valueAtTime(self, t):
        return self.myFunction(t)
    
    def maxTime(self):
        return self.timeRangeTuple[1]
    def minTime(self):
        return self.timeRangeTuple[0]
        
    def shiftForward(self, deltaT):
        self.T = self.T + deltaT
        self.timeRangeTuple = (self.timeRangeTuple[0] + deltaT, self.timeRangeTuple[1] + deltaT) 
        return self
        
    def integrate(self, dt):
        "Integrates Amplitude to desired dt"
        DT = dt 
        tValues = np.arange(self.minTime(), self.maxTime() + DT, DT)
        Values = self.valueAtTime(tValues)
        return scipy.integrate.simps(Values, dx = DT)
        
        
    def integrateInSimulationSpace(self):
        "Integrates Amplitude to mySpace.dt"
        return self.integrate(self.mySpace.dt)
        
    def setPhase(self, newPhase):
        self.phi = newPhase
    
class zeroTimeFunction(timeFunction):
    "A Time Function smart enough to know it's zero"
    def __init__(self, space):
        self.mySpace = space
        self.myFunction = 0.0
        
        self.timeRangeTuple = (np.inf, -np.inf)
        self.timePillow = 0.0
        self.T = 0.0
        self.sigma = 0
        
    def valueAtTime(self, t):
        try:
            return np.zeros(t.shape)
        except:
            return 0.0
        
    def shiftForward(self, deltaT):
        return self
    
    def integrate(self, a):
        return 0.0
        
    def totalPower(self):
        return 0.0
        
    def plusFunction(self):
        return zeroTimeFunction(self.mySpace)
                 
    def minusFunction(self):
        return zeroTimeFunction(self.mySpace)
        
    def pulseCopyAtNewTime(self, newTime):
        return self
        
    def pulseCopyJumpedForward(self, amountOfTimeToJumpForward):
        return self
        
    def fourierTransform(self):
        return zeroTimeFunction(self.mySpace)
        
    def myFourierTransformedFunction(self, w):
        return np.zeros(w.shape)
        
class deltaFunction(timeFunction):
    "A Function with ampltiude a/dt at specified time=loaction"
    def __init__(self, space, location, dt=None):
        self.mySpace = space
        if dt is None:
            self.DT = self.mySpace.dt
        else:
            self.DT = dt
        self.height = 1.0 / self.mySpace.dt
    
        self.T = location        
        
        self.timePillow = 5.0 * self.mySpace.dt #just to be safe
        
        self.timeRangeTuple = (location - self.DT, location + self.DT)
        
    def valueAtTime(self, t):
        try:
            t.shape
        except:
            if np.abs(t - self.T) < self.DT:
                return self.height
            else:
                return 0.0
        ones = np.ones(t.shape)
        zeros = np.zeros(t.shape)
        output = np.where(np.abs(t - self.T) > self.DT, zeros, ones)
        return output * self.height
            
    def pulseCopyAtNewTime(self, newTime):
        output = copy.deepcopy(self)
        output.T = newTime
        return output
        
    def pulseCopyJumpedForward(self, amountOfTimeToJumpForward):
        output = copy.deepcopy(self)
        output.T = amountOfTimeToJumpForward + self.T
        return output
    
    def totalPower(self):
        return 1.0
        
class GaussianPulse(timeFunction):
    "An abstract object to hold useful methods for dealing with Gaussian pulses"
    def __init__(self,
                 space,
                 centerOmega, 
                 timeSpread, 
                 centerTime=None, 
                 pulseDirection = None, 
                 phi = 0.0,
                 amplitude = 1.00E-8,
                 normalizePower = False,
                 normalizeIntegral = True,
                 frequency_sign = 1.0):
        self.mySpace = space
        
        if pulseDirection is None: #not important yet, may be implemented later
            pass
        
        self.omega = centerOmega
        self.sigma = timeSpread
        self.phi   = 0
        
        self.k = pulseDirection
        
        
        targetTimePillow = 6.0 * self.sigma #amount of padding needed between center of pulse and esdge of time
        nDtInTargetPillow = np.ceil(targetTimePillow / self.mySpace.dt) + 1
        
        self.timePillow = nDtInTargetPillow * self.mySpace.dt
        #if no time is specified, then just give the pulse the needed pillow
        if centerTime == None:
            centerTime = self.timePillow
        self.frequency_sign = frequency_sign
        #search for the closest index to the specified center time
        self.T     = centerTime
        
        self.timeRangeTuple =  (self.T - self.timePillow, self.T + self.timePillow)
        
        #the goal is to normalize the power or integral of the square of the function to one        
        self.amp   = amplitude #start here; this is needed to make the function work
    
        if normalizePower:
            normalizer = self.totalPower() 
    
            self.amp = self.amp  / math.sqrt(normalizer)      
        if normalizeIntegral:
            normalizer = self.totalIntegral() 
    
            self.amp = self.amp  / normalizer
            
    def shiftToCenterTime(self, newT):
        "Gives pulse a new center time"
        delta_T = newT - self.T
        self.T = newT
        self.timeRangeTuple = (self.timeRangeTuple[0] + delta_T, self.timeRangeTuple[1] + delta_T)
        
    def totalPower(self):
        "Integral of the absolute value squared of the amplitude"
        DT = self.mySpace.dt #/ 10.0
        tValues = np.arange(self.minTime(), self.maxTime(), DT)
        Values = np.abs(self.valueAtTime(tValues))**2.0
        return scipy.integrate.simps(Values, dx = DT)
        
    def totalIntegral(self):
        "Integral of the absolute value of the amplitude"
        DT = self.mySpace.dt #/ 10.0
        tValues = np.arange(self.minTime(), self.maxTime(), DT)
        Values = np.abs(self.valueAtTime(tValues))
        return scipy.integrate.simps(Values, dx = DT)
        
    def pulseCopyAtNewTime(self, newTime):
        "Make copy and move to new time"
        output = copy.deepcopy(self)
        output.T = newTime
        return output
        
    def pulseCopyJumpedForward(self, amountOfTimeToJumpForward):
        "Make copy and jump the pulse forward"
        output = copy.deepcopy(self)
        output.T = amountOfTimeToJumpForward + self.T
        return output

        
class GaussianPulseTooCloseToEdgeOfTimeException(Exception):
    def __init__(self, value):
        self.value = value

class GaussianCosinePulse(GaussianPulse):
    "Completely Real Pulse"
    def myFunction(self, t):
        coef = self.amp
        shiftedTime = t - self.T
        
        cosArg  = -self.omega * shiftedTime + self.phi
        cosTerm = np.cos(cosArg)
        gausArg  = -shiftedTime**2.0 / (2.0 * self.sigma**2.0)
        gausTerm = np.exp(gausArg)
        
        return coef * cosTerm * gausTerm
        
    def myFourierTransformedFunction(self, w):
        return self.minusFunction().myFourierTransformedFunction(w) + self.plusFunction().myFourierTransformedFunction(w)
        
    def plusFunction(self):
        return GaussianKPlusPulse(
                 self.mySpace,
                 self.omega, 
                 self.sigma, 
                 self.T, 
                 self.k, 
                 self.phi,
                 self.amp / 2.0,
                 normalizePower = False,
                 normalizeIntegral = False,
                 frequency_sign = -1.0)
                 
    def minusFunction(self):
        return GaussianKMinusPulse(
                 self.mySpace,
                 self.omega, 
                 self.sigma, 
                 self.T, 
                 self.k, 
                 self.phi,
                 self.amp / 2.0,
                 normalizePower = False,
                 normalizeIntegral = False,
                 frequency_sign = 1.0)
        
        
class GaussianKPlusPulse(GaussianPulse): 
    "'Forward' Pulse which is complex and has positive energy"
    def myFunction(self, t):
        coef = self.amp 
        shiftedTime = t - self.T
        
        expArg  = -self.omega * shiftedTime + self.phi
        expTerm = np.exp(1.0j * expArg)
        
        gausArg  = -shiftedTime**2.0 / (2.0 * self.sigma**2)
        gausTerm = np.exp(gausArg)   
        
        return coef * expTerm * gausTerm
        
    def myFourierTransformedFunction(self, w):
        """Defined as $\int e^{i \omega t} E(t) dt $ """
        coef = self.amp * np.sqrt(2.0 * np.pi * self.sigma**2.0)
        
        oscPart = np.exp(1.0j *w * self.T)
        
        expPart = np.exp(-self.sigma**2 * (self.omega - w)**2 / 2.0)
        
        return coef * oscPart * expPart

class GaussianKMinusPulse(GaussianPulse):      
    "'Backward' Pulse which is complex and negative energy"
    
    def myFunction(self, t):
        coef = self.amp 
        shiftedTime = t - self.T
        
        expArg  = self.omega * shiftedTime - self.phi
        expTerm = np.exp(1.0j * expArg)
        
        gausArg  = -shiftedTime**2.0 / (2.0 * self.sigma**2.0)
        gausTerm = np.exp(gausArg)   
        
        return coef * expTerm * gausTerm
        
        
    def myFourierTransformedFunction(self, w):
        """Defined as $\int e^{i \omega t} E(t) dt $ """
        coef = self.amp * np.sqrt(2.0 * np.pi * self.sigma**2.0)
        
        oscPart = np.exp(1.0j *w * self.T)
        
        expPart = np.exp(-self.sigma**2 * (self.omega + w)**2 / 2.0)
        
        return coef * oscPart * expPart
    
class GaussianRazorBladedPulse(timeFunction):
    "Object to handle a pulse which has been razor-bladed in frequency space"
    k_MULTIPLIER = 80000000.0
    time_window_high_frequency_multiplier = 10.0
    time_domain_multiplier = 40
    ZERO_TOLERANCE = 1.0E-3
    
    def __init__(self, gaussian_pulse, cutoff_omega_low, cutoff_omega_high):
        self.my_underlying_pulse = copy.deepcopy(gaussian_pulse)
        self.mySpace = self.my_underlying_pulse.mySpace
        self.low_frequency_cutoff = cutoff_omega_low
        self.high_frequency_cutoff = cutoff_omega_high
        
        self.T = gaussian_pulse.T
        self.sigma = gaussian_pulse.sigma
        self.frequency_sign = gaussian_pulse.frequency_sign
        self.omega = gaussian_pulse.omega
        
        
        self.k = GaussianRazorBladedPulse.k_MULTIPLIER / gaussian_pulse.mySpace.dt
        
        self.normalizer = self.my_underlying_pulse.totalIntegral()
        self.frequency_over_time_max_value_ratio = 1.0
        
        dt = gaussian_pulse.mySpace.dt
        self.dt = dt
        window_size_t = GaussianRazorBladedPulse.time_window_high_frequency_multiplier / self.high_frequency_cutoff
        
        underlying_start_time, underlying_end_time = gaussian_pulse.timeRangeTuple
        t_max = self.T + GaussianRazorBladedPulse.time_domain_multiplier * max(np.abs(underlying_start_time), np.abs(underlying_end_time)) + dt
        t_min = self.T - GaussianRazorBladedPulse.time_domain_multiplier * max(np.abs(underlying_start_time), np.abs(underlying_end_time))
        
        self.master_t = np.arange(t_min, t_max, dt)
        self.master_w = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(self.master_t.shape[0], d = self.master_t[1] - self.master_t[0]))
        

        self.f_shift_center = 0.0
        self.master_frequency_values = self.myFourierTransformedFunction_initial(self.master_w)
        self.master_frequency_interpolant = scipy.interpolate.interp1d(self.master_w, self.master_frequency_values, fill_value=0.0, bounds_error = False)
        
        self.master_time_values = np.fft.fftshift(np.fft.ifft(self.master_frequency_values)) / (t_max - t_min)
        self.master_time_interpolant = scipy.interpolate.interp1d(self.master_t, self.master_time_values)
        
        
        #find a more reasonable start time to allow easier calculation:
        finding_start = True
        t_f = underlying_start_time

        maximum_time_amplitude = np.max(np.abs(self.master_time_values))        
        
        while finding_start:
            t_0 = t_f - window_size_t
            t_vals = np.linspace(t_0, t_f, 100)
            func_values = self.master_time_interpolant(t_vals)
            rms_Err = np.std(np.abs(func_values) / maximum_time_amplitude)
            
            if rms_Err <GaussianRazorBladedPulse.ZERO_TOLERANCE:
                start_time = t_0
                finding_start = False
            t_f = t_0
        print "found start time at %f femtoseconds" % self.mySpace.unitHandler.femtosecondsFromTime(start_time)
                
        finding_end = True
        t_0 = underlying_end_time
        while finding_end:
            t_f = t_0 + window_size_t
            t_vals = np.linspace(t_0, t_f, 100)
            func_values = self.master_time_interpolant(t_vals)
            rms_Err = np.std(np.abs(func_values) / maximum_time_amplitude)
            if rms_Err < GaussianRazorBladedPulse.ZERO_TOLERANCE:
                end_time = t_f
                finding_end = False
            t_0 = t_f
            
        print "found end time at %f femtoseconds" % self.mySpace.unitHandler.femtosecondsFromTime(end_time)
        
        
        
        self.timeRangeTuple = (start_time , end_time )
        self.natural_time = np.arange(start_time, end_time, dt)
        
        self.n_time_point = self.natural_time.shape[0]
        
        self.natural_frequencies = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(self.n_time_point, d = self.dt))
        
        self.time_interpolant = scipy.interpolate.interp1d(self.natural_time, self.master_time_interpolant(self.natural_time), fill_value=0.0, bounds_error = False)
        
        #we want the pulse to turn on at t=0
        shift = - start_time 
        self.shiftForward(shift)
        
        #where is the center of the pulse?
        max_index = np.argmax(self.myFunction(self.natural_time))
        max_amplitude_time = self.natural_time[max_index]
        
        start_pillow = max_amplitude_time - self.timeRangeTuple[0]
        end_pillow = self.timeRangeTuple[1] - max_amplitude_time 
        
        self.timePillow = np.max([start_pillow, end_pillow])
        
        
    def myFunction(self, t):
        return self.normalizer * self.time_interpolant(t)
        
    def myFourierTransformedFunction_initial(self, w):
        return self.normalizer * self.my_underlying_pulse.myFourierTransformedFunction(w) * self.razorBladeWindow(w)
        
    def myFourierTransformedFunction(self, w):
        return np.exp(-1.0j * w * self.f_shift_center) * self.master_frequency_interpolant(w)
        
    def razorBladeWindow(self, w):
        return 1.0 / ( (1.0 + np.exp(-2.0 * self.k * (w - self.low_frequency_cutoff))) *  (1.0 + np.exp(-2.0 * self.k * (self.high_frequency_cutoff - w)))  )
#        return 1.0 / ((1.0 + np.exp(-2.0 * self.k * (self.high_frequency_cutoff - w)))  )
        
    def totalIntegral(self):
        tValues = self.natural_time
        DT = tValues[1] - tValues[0]
        Values = np.abs(self.myFunction(self.natural_time))
        return scipy.integrate.simps(Values, dx = DT)
        
    def totalPower(self):
        tValues = self.natural_time
        DT = tValues[1] - tValues[0]
        Values = np.abs(self.myFunction(self.natural_time))**2
        return scipy.integrate.simps(Values, dx = DT)
        
    def shiftForward(self, jump_amount):
        amplitude = self.myFunction(self.natural_time)
        
        self.natural_time = self.natural_time + jump_amount
        self.f_shift_center = self.f_shift_center + jump_amount
        
        self.timeRangeTuple = (self.timeRangeTuple[0] + jump_amount, self.timeRangeTuple[1] + jump_amount)
        
        self.time_interpolant = scipy.interpolate.interp1d(self.natural_time, amplitude, fill_value=0.0, bounds_error = False)
        
        current_integral_amount = self.totalIntegral()
        
        self.normalizer  = self.normalizer / current_integral_amount
        
        return self
        
        
    def plusFunction(self):
        return GaussianRazorBladedPulse(self.my_underlying_pulse, self.low_frequency_cutoff, self.high_frequency_cutoff)
    def minusFunction(self):
        return GaussianRazorBladedPulse(self.my_underlying_pulse, self.low_frequency_cutoff, self.high_frequency_cutoff)
        
        



if __name__ == "__main__":
    #Some useful test code
    n=100
    mySpace = Spacetime.Spacetime(xMax = 10,
                 numberOfNuclearDimenions = 2,
                 numberOfElectronicDimensions = 4,
                 numberOfSimulationSpacePointsPerNuclearDimension = 200,
                 dt = .05)
    a = GaussianCosinePulse(mySpace, 
                            centerOmega=7.5, 
                            timeSpread=.25, 
                            centerTime=1, 
                            amplitude=.6)
    b = GaussianCosinePulse(mySpace, 
                            centerOmega=12.5, 
                            timeSpread=.5, 
                            centerTime=8, 
                            amplitude=.6)
    c = GaussianCosinePulse(mySpace, 
                            centerOmega=12.5, 
                            timeSpread=.9, 
                            centerTime=9, 
                            amplitude=.6)
                            
    
    d = c * b
    d.plot()
    plt.show()
    print d.integrate(1.0)
    
    
        
    
    
    
    
    
    
    
    
    
    
