# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:40:19 2013

@author: joey
"""
import multiprocessing
import itertools
import sys
sys.path.append('..')

import numpy as np
import pickle
import matplotlib
import scipy.integrate
import scipy.interpolate
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import signal


import pandas as pd

dipole_data = pd.read_csv("dipoles.csv")
potential_surfaces = pd.read_csv("potential_surfaces.csv")
wavefunction_data = pd.read_csv("GS_wavefunctions.csv")

number_of_wavefunctions = 10
#dipole_data = np.genfromtxt("dipoles.csv", delimiter=',')
#potential_surfaces = pd.read_csv("potential_surfaces.csv")

system_dictionary = {}


potential_R = potential_surfaces['R'].values
dipole_R = dipole_data['R'].values
master_R = potential_R

potential_dictionary = {}
potential_dictionary['R'] = potential_R
potential_dictionary[(0,0)] = potential_surfaces['GS'].values
potential_dictionary[(1,1)] = potential_surfaces['1 3_'].values
potential_dictionary[(2,2)] = potential_surfaces['2 3_'].values
potential_dictionary[(3,3)] = potential_surfaces['3 3_'].values
potential_dictionary[(4,4)] = potential_surfaces['4 3_'].values

potential_dictionary[(1,2)] = potential_surfaces['1_2'].values
potential_dictionary[(1,3)] = potential_surfaces['1_3'].values
potential_dictionary[(2,3)] = potential_surfaces['2_3'].values
potential_dictionary[(3,4)] = potential_surfaces['3_4'].values
#transpose for good measure/hermiticity
#potential_dictionary[(2,1)] = potential_surfaces['1_2'].values
#potential_dictionary[(3,1)] = potential_surfaces['1_3'].values
#potential_dictionary[(3,2)] = potential_surfaces['2_3'].values
#potential_dictionary[(4,3)] = potential_surfaces['3_4'].values

dipole_dictionary = {}
dipole_dictionary['R'] = master_R
#data is a bit more structured here so we can loop easily

for dipole_key in dipole_data.keys():
    if dipole_key == 'R':
        continue
    data = dipole_data[dipole_key].values
    #pad the array so it is the same length as the potential surfaces
    last_value = data[-1]
    first_value = data[0]
    
    interpolated_data = scipy.interp(master_R, dipole_R, data) #constant value fill
#    interpolant = scipy.interpolate.interp1d(dipole_R, data, bounds_error=False, fill_value="extrapolate") #linear interpolant fill
#    interpolated_data = interpolant(master_R)
    
    indeces = dipole_key.split(",")
    i_1 = int(indeces[0])
    i_2 = int(indeces[1])
    if i_1 == i_2:
        i_1 = 0
    
    dipole_dictionary[(i_1, i_2)] = interpolated_data
#    dipole_dictionary[(i_2, i_1)] = interpolated_data


#wavefunctions
wf_values = {}
wf_R_angstroms = wavefunction_data['R (ang)'].values[0:-1]
wf_values['R_angstroms'] = master_R
energies = []
for i in range(number_of_wavefunctions):
    data = wavefunction_data["V=%i" %i].values[0:-1]
    interpolated_data = scipy.interp(master_R, wf_R_angstroms, data)
    wf_values[i] = interpolated_data
    E = wavefunction_data['E (wavenumbers)'].values[i]
    E = float(E.replace(',',''))
    energies.append(E)

wf_values['E_wavenumbers'] = np.array(energies)


#PLOT
plt.figure()
plt.title(r"$V_{i,j}(R)$")
for pot_key in potential_dictionary.keys():
    data = potential_dictionary[pot_key]
    
    if pot_key != 'R' and pot_key[1] >= pot_key[0]:
        plt.plot(master_R, data, label = str(pot_key))
plt.legend(loc=0)  
plt.xlim(0, 4)


plt.figure()
plt.title(r"$\mu_{i,j}(R)$")
for dipole_key in dipole_dictionary.keys():
    data = dipole_dictionary[dipole_key]
    
    if dipole_key != 'R' and dipole_key[1] >= dipole_key[0]:
        plt.plot(master_R, data, label = str(dipole_key))
plt.legend(loc=0) 
plt.xlim(1.2, 15)
plt.ylim(-.6, 5) 


plt.figure()
plt.title("GS Wavefunctions")

V_g = potential_dictionary[(0,0)]
E_offset = np.min(V_g)

plt.plot(potential_R, V_g, label=r"$V_g(R)$")
plt.hlines(energies + E_offset, np.min(master_R), np.max(master_R), colors="black")

psi_norm = np.nanmax(np.abs(wf_values[0]))
for i in range(number_of_wavefunctions):
    psi = wf_values[i] 
    psi = psi / psi_norm
    psi = psi * energies[0]
    print np.max(psi)
    E = wf_values['E_wavenumbers'][i]
    plt.plot(master_R, psi + E + E_offset, label="n = " + str(i))
    

plt.xlim(1.7, 2.8)
plt.ylim(0 + E_offset, -6500)
plt.legend(loc=4)
plt.ylabel("Energy / wavenumbers")
plt.xlabel("R / Angstroms")

dipole_file = open("dipoles.pkl", 'wb')
pickle.dump(dipole_dictionary, dipole_file)
dipole_file.close()

potential_file = open("potentials.pkl", 'wb')
pickle.dump(potential_dictionary, potential_file)
potential_file.close()

wf_file = open("wavefunctions.pkl", 'wb')
pickle.dump(wf_values, wf_file)
wf_file.close()



