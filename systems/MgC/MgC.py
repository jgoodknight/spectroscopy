# -*- coding: utf-8 -*-
"""
Created on Fri Jan 04 13:23:27 2013

@author: Joey
"""
import pickle
import os



import numpy as np
import scipy.interpolate


import spectroscopy.Spacetime as Spacetime
import spectroscopy.ElectronicOperator as ElectronicOperator
import spectroscopy.NuclearWavefunction as NuclearWavefunction
import spectroscopy.NuclearOperator as NuclearOperator
import spectroscopy.ElectronicWavefunction as ElectronicWavefunction

myDir = os.path.dirname(os.path.realpath(__file__)) + "/"
print myDir

rotating_wave_offset_wavenumbers = -12189.0 - 217.643 - 92.031
#rotating_wave_offset_wavenumbers = 0.0

pulse_central_frequency_wavenumbers = 6370.0

mass_Mg_amu = 24.305
mass_C_amu  = 12.011

reduced_mass_MgC_amu = mass_Mg_amu * mass_C_amu/ (mass_Mg_amu + mass_C_amu)

mass_electron_amu = 0.000548579909

reduced_mass_MgC_electronMasses = reduced_mass_MgC_amu / mass_electron_amu

temp = open(myDir + "dipoles.pkl", "rb")
dipole_data = pickle.load(temp)
temp.close()
temp = open(myDir + "potentials.pkl", "rb")
potential_data = pickle.load(temp)
temp.close()
temp = open(myDir + "wavefunctions.pkl", 'rb')
wavefunction_data = pickle.load(temp)
temp.close()


R_angstroms_original = potential_data['R']
R_angstroms = potential_data['R']
R_min_angstroms = R_angstroms[0]
R_max_angstroms = R_angstroms[-1]
#keep this put for now; 2400 is likely far too much to simulate but we'll see.
n_R_points = R_angstroms.shape[0]

R_range_angstroms = R_max_angstroms - R_min_angstroms
R_minMax_angstroms = R_range_angstroms / 2.0

R_angstroms = R_angstroms - R_min_angstroms - R_minMax_angstroms

R_angstroms_stretched_limit = R_minMax_angstroms * 1.1
n_R_stretched_points = n_R_points 
R_angstroms_stretched = np.linspace(-R_angstroms_stretched_limit, R_angstroms_stretched_limit, n_R_stretched_points)


##all units from here on out are defined by mySpace
mySpace = Spacetime.Spacetime(
             xMax_tuple = None, #define later
             xMax_tuple_angstroms = (R_angstroms_stretched_limit,),
             numberOfNuclearDimenions = 1,
             numberOfElectronicDimensions = 5,
             numberOfSimulationSpacePointsPerNuclearDimension_tuple = (n_R_stretched_points,),
             dt_SECONDS = .00100E-15,
             UnityMassInElectronMasses = reduced_mass_MgC_electronMasses)

pulse_central_frequency = mySpace.unitHandler.energyUnitsFromWavenumbers(pulse_central_frequency_wavenumbers)

for pot_key in potential_data.keys():
    dat = potential_data[pot_key]
    dat = mySpace.unitHandler.energyUnitsFromWavenumbers(dat)
    interper = scipy.interpolate.interp1d(R_angstroms, dat, bounds_error=False, fill_value="extrapolate")
    potential_data[pot_key] = interper(R_angstroms_stretched)
    
    
for dip_key in dipole_data.keys():
    dat = dipole_data[dip_key]
    interper = scipy.interpolate.interp1d(R_angstroms, dat, bounds_error=False, fill_value="extrapolate")
    dipole_data[dip_key] = interper(R_angstroms_stretched)
    
    
for wf_key in wavefunction_data.keys():
    if isinstance(wf_key, str):
        continue
    dat = wavefunction_data[wf_key]
    interper = scipy.interpolate.interp1d(R_angstroms, dat, bounds_error=False, fill_value="extrapolate")
    wavefunction_data[wf_key] = interper(R_angstroms_stretched)
    
rotating_wave_offset = mySpace.unitHandler.energyUnitsFromWavenumbers(rotating_wave_offset_wavenumbers)

wavefunction_data['E'] = mySpace.unitHandler.energyUnitsFromWavenumbers(wavefunction_data['E_wavenumbers'])

R = mySpace.unitHandler.lengthUnitsFromMeters(R_angstroms_stretched * 1e-10)
oneD_xValues = R # for now!


nuclear_ground = NuclearOperator.userSpecifiedPotential_Hamiltonian(mySpace, mass = 1.0, potential_surface = potential_data[(0,0)] + 0 , xValues = oneD_xValues, wavefunction_dictionary = wavefunction_data)
nuclear_excited_1 = NuclearOperator.userSpecifiedPotential_Hamiltonian(mySpace, mass = 1.0, potential_surface = potential_data[(1,1)] + rotating_wave_offset, xValues = oneD_xValues)
nuclear_excited_2 = NuclearOperator.userSpecifiedPotential_Hamiltonian(mySpace, mass = 1.0, potential_surface = potential_data[(2,2)] + rotating_wave_offset, xValues = oneD_xValues)
nuclear_excited_3 = NuclearOperator.userSpecifiedPotential_Hamiltonian(mySpace, mass = 1.0, potential_surface = potential_data[(3,3)] + rotating_wave_offset, xValues = oneD_xValues)
nuclear_excited_4 = NuclearOperator.userSpecifiedPotential_Hamiltonian(mySpace, mass = 1.0, potential_surface = potential_data[(4,4)] + rotating_wave_offset, xValues = oneD_xValues)



electronic_ground = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [nuclear_ground] )

electronic_excited_1 = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [nuclear_excited_1 ] )
electronic_excited_2 = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [nuclear_excited_2 ] )
electronic_excited_3 = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [nuclear_excited_3 ] )
electronic_excited_4 = NuclearOperator.nuclearHamiltonian(mySpace, listOfOneDimensionalHamiltonians = [nuclear_excited_4 ] )

nuclear_coupling_12 = NuclearOperator.userSpecifiedSurface_positionOperator(mySpace, potential_data[(1,2)])
nuclear_coupling_23 = NuclearOperator.userSpecifiedSurface_positionOperator(mySpace, potential_data[(2,3)])
nuclear_coupling_13 = NuclearOperator.userSpecifiedSurface_positionOperator(mySpace, potential_data[(1,3)])
nuclear_coupling_34 = NuclearOperator.userSpecifiedSurface_positionOperator(mySpace, potential_data[(3,4)])



groundStateNuclearWF = NuclearWavefunction.nuclearWavefunction(mySpace, groundStateNuclearHamiltonian = electronic_ground )


ElectronicHamiltonian = ElectronicOperator.ElectronicHamiltonian(mySpace, [(0,0, electronic_ground),
                                                            (1,1, electronic_excited_1),
                                                            (2,2, electronic_excited_2),
                                                            (3,3, electronic_excited_3),
                                                            (4,4, electronic_excited_4),
                                                            (1,2, nuclear_coupling_12),
                                                            (2,1, nuclear_coupling_12),
                                                            (2,3, nuclear_coupling_23),
                                                            (3,2, nuclear_coupling_23),
                                                            (1,3, nuclear_coupling_13),
                                                            (3,1, nuclear_coupling_13),
                                                            (3,4, nuclear_coupling_34),
                                                            (4,3, nuclear_coupling_34)])



initialEWF = ElectronicWavefunction.electronicWavefunction(mySpace,
                       listOfNuclearWavefunctions = [groundStateNuclearWF, 0, 0, 0, 0],
                       Normalize=True)
                       
dipole_indeces_op_list = []

for dipole_key in dipole_data.keys():
    if dipole_key == "R":
        continue
    i = dipole_key[0]
    j = dipole_key[1]
    operator = NuclearOperator.userSpecifiedSurface_positionOperator(mySpace, dipole_data[dipole_key])
    dipole_indeces_op_list.append((i, j, operator))
    dipole_indeces_op_list.append((j, i, operator))



xTransitionDipole = ElectronicOperator.ElectronicPositionOperator(mySpace, dipole_indeces_op_list)
yTransitionDipole = ElectronicOperator.ElectronicPositionOperator(mySpace, [])
zTransitionDipole = ElectronicOperator.ElectronicPositionOperator(mySpace, [])

transitionDipoleTuple = (xTransitionDipole, yTransitionDipole, zTransitionDipole)

