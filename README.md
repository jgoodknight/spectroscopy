# Spectroscopy

Makes nonlinear spectroscopy of vibrational systems easy and object-oriented.

The user need only define a system file and then feed its constituents into an experiment object and observables such as Absorption, Pump Probe and Transient Grating can easily be calculated.

For details on the physics of what's going on here, see the introduction section of my forthcoming thesis which will be published at my github page: https://github.com/jgoodknight/

For details on how to use the code, I think it will be much more instructive to look at the examples folder.

Install by 'python setup.py install' then you'll be able to run the examples provided in the examples folder.  You must write a python script to define the system you wish to do calculations on and I've provided many such systems in the systems folder.  

As a note, while programming this I learned a ton about both the physics and the computer science of what I was doing, but I only felt like I had the time to fix the physics.  But I never the time to go back and make huge structural changes that would be needed for it to run as fast as it could.  I hope to someday get back to it but for now, I think it works great if you're not terribly resource constrained and it could be great for educational demonstrations.
