## Description
Simulations and analyses codes pertaining to Refs [1] and [2]. 
Simulation code specific to Ref [2] is provided in the subdirectory mitotic/1dsims_symmetries_and_dynamics.


*Code dependencies:* 
- https://github.com/mirnylab/mirnylib-legacy
- https://github.com/mirnylab/openmm-polymer-legacy.git

*Mitotic codes:* 
- looplib contains code needed for 1D Gillespie simulations of extrusion and analysis of loop extruder configurations. 
Codes were originally used for Ref [1], except for the module simlef_mix_slidedistrib, which is used for simulations described in Ref [2].
This is an extended version of looplib by Anton Goloborodko 
(original library: https://github.com/golobor/looplib/tree/master/looplib)
- 1dsims contains code for running 1D extrusion simulations with looplib described in Ref [1]
- 1dsims_symmetries_and_dynamics contains code for running 1D extrusion simulations with looplib described in Ref [2]

*Mitotic and Interphase codes:
- polysims contains code for running 3D extrusion simulations with openmm-polymer

## References
[1] Edward J. Banigan*, Aafke A. van den Berg*, Hugo B. Brandão*, John F. Marko, Leonid A. Mirny. 
eLife 9:e53558 (2020). Also available as bioRxiv 815340; doi: https://doi.org/10.1101/815340 
*These authors contributed equally

[2] Edward J. Banigan and L. A. Mirny. eLife 9:63528 (2020). Also available as bioRxiv 
2020.09.22.309146; doi: https://doi.org/10.1101/2020.09.22.309146
