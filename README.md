Copyright (c) 2020 Edward Banigan, Aafke van den Berg, and Hugo Brandão

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## Description
Simulations and analyses codes pertaining to Ref [1].

*Code dependencies:* 
- https://github.com/mirnylab/mirnylib-legacy
- https://github.com/mirnylab/openmm-polymer-legacy.git

*Mitotic codes:* 
- looplib contains code needed for 1D Gillespie simulations of extrusion and analysis of loop extruder configurations. 
This is an extended version of looplib by Anton Goloborodko 
(original library: https://github.com/golobor/looplib/tree/master/looplib)
- 1dsims contains code for running 1D extrusion simulations with looplib

*Mitotic and Interphase codes:
- polysims contains code for running 3D extrusion simulations with openmm-polymer

## References
[1] Edward J. Banigan*, Aafke A. van den Berg*, Hugo B. Brandão*, John F. Marko, Leonid A. Mirny.
bioRxiv 815340; doi: https://doi.org/10.1101/815340 
*These authors contributed equally


