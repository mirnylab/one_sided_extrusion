import numpy as np
from openmmlib import polymerutils
import tools


import os
os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "") + ":/usr/local/cuda/lib64:/usr/local/openmm/lib"
import simtk.openmm as openmm
import simtk.unit as units
nm = units.meter * 1e-9
fs = units.second * 1e-15
ps = units.second * 1e-12
from openmmlib.openmmlib import Simulation

class newSim(Simulation):
    """
    class in case you want to add methods in addition to what's already included in openmm-polymer
    """
    def sample_func():
        pass

    def addExtensionalForce(self,magnitude=0):#,monolist=None):
        """
        Add a force that pulls first monomer toward x=-\infty last monomer toward x=\infty (or vice versa?)
        """
        energy= ("direc * F * x")
        extforce= self.mm.CustomExternalForce(energy)
        self.forceDict["ExtensionalForce"]=extforce
        extforce.addGlobalParameter("F", magnitude * self.kT / nm)
        extforce.addPerParticleParameter("direc")
        extforce.addParticle(0, [1])
        extforce.addParticle(int(self.N)-1, [-1])



def init_positions(num_monos, polymer="isotropic",length=50):
    """
    Create list of monomer positions to be fed into sim.load()
    """
    step_size= 1.0
    
    if polymer == "boxed":
        step_size= length/np.sqrt(num_monos)
        init_polymer= tools.create_boxed_random_walk(step_size, num_monos, max_length=length)
        print("polymer is boxed")
    elif polymer == "extended":
        step_size= length/np.sqrt(num_monos)
        init_polymer= polymerutils.create_random_walk(1,num_monos)
        print("polymer is extended")
    elif polymer == "compact":
        init_polymer=polymerutils.grow_rw(num_monos, int(length) - 2)
        print("polymer is compact")
    else: # isotropic
        init_polymer = tools.create_isotropic_random_walk(step_size, num_monos)
        print("polymer is isotropic, input was", polymer)

    return init_polymer



def construct_chains_list(num_monos,construction="segregation"):
    """
    Create list of tuples for chains
    """
    
    if construction == "single":
        chains_list=[(0,num_monos,0)]
        print("chain construction is 'single'")
    elif construction == "ring":
        chains_list=[(0,num_monos,1)]
        print("chain construction is 'ring'")
    else: #segregation
        chains_list=[(0,num_monos//2,0),(num_monos//2,num_monos,0)]
        print("chain construction is segregation, input was", construction)
    
    return chains_list


def list_extra_bonds(num_monos,bondstruc="centromere"):
    """
    make a a list of bonds to add.
    option "centromere" assumes chains were constructed with "segregation"
    """
    
    if bondstruc == "centromere":
        bondlist=[{"i":num_monos//4, "j":3*num_monos//4,"bondType":"Harmonic"}]
        print("bondstruc is centromere")
    else:#none.
        bondlist=[]
        print("bondstruc is none, input was", bondstruc)
    return bondlist
   
