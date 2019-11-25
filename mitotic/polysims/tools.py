import sys, numpy
import numpy as np


"""
Library contains;
argList- class to parse argument list for compartment sim. uses speciifed keywords, but this kind of code
        could in principle be used elsewhere for arbitrary command line options
create_isotropic_random_walk()- make an isotropic random walk
create_boxed_random_walk()- quasi-isotropic random walk in a boxed 3D region
rand_direction() - choose a random direction (i.e., point on a sphere)
print_text_file - print a text file that can be read by vmd
"""


###########################################################################
class argsList:
    #A simple class to parse an arg list of the form param1=value1, etc., with 
    #spaces delineating arguments. e.g., execute:
    #python run_sim.py param1=0 param2=1000 param3=True
    #Help option specific to my compartment sims, but basic structure can be reused.

    def __init__(self):
        self.arg_dict= {}
        self.parseArgs()
    
    def parseArgs(self):
        if "?" in sys.argv:
            print("""
                Format:
                    param1=value1 param2=value2 ...etc...
                Options:
                    Structure/config
                    monos - int - number of monomers - default 60000
                    density - float - monomer volume fraction - default 0.1
                    pconfig - str - determines how polymer initialized - default "compact"
                    cconfig - str - determines how chains are initialized - default "segregation"
                    bconfig - str - determines extra bonds to add - default "centromere"
                    extforce - float - extensional force on two ends of polymer - default 0
                    polybond - float - wiggle distance for polymer bonds - default 0.1
                    smcbond - float - wiggle distance for smc bonds - default 0.2 
                    stiffness - float - polymer stiffness - default 0

                    lefsattract - int - do inactive sides of 1-sided lefs attract each other - default 0
                    lefenergy - float - energy w/ which lefs attract each other - default 0.5
                    lefrange - float - range of attractive potential between lefs - default 1.5
                    anyattract - int - allow non-specific, static attractions - default 0
                    anyenergy - float - energy for non-specific attraction - default 0.1
                    numattract - int - number of non-specific attractions - default 60000

                    LEF params
                    lifetime - float - mean time of LEF on polymer - default 500
                    separation - int - mean distance between LEFs - default 100
                    frac - float - fraction of LEFs that are one-sided - default 1.
                    pause - float - probability of a pause - default 0.06 
                    paired - int - whether or not we use "paired" (dimer) LEFs; use with frac=0 - default 0
                    slide - int - diffusive sliding of inactive arm; use with frac=1 - default 0
                    slidepause - float - pausing probability for sliding - default 0.995
                    switch - float - switch prob - default 0

                    Simulation
                    thermostat - double - collision rate / drag - default 0.01
                    errortol - double - error tolerance for variable Langevin integrator - default 0.02                    
                    integrator - string - which integrator to use - default "langevin"
                    dt - float - size of timestep with fixed integrator - default 80 (fs)
                    
                    Sim time
                    numblocks - int - num blocks to save - default 100
                    blocksteps - int - num polymer steps per smc step - default 250
                    blockskip - int - num blocks to skip between saves - default 10
                    startskip - int - num cycles of smc to skip before starting saves - default 20
                    Note: total SMC steps = (numblocks+startskip)*blockskip = 1200 default
                    
                    Initialization/misc
                    log - string - name of the log file - default None + automatic naming scheme
                    flag - string - gets added to end of data directory - default ""
                    gpu - int - choice of GPU - default "default"
                """)
            exit()
        
        if len(sys.argv) > 1:
            for element in sys.argv[1:]:
                var_name=""
                var_value=""

                if "=" not in element: #assume end of commands
                    break
                
                for k, char in enumerate(element):
                    if k < element.index("="):
                        var_name= var_name+char
                    elif k > element.index("="):
                        var_value= var_value+char
            
                self.arg_dict[var_name]=var_value




def create_isotropic_random_walk(step_size, N, segment_length=1):
    """
    Create an isotropic random walk of positions, e.g., for an initial polymer config.
    Required arguments are a step size and number of steps/particles
    segment_length, i.e., particles per step is default 1
    """
    theta = np.repeat(np.random.uniform(0., 1., N // segment_length + 1),
                      segment_length)
    theta = np.pi * theta[:N]
    phi = np.repeat(np.random.uniform(0., 1., N // segment_length + 1),
                  segment_length)
    phi = 2.0 * np.pi * phi[:N]
    dists= np.repeat(np.random.uniform(step_size-.1,step_size+.1,N//segment_length + 1), segment_length)
    dists= dists[:N]
    x = dists * numpy.cos(phi) * numpy.sin(theta)
    y = dists * numpy.sin(phi) * numpy.sin(theta)
    z = dists * numpy.cos(theta)
    x, y, z = np.cumsum(x), np.cumsum(y), np.cumsum(z)
    return np.vstack([x, y, z]).T



def create_boxed_random_walk(step_size, N, segment_length=1, max_length=100):
    """
    Random walk, but gets rescaled to be contained within a cube of size max_legnth^3
    """
    theta = np.repeat(np.random.uniform(0., 1., N // segment_length + 1),
                      segment_length)
    theta = np.pi * theta[:N]
    phi = np.repeat(np.random.uniform(0., 1., N // segment_length + 1),
                  segment_length)
    phi = 2.0 * np.pi * phi[:N]
    dists= np.repeat(step_size*np.random.uniform(0.9,1.1,N//segment_length + 1), segment_length)
    dists= dists[:N]
    x = dists * numpy.cos(phi) * numpy.sin(theta)
    y = dists * numpy.sin(phi) * numpy.sin(theta)
    z = dists * numpy.cos(theta)
    x, y, z = np.cumsum(x), np.cumsum(y), np.cumsum(z)
    
    maxX=max(x)
    minX=min(x)
    maxY=max(y)
    minY=min(y)
    maxZ=max(z)
    minZ=min(z)
    
    x = np.array([xi * max_length / (maxX-minX) for xi in x])
    y = np.array([yi * max_length / (maxY-minY) for yi in y])
    z = np.array([zi * max_length / (maxZ-minZ) for zi in z])

    
    return np.vstack([x, y, z]).T


def rand_direction(dimension=3):
    """
    Choose a random direction / point on surface of a sphere.  Default is 3d
    """
    sumsq=999.
    while sumsq>1.:
        v=np.array([2*(np.random.uniform()-0.5) for i in range(dimension)])
        sumsq=np.sum(v[i]**2 for i in range(dimension))
    return v/np.sqrt(sumsq)
    


def print_text_file(sim, folder_name, smc_list, step, boxL, optional_list=[]):
    """Assumes centromere / segregation"""
    data = sim.getData()
    if len(data[0]) != 3:
        data=np.transpose(data)
    if len(data[0]) != 3:
        print("wrong data?")
        return

    occ_list = [smc[0] for smc in smc_list]
    occ_list.extend([smc[1] for smc in smc_list])
    maxL=1.1*boxL

    outfile= folder_name+"/coords_{0}.pdb".format(step)
    with open(outfile, "w") as myfile:
        for ii, particle in enumerate(data):
            idnum=ii
            
            #print everything twice to allow visualization in VMD...
            if ii<len(data)//2:
                color='A'
            else:
                color='B'
            if (ii==len(data)//4) or (ii==3*len(data)//4):
                color='E'
            if ii in optional_list:
                color='F'
            if idnum<99999:
                myfile.write("ATOM  {0:5d}  O   ASP {1}{2:4d}".format(idnum+1, color, 1))
            else:
                myfile.write("ATOM  {0:5d} O   ASP {1}{2:4d}".format(idnum+1, color, 1))
            myfile.write("      ")
            
            if ii not in occ_list:
                x=str(particle[0])
                y=str(particle[1])
                z=str(particle[2])
            else:
                x=str(maxL)
                y=str(maxL)
                z=str(maxL)
            if len(x)<6:
                x=x+" "*(6-len(x))
            if len(y)<6:
                y=y+" "*(6-len(y))
            if len(z)<6:
                z=z+" "*(6-len(z))
            myfile.write("{0}  {1}  {2}  1.00  0.00".format(x[:6], y[:6], z[:6]))
            myfile.write("\n")

        for ii, particle in enumerate(data):
            idnum=ii+len(data)
            ###now smcs
            if ii<len(data)//2:
                color='C'
            else:
                color='D'
            if (ii==len(data)//4) or (ii==3*len(data)//4):
                color='E'
            if idnum<99999:
                myfile.write("ATOM  {0:5d}  O   ASP {1}{2:4d}".format(idnum+1, color, 1))
            else:
                myfile.write("ATOM  {0:5d} O   ASP {1}{2:4d}".format(idnum+1, color, 1))
            myfile.write("      ")
            if ii in occ_list:
                x=str(particle[0])
                y=str(particle[1])
                z=str(particle[2])
            else:
                x=str(maxL)
                y=str(maxL)
                z=str(maxL)
            if len(x)<6:
                x=x+" "*(6-len(x))
            if len(y)<6:
                y=y+" "*(6-len(y))
            if len(z)<6:
                z=z+" "*(6-len(z))
            myfile.write("{0}  {1}  {2}  1.00  0.00".format(x[:6], y[:6], z[:6]))
            myfile.write("\n")        
            
            
