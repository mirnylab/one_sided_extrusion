import sys, numpy
import numpy as np


"""
Library contains;
argList- class to parse argument list. uses specifed keywords, but this kind of code
        could in principle be used elsewhere for arbitrary command line options.
        adapted from comp sim
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
                    length - int - number of lattice sites - default 10000
                    smcs - int - number of SMCs - default 100
                    frac - float - fraction of smcs that are one-sided instead of two-sided - default 1
                    paired - int - whether or not to use a "handcuff"/"butterfly" model - default 0
                    pushers - int - can active legs push inactive/diffusing/stationary legs - default 0
                    strongpush - int - can active leg push multiple inactive legs (if pushing allowed) - default 1

                    Kinetics
                    off - float - smc off rate - default 0.001
                    extend - float - smc loop extension rate - default 1
                    shrink - float - smc loop shrink rate - default 0
                    switch - float - smc leading leg switch rate - default 0
                    rebind - float - smc rebinding (and activation) time - default 10
                    slide - float - rate of sliding (in either direction, or extend only if slide2 present) of inert leg - default 0
                    slide2 - float - shrink rate for "sliding" leg - default = slide
                    beltoff - float - rate at which safety belt is released - default 0
                    belton - float - rate at which safety belt reattaches - default 0.1
                    emptybind - float - probability of rebinding a site not adjacent to a LEF if selected - default 1.0
                    loadoutward - int - whether smcs with load bias to other smcs are forced to load outward - default 0
                    lock - float - factor that dissoc time gets multiplied by if LEFs are adjacent - default 1.0
                    blocked - int - whether or not locking only applies for LEFs that have extrusion blocked (only works w/ 1-sided currently) - default 0

                    Simulation
                    lifetimes - float - num SMC lifetimes to simulate for - default 1000
                    snapshots - int - number of snapshots - default 400

                    Initialization/misc
                    log - string - name of the log file - default None + automatic naming scheme
                    restart - string - name of file in restart directory to start sim from - default "" (ignored)
                    flag - string - gets added to end of data directory - default ""
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

