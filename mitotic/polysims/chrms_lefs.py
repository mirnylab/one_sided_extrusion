import pickle
import os
from time import sleep
import numpy as np
from openmmlib import polymerutils
from openmmlib.polymerutils import scanBlocks
from openmmlib.openmmlib import Simulation
from openmmlib.polymerutils import grow_rw

import tools
import polymerSimSetup as pss

import pyximport; pyximport.install()
from smcTranslocator import smcTranslocatorDirectional

import gc


# -------defining parameters----------
#  -- basic loop extrusion parameters
SEPARATION = 100 # N // SEPARATION = num smcs
LIFETIME = 500
N = 60000   # number of monomers
pconfig="compact" #which polymer configuration we'll use.
cconfig="segregation" #which chain config to use
bconfig="centromere" # generally goes with cconfig...

COMPACT_FIRST = 1
PREBLOCKS=100
preblock_steps = 5000

PAUSEPROB=0.06
FRACTION_ONESIDED=1.
PAIRED=0
SLIDE=0
SLIDE_PAUSEPROB=0.995
SWITCH=0
TRANSPARENT=0

dens = 0.1
box = (N / dens) ** (1./3.)  # density = 0.1.

LEFS_ATTRACT = 0
LEF_ATTRACTION_ENERGY = 0.5
LEF_ATTRACTION_RANGE = 1.5  

ANY_ATTRACT = 0
ANY_ATTRACTION_ENERGY = 0.1
NUM_ATTRACT = N

# parameters for smc bonds 
smcBondWiggleDist = 0.2
smcBondDist = 0.5
#polymer itself
stiff = 0
polymerBondWiggleDist = 0.1

block = 0  # starting block 
smcStepsPerBlock = 1  # now doing 1 SMC step per block 
steps = 250   # steps per block (now extrusion advances by one step per block) #this is how many polymer sim steps per SMC step
saveEveryBlocks = 10   # save every 10 blocks (saving every block is now too much almost)
skipSavedBlocksBeginning = 20  # how many blocks (saved) to skip after you restart LEF positions
totalSavedBlocks = 100 #40  # how many blocks to save (number of blocks done is totalSavedBlocks * saveEveryBlocks)
restartMilkerEveryBlocks = 100 
BLOCK_SKIP = 25 #for text files for vmd

gpu_choice=0
INTEGRATOR="langevin"
ERROR_TOL=0.02
dt=80
THERMOSTAT=0.01

EXT_FORCE=0.


#folder 
main_folder= "output/"
folder = "trajectory"
NO_LOG_NAME=True#is log currently unnamed?
FLAG=""


####command line options##############################
#parse argument list
params= tools.argsList()
for p in params.arg_dict:
    print(p, params.arg_dict[p])

if "log" in params.arg_dict:
    log_name=params.arg_dict["log"]
    NO_LOG_NAME=False
    
if "pconfig" in params.arg_dict:
    pconfig=params.arg_dict["pconfig"]
if "cconfig" in params.arg_dict:
    cconfig=params.arg_dict["cconfig"]
if "bconfig" in params.arg_dict:
    bconfig=params.arg_dict["bconfig"]
    
if "density" in params.arg_dict:
    dens= float(params.arg_dict["density"])
    box = (N/dens)**(1./3.)
if "monos" in params.arg_dict:
    N=int(params.arg_dict["monos"])
    box = (N/dens)**(1./3.)
    
if "separation" in params.arg_dict:
    SEPARATION=int(params.arg_dict["separation"])
if "lifetime" in params.arg_dict:
    LIFETIME=float(params.arg_dict["lifetime"])
if "pause" in params.arg_dict:
    PAUSEPROB=float(params.arg_dict["pause"])
if "frac" in params.arg_dict:
    FRACTION_ONESIDED=float(params.arg_dict["frac"])
if "paired" in params.arg_dict:
    PAIRED=int(params.arg_dict["paired"])
if "switch" in params.arg_dict:        
    SWITCH=float(params.arg_dict["switch"])  
if "slide" in params.arg_dict:
    SLIDE=int(params.arg_dict["slide"])
if "slidepause" in params.arg_dict:
    SLIDE_PAUSEPROB=float(params.arg_dict["slidepause"])

if "transparent" in params.arg_dict:
    TRANSPARENT=int(params.arg_dict["transparent"])

if "extforce" in params.arg_dict:
    EXT_FORCE=float(params.arg_dict["extforce"])
if "polybond" in params.arg_dict:
    polymerBondWiggleDist=float(params.arg_dict["polybond"])
if "smcbond" in params.arg_dict:
    smcBondWiggleDist=float(params.arg_dict["smcbond"])

if "stiffness" in params.arg_dict:
    stiff = float(params.arg_dict["stiffness"])

if "lefsattract" in params.arg_dict:
    LEFS_ATTRACT = int(params.arg_dict["lefsattract"])
if "lefenergy" in params.arg_dict:
    LEF_ATTRACTION_ENERGY = float(params.arg_dict["lefenergy"])
if "lefrange" in params.arg_dict:
    LEF_ATTRACTION_RANGE = float(params.arg_dict["lefrange"])
if "anyattract" in params.arg_dict:
    ANY_ATTRACT = int(params.arg_dict["anyattract"])
if "anyenergy" in params.arg_dict:
    ANY_ATTRACTION_ENERGY = float(params.arg_dict["anyenergy"])
if "numattract" in params.arg_dict:
    NUM_ATTRACT = int(params.arg_dict["numattract"])

if "thermostat" in params.arg_dict:
    THERMOSTAT= float(params.arg_dict["thermostat"])
if "errortol" in params.arg_dict:
    ERROR_TOL= float(params.arg_dict["errortol"])
if "integrator" in params.arg_dict:
    INTEGRATOR = params.arg_dict["integrator"]
    if INTEGRATOR is "Brownian":
        if THEROMSTAT<1:
            print("Warning: Brownian integrator with thermostat=", THERMOSTAT)
if "dt" in params.arg_dict:
    dt = float(params.arg_dict["dt"])

if "numblocks" in params.arg_dict:
    totalSavedBlocks=int(params.arg_dict["numblocks"])
if "blocksteps" in params.arg_dict:
    steps=int(params.arg_dict["blocksteps"])
if "blockskip" in params.arg_dict:
    saveEveryBlocks=int(params.arg_dict["blockskip"])
if "pdbskip" in params.arg_dict:
    BLOCK_SKIP=int(params.arg_dict["pdbskip"])
if "startskip" in params.arg_dict:
    skipSavedBlocksBeginning=int(params.arg_dict["startskip"])

if "gpu" in params.arg_dict:
    gpu_choice=params.arg_dict["gpu"]
    
if "flag" in params.arg_dict:
    FLAG= params.arg_dict["flag"] #append a str to the end of a folder

    
# code to automatically fix intervals if inputs are not compatible.
# The code guarantees *at least* the value that was input for each
if restartMilkerEveryBlocks<100:
    restartMilkerEveryBlocks= 100
if restartMilkerEveryBlocks % saveEveryBlocks > 0:
    print("restartMilkerEveryBlocks", restartMilkerEveryBlocks, "with saveEveryBlocks", saveEveryBlocks, "not allowed")
    if saveEveryBlocks < restartMilkerEveryBlocks:
        temp=restartMilkerEveryBlocks//saveEveryBlocks
        if restartMilkerEveryBlocks%temp == 0:
            saveEveryBlocks= restartMilkerEveryBlocks//temp
        else:
            saveEveryBlocks=restartMilkerEveryBlocks//temp
            while restartMilkerEveryBlocks % saveEveryBlocks > 0:
                saveEveryBlocks = saveEveryBlocks+1
                temp=restartMilkerEveryBlocks//saveEveryBlocks
                print(temp,saveEveryBlocks)
    else:
        restartMilkerEveryBlocks=saveEveryBlocks
    print("changed to", restartMilkerEveryBlocks, "and", saveEveryBlocks)

if (skipSavedBlocksBeginning * saveEveryBlocks) % restartMilkerEveryBlocks > 0:
    print("skipSavedBlocksBeginning", skipSavedBlocksBeginning, "not allowed with saveEveryBlocks", saveEveryBlocks, "and restartMilkerEveryBlocks", restartMilkerEveryBlocks)
    if skipSavedBlocksBeginning * saveEveryBlocks < restartMilkerEveryBlocks:
        skipSavedBlocksBeginning= restartMilkerEveryBlocks//saveEveryBlocks
    else:
        while skipSavedBlocksBeginning * saveEveryBlocks % restartMilkerEveryBlocks > 0:
            skipSavedBlocksBeginning= skipSavedBlocksBeginning+1
    print("skipSavedBlocksBeginning changed to", skipSavedBlocksBeginning)

if (totalSavedBlocks * saveEveryBlocks) % restartMilkerEveryBlocks > 0:
    print("totalSavedBlocks", totalSavedBlocks, "not allowed with saveEveryBlocks", saveEveryBlocks, "and restartMilkerEveryBlocks", restartMilkerEveryBlocks)
    if totalSavedBlocks * saveEveryBlocks < restartMilkerEveryBlocks:
        totalSavedBlocks= restartMilkerEveryBlocks//saveEveryBlocks
    else:
        while (totalSavedBlocks * saveEveryBlocks)%restartMilkerEveryBlocks > 0:
            totalSavedBlocks=totalSavedBlocks+1
    print("totalSavedBlocks changed to", totalSavedBlocks)


assert restartMilkerEveryBlocks % saveEveryBlocks == 0 
assert (skipSavedBlocksBeginning * saveEveryBlocks) % restartMilkerEveryBlocks == 0 
assert (totalSavedBlocks * saveEveryBlocks) % restartMilkerEveryBlocks == 0 

savesPerMilker = restartMilkerEveryBlocks // saveEveryBlocks
milkerInitsSkip = saveEveryBlocks * skipSavedBlocksBeginning  // restartMilkerEveryBlocks
milkerInitsTotal  = (totalSavedBlocks + skipSavedBlocksBeginning) * saveEveryBlocks // restartMilkerEveryBlocks
print("Milker will be initialized {0} times, first {1} will be skipped. restartMilkerEveryBlocks={2}".format(milkerInitsTotal, milkerInitsSkip,restartMilkerEveryBlocks))


trialnumber=1
while True:
    folder= main_folder+"output{0:06d}_LE_".format(trialnumber)
    if NO_LOG_NAME:
        log_name= "log_LE_"
    if not os.path.exists(main_folder):
        os.mkdir(main_folder)
    folder=folder+"density{0}_n{1}_{2}_{3}_{4}_lifetime{5}_separation{6}_pauseprob{7}".format(dens,N,pconfig,cconfig,bconfig,
                                                                                          LIFETIME, SEPARATION, PAUSEPROB)
    if NO_LOG_NAME:
        log_name=log_name+"density{0}n{1}{2}{3}{4}lifetime{5}separation{6}pauseprob{7}".format(dens,N,pconfig,cconfig,bconfig,
                                                                                            LIFETIME, SEPARATION, PAUSEPROB)
    if FRACTION_ONESIDED < 1.0:
        folder=folder+"_frac{0}".format(FRACTION_ONESIDED)
        if NO_LOG_NAME: 
            log_name=log_name+"frac{0}".format(FRACTION_ONESIDED)
    if PAIRED:
        folder=folder+"_paired"
        if NO_LOG_NAME:
            log_name=log_name+"paired"
    if EXT_FORCE > 0:
        folder=folder+"_F{0}".format(EXT_FORCE)
        if NO_LOG_NAME:
            log_name=log_name+"F{0}".format(EXT_FORCE)
    if LEFS_ATTRACT:
        folder=folder+"_lefattr{0}".format(LEF_ATTRACTION_ENERGY)
        if NO_LOG_NAME:
            log_name=log_name+"lefattr{0}".format(LEF_ATTRACTION_ENERGY)
        if not (LEF_ATTRACTION_RANGE == 1.5):
            folder=folder+"_range{0}".format(LEF_ATTRACTION_RANGE)
            if NO_LOG_NAME:
                log_name=log_name+"lefrange{0}".format(LEF_ATTRACTION_RANGE)
    if ANY_ATTRACT:
        folder=folder+"_anyattr{0}".format(ANY_ATTRACTION_ENERGY)
        if NO_LOG_NAME:
            log_name=log_name+"anyattr{0}".format(ANY_ATTRACTION_ENERGY)
        if NUM_ATTRACT < N:
            folder=folder+"_numattr{0}".format(NUM_ATTRACT)
            if NO_LOG_NAME:
                log_name=log_name+"numattr{0}".format(NUM_ATTRACT)
    if stiff>0.:
        folder=folder+"_stiffness{0}".format(stiff)
    folder=folder+FLAG
    if NO_LOG_NAME:
        log_name=log_name+FLAG
        log_name=log_name+".txt"
    
    sleep(int(np.random.randint(0,12)))
    if os.path.exists(folder):
        trialnumber= trialnumber+1
        continue
    else:
        os.mkdir(folder)
        break


print("trial number", trialnumber)
print(folder)
sleep(1)


#################################################################################################

class smcTranslocatorMilker(object):

    def __init__(self, smcTransObject):
        """
        :param smcTransObject: smc translocator object to work with
        """
        self.smcObject = smcTransObject
        self.allBonds = []

    def setParams(self, activeParamDict, inactiveParamDict):
        """
        A method to set parameters for bonds.
        It is a separate method because you may want to have a Simulation object already existing

        :param activeParamDict: a dict (argument:value) of addBond arguments for active bonds
        :param inactiveParamDict:  a dict (argument:value) of addBond arguments for inactive bonds

        """
        self.activeParamDict = activeParamDict
        self.inactiveParamDict = inactiveParamDict


    def setup(self, bondForce,  blocks = 100, smcStepsPerBlock = 1, attractiveForce=None):
        """
        A method that milks smcTranslocator object
        and creates a set of unique bonds, etc.

        :param bondForce: a bondforce object (new after simulation restart!)
        :param blocks: number of blocks to precalculate
        :param smcStepsPerBlock: number of smcTranslocator steps per block
        :return:
        """


        if len(self.allBonds) != 0:
            raise ValueError("Not all bonds were used; {0} sets left".format(len(self.allBonds)))

        self.bondForce = bondForce
        self.attractiveForce = attractiveForce

        #precalculating all bonds
        allBonds = []
        all_immobile = []
        #get bond state at time from now until blocks into the future
        for dummy in range(blocks):
            self.smcObject.steps(smcStepsPerBlock)
            if (FRACTION_ONESIDED > 0) and (N > SEPARATION):
                left, right, inactive = self.smcObject.getSMCs()
                bonds = [(int(i), int(j)) for i,j in zip(left, right)]
                immobile_idxs = [bonds[i][inactive[i]-1] for i in range(len(inactive)) if inactive[i] > 0]
            elif not PAIRED:
                left, right = self.smcObject.getSMCs()
                bonds = [(int(i), int(j)) for i,j in zip(left, right)]
            else:
                left, right, center = self.smcObject.getSMCs()
                bonds = [(int(i), int(j)) for i,j in zip(left, center) if not int(i)==int(j)]
                bonds.extend([(int(i), int(j)) for i,j in zip(center, right) if not int(i)==int(j)])
            allBonds.append(bonds)#append list of bonds for this time

            if LEFS_ATTRACT:
                all_immobile.append(immobile_idxs)

        self.allBonds = allBonds
        self.uniqueBonds = list(set(sum(allBonds, [])))

        #adding forces and getting bond indices
        self.bondInds = []
        self.curBonds = allBonds.pop(0)
        if LEFS_ATTRACT:
            self.all_immobile = all_immobile
            self.cur_immobile = all_immobile.pop(0)

        for bond in self.uniqueBonds:
            paramset = self.activeParamDict if (bond in self.curBonds) else self.inactiveParamDict
            ind = bondForce.addBond(bond[0], bond[1], **paramset)
            self.bondInds.append(ind)

        if LEFS_ATTRACT:
            for i in self.cur_immobile:
                attractiveForce.setParticleParameters(i, (1,0)) # LEF heads to add as attractive

        self.bondToInd = {i:j for i,j in zip(self.uniqueBonds, self.bondInds)}

        if LEFS_ATTRACT:
            return self.curBonds,[], self.cur_immobile
        else:
            return self.curBonds,[]


    def step(self, context, verbose=False):
        """
        Update the bonds to the next step.
        It sets bonds for you automatically!
        :param context:  context
        :return: (current bonds, previous step bonds); just for reference
        """
        if len(self.allBonds) == 0:
            raise ValueError("No bonds left to run; you should restart simulation and run setup  again")

        pastBonds = self.curBonds
        self.curBonds = self.allBonds.pop(0)  # getting current bonds

        bondsRemove = [i for i in pastBonds if i not in self.curBonds]
        bondsAdd = [i for i in self.curBonds if i not in pastBonds]

        if LEFS_ATTRACT:
            past_immobile = self.cur_immobile
            self.cur_immobile = self.all_immobile.pop(0)
            immobile_remove = [i for i in past_immobile if i not in self.cur_immobile]
            immobile_add = [i for i in self.cur_immobile if i not in past_immobile]

        bondsStay = [i for i in pastBonds if i in self.curBonds]
        if verbose:
            print("{0} bonds stay, {1} new bonds, {2} bonds removed".format(len(bondsStay),
                                                                            len(bondsAdd), len(bondsRemove)))
        bondsToChange = bondsAdd + bondsRemove
        bondsIsAdd = [True] * len(bondsAdd) + [False] * len(bondsRemove)
        for bond, isAdd in zip(bondsToChange, bondsIsAdd):
            ind = self.bondToInd[bond]
            paramset = self.activeParamDict if isAdd else self.inactiveParamDict
            self.bondForce.setBondParameters(ind, bond[0], bond[1], **paramset)  # actually updating bonds
        self.bondForce.updateParametersInContext(context)  # now run this to update things in the context

        if LEFS_ATTRACT:
            immobile_change = immobile_add + immobile_remove
            immobile_is_add = [1]*len(immobile_add) + [0]*len(immobile_remove)
            for immobile, is_add in zip(immobile_change, immobile_is_add):
                self.attractiveForce.setParticleParameters(immobile, (is_add, 0))

            #return self.curBonds, pastBonds, cur_immobile, past_immobile
        
        return self.curBonds, pastBonds

    
    
#do initialization for the polymer
data=pss.init_positions(N,polymer=pconfig,length=box)
chains_list=pss.construct_chains_list(N,construction=cconfig)
bonds_to_add=pss.list_extra_bonds(N,bondstruc=bconfig)

 
def initModel():
    # this just inits the simulation model.
    birthArray = np.zeros(N, dtype=np.double) + 0.1
    deathArray = np.zeros(N, dtype=np.double) + 1. / LIFETIME
    stallDeathArray = deathArray
    pauseArray = np.zeros(N, dtype=np.double) + PAUSEPROB 
    slidePauseArray = np.zeros(N, dtype=np.double) + SLIDE_PAUSEPROB

    stallList = [ch[0] for ch in chains_list]
    stallList.extend([ch[1]-1 for ch in chains_list])
    print("stallList after chains:", stallList)
    if bconfig == "centromere":
        stallList.extend([N//4,3*N//4])
        print("extended stallList:", stallList)
    stallLeftArray = np.zeros(N, dtype = np.double)
    stallRightArray = np.zeros(N, dtype = np.double)
    for i in stallList:
        stallLeftArray[i]=1.
        stallRightArray[i]=1.
    
    smcNum = N // SEPARATION
    oneSidedArray = np.ones(smcNum, dtype=np.int64)
    for i in range(int((1.-FRACTION_ONESIDED)*smcNum)):
        oneSidedArray[i] = 0

    switchArray=np.zeros(smcNum, dtype=np.double) + SWITCH
    

    SMCTran = smcTranslocatorDirectional(birthArray, deathArray, stallLeftArray, stallRightArray, pauseArray,
                                         stallDeathArray, smcNum, oneSidedArray, paired=PAIRED, slide=SLIDE, slidepause=slidePauseArray, switch=switchArray, transparent=TRANSPARENT) 
    return SMCTran


#####entangle the polymers together first 
if COMPACT_FIRST:
    my_sim = pss.newSim(timestep=dt,thermostat=THERMOSTAT)

    my_sim.setup(platform="CUDA", PBC=False, PBCbox=[box, box, box],
                 GPU=gpu_choice, integrator=INTEGRATOR, errorTol=ERROR_TOL,
                 precision="mixed")  # set up GPU here
    my_sim.saveFolder(folder)
    my_sim.load(data)
    my_sim.setChains(chains_list)
    for b in bonds_to_add:
        my_sim.addBond(b["i"],b["j"],bondType=b["bondType"])

    my_sim.addHarmonicPolymerBonds(wiggleDist=polymerBondWiggleDist)
    if stiff > 0:
        my_sim.addGrosbergStiffness(stiff)

    my_sim.addPolynomialRepulsiveForce(trunc=1.5, radiusMult=1.05)
    

    #spherically confining for initialization
    if pconfig=="compact":
        my_sim.addSphericalConfinement(density=0.35, k=3.)

    my_sim.step = block

    for ii in range(PREBLOCKS): 
        my_sim.doBlock(steps=preblock_steps, increment=False)
        if ii%10==0:
            print("preblock step", ii)
    my_sim.save()
    print("printing at", my_sim.step)
    tools.print_text_file(my_sim, folder, [], my_sim.step,box)

    data = my_sim.getData()  # save data and step, and delete the simulation
    block = my_sim.step
    del my_sim
    sleep(0.2)


##############

SMCTran = initModel()  # defining actual smc translocator object 


# now polymer simulation code starts

# ------------feed smcTran to the milker---
SMCTran.steps(10*LIFETIME)  # first steps to "equilibrate" SMC dynamics. If desired of course. 
milker = smcTranslocatorMilker(SMCTran)   # now feed this thing to milker (do it once!)
#--------- end new code ------------


randomize_particles=[]
if ANY_ATTRACT:
    if NUM_ATTRACT < N:
        randomize_particles = list(zip(np.arange(N), np.random.uniform(0, 1, N)))
        randomize_particles.sort(key = lambda x: x[1])
        randomize_particles = list(np.array(np.array(randomize_particles)[:NUM_ATTRACT,0], dtype = np.int64))  #select desired num particles, randomly


for milkerCount in range(milkerInitsTotal):
    doSave = milkerCount >= milkerInitsSkip
    
    # simulation parameters are defined below 
    my_sim = pss.newSim(timestep=dt, thermostat=THERMOSTAT)
    my_sim.setup(platform="CUDA", PBC=True, PBCbox=[box, box, box], 
                 GPU=gpu_choice, integrator=INTEGRATOR, errorTol=ERROR_TOL, 
                 precision="mixed")  # set up GPU here
    my_sim.saveFolder(folder)
    my_sim.load(data)
    
    my_sim.setChains(chains_list)
    for b in bonds_to_add:
        my_sim.addBond(b["i"],b["j"],bondType=b["bondType"])
    
    my_sim.addHarmonicPolymerBonds(wiggleDist=polymerBondWiggleDist)
    if stiff > 0:
        my_sim.addGrosbergStiffness(stiff)

    if LEFS_ATTRACT:
        #everything particle given SSW, but in milker, I manually set LEF particles to have non-zero attraction energy
        if ANY_ATTRACT:
            if NUM_ATTRACT < N:
                print("Error: LEFS_ATTRACT and ANY_ATTRACT with NUM_ATTRACT<N do not function properly together")
                exit()
            else:
                tmp_E=ANY_ATTRACTION_ENERGY
        else:
            tmp_E=0.
        my_sim.addSelectiveSSWForce([], [], 
                                    repulsionEnergy=1.5, repulsionRadius=1.05, 
                                    attractionEnergy=tmp_E, attractionRadius=LEF_ATTRACTION_RANGE,
                                    selectiveAttractionEnergy=LEF_ATTRACTION_ENERGY)
    elif ANY_ATTRACT:
        if NUM_ATTRACT < N:
            #second input is list of hard particles, which is all of them.
            my_sim.addSelectiveSSWForce(randomize_particles, [], repulsionEnergy=1.5, repulsionRadius=1.05, attractionEnergy=0., selectiveAttractionEnergy=ANY_ATTRACTION_ENERGY)
            print("attractive particle list:", randomize_particles)
        else:
            my_sim.addSmoothSquareWellForce(repulsionEnergy=7.5, repulsionRadius=1.05, attractionEnergy=ANY_ATTRACTION_ENERGY)
    if not (LEFS_ATTRACT or ANY_ATTRACT):
        my_sim.addPolynomialRepulsiveForce(trunc=1.5, radiusMult=1.05)

    if EXT_FORCE > 0:
        my_sim.addExtensionalForce(EXT_FORCE)

    my_sim.step = block

    # ------------ initializing milker; adding bonds ---------
    # copied from addBond
    kbond = my_sim.kbondScalingFactor / (smcBondWiggleDist ** 2)
    bondDist = smcBondDist * my_sim.length_scale

    activeParams = {"length":bondDist,"k":kbond}
    inactiveParams = {"length":bondDist, "k":0}
    milker.setParams(activeParams, inactiveParams)
     
    # this step actually puts all bonds in and sets first bonds to be what they should be
    if not LEFS_ATTRACT:
        milker.setup(bondForce=my_sim.forceDict["HarmonicBondForce"],
                     blocks=restartMilkerEveryBlocks,   # default value; milk for 100 blocks
                     smcStepsPerBlock=smcStepsPerBlock)  # now only one step of SMC per step
    else: #just feeds in a variable telling milker what the attractiveForce is in the openmm sim, so particle params can be manipulated as LEFs move
        milker.setup(bondForce=my_sim.forceDict["HarmonicBondForce"],
                     blocks=restartMilkerEveryBlocks,   # default value; milk for 100 blocks
                     smcStepsPerBlock=smcStepsPerBlock, attractiveForce=my_sim.forceDict["Nonbonded"])  # now only one step of SMC per step
    print("Restarting milker")

    my_sim.doBlock(steps=steps, increment=False)  # do block for the first time with first set of bonds in

    ###################
    for i in range(restartMilkerEveryBlocks - 1):
        curBonds, pastBonds = milker.step(my_sim.context)  # this updates bonds. You can do something with bonds here

        if i % saveEveryBlocks == (saveEveryBlocks - 2):  
            my_sim.doBlock(steps=steps, increment = doSave)
            if doSave: 
                my_sim.save()
                pickle.dump(curBonds, open(os.path.join(my_sim.folder, "SMC{0}.dat".format(my_sim.step)),'wb'))

                if my_sim.step % BLOCK_SKIP == 0:
                    print("smc printing at", my_sim.step, "with {0} smcs".format(len(curBonds)))
                    tools.print_text_file(my_sim, folder, curBonds, my_sim.step, box, optional_list=randomize_particles)
        else:
            my_sim.integrator.step(steps)  # do steps without getting the positions from the GPU (faster)

    data = my_sim.getData()  # save data and step, and delete the simulation
    block = my_sim.step
    del my_sim

    sleep(0.2)  # wait 200ms for sanity (to let garbage collector do its magic)


os.system("mv "+log_name+" "+folder)

gc.collect()
