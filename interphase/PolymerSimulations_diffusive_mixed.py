import pickle
import os
import time
import numpy as np
import sys
import shutil
from openmmlib import openmmlib
from openmmlib import polymerutils
from openmmlib.polymerutils import scanBlocks
from openmmlib.openmmlib import Simulation
from openmmlib.polymerutils import grow_rw
import analysis_plot_lib
sys.path.append("/net/levsha/share/homes/aafke/libs/looplib/")
if not "/net/levsha/share/homes/aafke/miniconda3/lib/python3.6/site-packages/looplib-0.1-py3.6-linux-x86_64.egg" in sys.path:
    sys.path.append("/net/levsha/share/homes/aafke/miniconda3/lib/python3.6/site-packages/looplib-0.1-py3.6-linux-x86_64.egg")
from looplib import looptools
sys.path.insert(0, "/net/levsha/share/homes/aafke/Documents/PolymerCode")
import pyximport; pyximport.install()
from smcTranslocator_diffusive_mixed.pyx import smcTranslocatorDirectional

'''
Make sure that on average we have 1 smcstep/200 3D simulations
eg by setting
smcStepsPerBLock=1
steps=200

if smcStepsPerBlock<1, reduce the number of 3D simulations per simulation block accordingly
The speed of the smc is 1-PAUSEP. 

One consideration is that the number of smc steps between 3D simulation blocks should never exceed 1 too much, as it will 
result in jerky motion of the polymer.
'''

# -------defining parameters----------
# -- basic loop extrusion parameters--
GPU = sys.argv[1]
LIFETIME = int(sys.argv[2])   # 300 Lifetime 
SEPARATION = int(sys.argv[3]) # 200 Separation LEFs in number of monomers

PAUSEP=float(sys.argv[4]) # pause prob active arm, set to 0 if not using diffusive simulations
SLIDE_PAUSEP=float(sys.argv[5]) # pause prob passive arm


TADSizes =  [400,100,200,400,100,200,400,100,200,400,100,200,400,100,200,400,100,200,400,100,200,400,100,200]
N = sum(TADSizes) # number of monomers
smcStepsPerBlock = 1#100 # I take something like 1/steppingprobability, because stepping is not determistic. I usually choose the probability of stepping to be max 0.1.
stiff = 0                 # Polymer siffness in unit of bead size
dens = 0.2 # density in beads / volume. The density can roughly be estimated by looking at the amount of DNA in a nuclear volume.
box = (N / dens) ** 0.33  # Define size of the bounding box for Periodic Boundary Conditions
data = polymerutils.grow_rw(N, int(box) - 2)  # creates a compact conformation 
block = 0  # starting block 

dt = 1*(1-PAUSEP)# Time step of each smcTranslocatorPol iteration, the time step is chosen such that MaxRate*dt=0.1,
#as this should give proper time step distributions.
steps =  int(200*dt) # nr of 3D simulation blocks btw advancing LEFs. For deterministic stepping choose 200-250 steps per block, otherwise, rescale with stepping probability. 
stg = 0.8 #Probability of stalling at TAD boundary

BELT_ON=0
BELT_OFF=1
SWITCH=0
PUSH=0
PAIRED=0
SLIDE=1# diffusive motion 
loop_prefactor=1.5  # Should be the same as in smcTranslocator
FULL_LOOP_ENTROPY=1 # Should be the same as in smcTranslocator
FRACTION_ONESIDED=1#float(sys.argv[4])#1

if dt==0:
    syst.exit('WARNING: dt=0')

'!!!!!!!Uncomment if entropy is considered'
if (SLIDE_PAUSEP < 1-np.exp(0.5*loop_prefactor)) or (SLIDE_PAUSEP < 1-0.5/np.cosh(0.5*loop_prefactor)):
    sys.exit('WARNING: slidepause prob is too small! simulations will bias toward forward sliding!')
    
#if ((PAUSEP>0.9) and (smcStepsPerBlock<10)):
#    sys.exit('WARNING: Make sure that smc steps per block is effectively 1')

saveEveryBlocks =  int(100/(smcStepsPerBlock*dt))#int(400/(smcStepsPerBlock*dt))   #  10 save every 10 blocks (saving every block is now too much almost)
skipSavedBlocksBeginning = int(20/(smcStepsPerBlock*dt))  # how many blocks (saved) to skip after you restart LEF positions
totalSavedBlocks = 4000#5000  # how many blocks to save (number of blocks done is totalSavedBlocks * saveEveryBlocks)
restartMilkerEveryBlocks = int(100/(smcStepsPerBlock*dt))   #int(100/(smcStepsPerBlock*dt))   
#Only one Hamiltonian can be loaded at a time to the simkt, but we want to update the bonds every time a LEF steps. Loading a new Hamiltonian costs a lot of time. Instead we precalculate bonds and load all positions at once as one big Hamiltonian and just change the prefactors. 

# parameters for smc bonds 
smcBondWiggleDist = 0.2
smcBondDist = 0.5

if len(sys.argv)!=6:
    sys.exit('Number of input arguments is not correct')

#folder and number of blocks
#folder = "trajectory"
folder = "/net/levsha/share/homes/aafke/Documents/PolymerSimulations/OneSidedVariations"

FullFileName=os.path.join(folder,"loopextrusion_Lifetime={0}_separation={1}_density={2}_N={3}_saveEveryBlocks={4}_totalSavedBlocks={5}_SLIDE_PAUSEPROB={6}_PAUSEPROB={7}_dt={8}_FRACTION_ONESIDED={9}_entropy".format(LIFETIME, SEPARATION, dens,N,saveEveryBlocks,totalSavedBlocks,SLIDE_PAUSEP,PAUSEP,dt,FRACTION_ONESIDED))

#FullFileName=os.path.join(folder,"NestedLoopTest2")

if os.path.isdir(FullFileName):
    var=input('Folder already exists, press 9 to stop or any other key to empty and recreate the folder and continue')
    if eval(var)==9:
        sys.exit('Stopped simulation')
    else:
        shutil.rmtree(FullFileName) # Remove folder and content
        
# assertions for easy managing code below 

assert restartMilkerEveryBlocks % saveEveryBlocks == 0 
assert (skipSavedBlocksBeginning * saveEveryBlocks) % restartMilkerEveryBlocks == 0 
assert (totalSavedBlocks * saveEveryBlocks) % restartMilkerEveryBlocks == 0 
assert smcStepsPerBlock*(1-PAUSEP)<3 # max number of steps per smc block should not be too large to prevent 'jerky' polymer motion

savesPerMilker = restartMilkerEveryBlocks // saveEveryBlocks
milkerInitsSkip = saveEveryBlocks * skipSavedBlocksBeginning  // restartMilkerEveryBlocks
milkerInitsTotal  = (totalSavedBlocks + skipSavedBlocksBeginning) * saveEveryBlocks // restartMilkerEveryBlocks
print("Time step = {0}, Milker will be initialized {1} times, first {2} will be skipped".format(dt,milkerInitsTotal, milkerInitsSkip))

# create filenames for Ekin, Epot and time

Ekin_fname = os.path.join(FullFileName,'Ekin.txt')
Epot_fname = os.path.join(FullFileName,'Epot.txt')
time_fname = os.path.join(FullFileName,'time.txt')
Rg_fname   = os.path.join(FullFileName,'Rg.txt')
Par_fname  = os.path.join(FullFileName,'Pars.txt')


def save_Es_ts_Rg():
    with open(time_fname, "a+") as time_file:
        time_file.write('%f\n'%(a.state.getTime()/openmmlib.ps))
    with open(Ekin_fname, "a+") as Ekin_file:
        Ekin_file.write('%f\n'%((a.state.getKineticEnergy())/a.N/a.kT))
    with open(Epot_fname, "a+") as Epot_file:
        Epot_file.write('%f\n'%((a.state.getPotentialEnergy()) /a.N /a.kT))
    with open(Rg_fname, "a+") as Rg_file:
        Rg_file.write('%f\n'%(analysis_plot_lib.rg(a.getData())))

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


    def setup(self, bondForce,  blocks = 100, smcStepsPerBlock = 1):
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

        #precalculating all bonds
        allBonds = []
        for dummy in range(blocks):
            self.smcObject.steps(smcStepsPerBlock)
            left, right = self.smcObject.getSMCs()
            bonds = [(int(i), int(j)) for i,j in zip(left, right)]
            allBonds.append(bonds)

        self.allBonds = allBonds
        self.uniqueBonds = list(set(sum(allBonds, []))) # 'sum' preserves order and makes one long list with bonds, 'set' creates a set with left bonds from different time points ordered from small to large and eliminates two equal bonds (also if they were created by different LEFs at different times). List turns set into a list with unique bonds at different time points.

        # adding forces and getting bond indices
        self.bondInds = []
        self.curBonds = allBonds.pop(0) # pop(0) removes and returns first list of bonds

        for bond in self.uniqueBonds:
            paramset = self.activeParamDict if (bond in self.curBonds) else self.inactiveParamDict
            ind = bondForce.addBond(bond[0], bond[1], **paramset)
            self.bondInds.append(ind)
        self.bondToInd = {i:j for i,j in zip(self.uniqueBonds, self.bondInds)}
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
        return self.curBonds, pastBonds

def initModel():
    # this just inits the simulation model. Put your previous init code here 
    birthArray = np.ones(N, dtype=np.double)*0.1
    #for i in np.cumsum(TADSizes)[0:-1]: #Increased loading at TAD borders
     #   birthArray[i+50]=10
    deathArray = np.zeros(N, dtype=np.double) + 1. / (LIFETIME/dt)
    stallLeftArray = np.zeros(N, dtype=np.double)#Probability of stalling. Choose np.zeros(N, dtype=np.double) if no stalling 
    stallRightArray = np.zeros(N, dtype=np.double)
    
    pauseArray = np.zeros(N, dtype=np.double) + PAUSEP 
    slidePauseArray = np.zeros(N, dtype=np.double) + SLIDE_PAUSEP
    
    stallDeathArray = np.zeros(N, dtype=np.double) + 1 / (LIFETIME/dt)
    smcNum = N // SEPARATION
    curPos = 0
    for size in TADSizes: # setting positions & strength of boundary elements, looping over each TAD
        curPos += size
        if curPos < len(stallLeftArray):
            stallLeftArray[curPos] = stg
            stallRightArray[curPos] = stg
                
    oneSidedArray = np.ones(smcNum, dtype=np.int64)
    for i in range(int((1.-FRACTION_ONESIDED)*smcNum)):
        oneSidedArray[i] = 0

    belt_on_array = np.zeros(smcNum, dtype=np.double) + BELT_ON
    belt_off_array = np.zeros(smcNum, dtype=np.double) + BELT_OFF

    spf=slidePauseArray*(1.-(1.-SLIDE_PAUSEP)*np.exp(-1.*loop_prefactor))
    spb=slidePauseArray*(1.-(1.-SLIDE_PAUSEP)*np.exp(loop_prefactor))
    #sps=spf+spb
    SMCTran = smcTranslocatorDirectional(birthArray, deathArray, stallLeftArray, stallRightArray, pauseArray,
                                         stallDeathArray, smcNum, oneSidedArray, paired=PAIRED, slide=SLIDE,
                                         slidepauseForward=spf, slidepauseBackward=spb, 
                                         #slidepauseSum=sps,
                                         switch=SWITCH, pushing=PUSH, belt_on=belt_on_array, belt_off=belt_off_array,SLIDE_PAUSEPROB=SLIDE_PAUSEP) 
    
    return SMCTran


SMCTran = initModel()  # defining actual smc translocator object 


# now polymer simulation code starts

# ------------feed smcTran to the milker---
SMCTran.steps(1000000)  # first steps to "equilibrate" SMC dynamics. If desired of course. 
milker = smcTranslocatorMilker(SMCTran)   # now feed this thing to milker (do it once!)
#--------- end new code ------------

for milkerCount in range(milkerInitsTotal):
    doSave = milkerCount >= milkerInitsSkip
    
    # simulation parameters are defined below 
    a = Simulation(timestep=80, thermostat=0.01)#Collision rate in inverse picoseconds, low collistion rate means ballistic like motion, default in openmmpolymer is 0.001. Motion polymer is not diffusive, this is ok for statistical average,
    #but not for dynamics of the polymer
    a.setup(platform="CUDA", PBC=True, PBCbox=[box, box, box], GPU=GPU, precision="mixed")  # set up GPU here, PBC=Periodic Boundary Conditions. Default integrator is langevin with 300 K, friction coefficient of 1/ps, step size 0.002ps
    a.saveFolder(FullFileName)
    a.load(data)
    a.addHarmonicPolymerBonds(wiggleDist=0.1) # WiggleDist controls distance at which energy of bond equals kT
    if stiff > 0:
        a.addGrosbergStiffness(stiff) # Chain stiffness is introduced by an angular potential U(theta)=stiff(1-cos(theta-Pi))
    a.addPolynomialRepulsiveForce(trunc=1.5, radiusMult=1.05) #Polynomial repulsive potential between particles. Has value trunc=3.0 at zero, stays flat until 0.6-0.7 and then drops to zero. For attraction between a selective set of particles, use LeonardJones or addSelectiveSSWForce (see blocks.py or ask Johannes)
    a.step = block

    # ------------ initializing milker; adding bonds ---------
    # copied from addBond
    kbond = a.kbondScalingFactor / (smcBondWiggleDist ** 2)
    bondDist = smcBondDist * a.length_scale

    activeParams = {"length":bondDist,"k":kbond}
    inactiveParams = {"length":bondDist, "k":0} 
    milker.setParams(activeParams, inactiveParams)
     
    # this step actually puts all bonds in and sets first bonds to be what they should be
    milker.setup(bondForce=a.forceDict["HarmonicBondForce"],
                blocks=restartMilkerEveryBlocks,   # default value; milk for 100 blocks
                 smcStepsPerBlock=smcStepsPerBlock)  # 
    print("Restarting milker")

    a.doBlock(steps=steps, increment=False)  # do block for the first time with first set of bonds in
    for i in range(restartMilkerEveryBlocks - 1):
        curBonds, pastBonds = milker.step(a.context)  # this updates bonds. You can do something with bonds here
        if i % saveEveryBlocks == (saveEveryBlocks - 2):  
            a.doBlock(steps=steps, increment = doSave)
            if doSave: 
                a.save()
                pickle.dump(curBonds, open(os.path.join(a.folder, "SMC{0}.dat".format(a.step)),'wb'))
                save_Es_ts_Rg() # save energies and time
        else:
            a.integrator.step(steps)  # do steps without getting the positions from the GPU (faster)

    data = a.getData()  # save data and step, and delete the simulation
    block = a.step
    del a
    
    time.sleep(0.2)  # wait 200ms for sanity (to let garbage collector do its magic)

with open(Par_fname,"a+") as Parfile:
    Parfile.write(" tau="+str(LIFETIME)+"\n Separation="+str(SEPARATION)+"\n N="+str(N)+"\n smcStepsPerBlock="+str(smcStepsPerBlock)+"\n stiff="+str(stiff)+"\n dens="+str(dens)+"\n block="+str(block) + "\n SaveEveryBlocks="+str(saveEveryBlocks)+"\n skipSavedBlocksBeginning="+str(skipSavedBlocksBeginning)+"\n totalSavedBlocks="+str(totalSavedBlocks)+"\n restartMilkerEveryBlocks="+str(restartMilkerEveryBlocks)+"\n smcBondWiggleDist="+str(smcBondWiggleDist)+"\n smcBondDist="+str(smcBondDist)+"\n SmcTimestep=" + str(dt)+"\n Mean TADsize"+str(np.mean(TADSizes)))