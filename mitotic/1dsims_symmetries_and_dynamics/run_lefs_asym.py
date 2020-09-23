import numpy as np
from mirnylib import h5dict
import pyximport; pyximport.install(
            setup_args={"include_dirs":np.get_include()},
                reload_support=True)
from looplib import looptools, simlef_onesided, simlef_mix, simlef_paired, simlef_pushers, simlef_mix_slidedistrib
import os, sys, glob, shelve, time

import tools



#####
###defaults
LENGTH=10000
NUM_SMC=100
OFF_RATE=0.001
EXTEND_RATE=1
SLIDING_RATE=0 # rate at which inert legs extend (and shrink, but only if SLIDING_RATE2 is not present)
SLIDING_RATE2=SLIDING_RATE # rate at which sliding legs shrink
ASYM_DISTRIB="" # for special cases with distributions of asymmetries
SHRINK_RATE=0
SWITCH_RATE=0
REBIND_TIME=10
LIFETIMES=400#1000#that's a lot. # number of smc lifetimes to simulate for
NUM_SNAPS=400 # number of snapshots to take
FRACTION_ONE_SIDED=1
PROC_NAME=b"extrusion with asymmetry" # need it to be b for bytes
#activation times by default are set by rebinding times
PAIRED=False #boolean variable to run simlef_paired instead.
PUSHERS=False
EMPTY_SITE_REBIND=1.0
LOAD_OUTWARD=0
LOCK_LIFE=1.0
BLOCKED_LOCK = 0

NUM_2 = 0
EXTEND_2 = 1
OFF_2 = 0.01

log_name="log_default_name.txt"
FLAG=""

####
###grab stuff from command line
#parse argument list
params= tools.argsList()
for p in params.arg_dict:
    print(p, params.arg_dict[p])

if "log" in params.arg_dict:
    log_name=params.arg_dict["log"]

if "length" in params.arg_dict:
    LENGTH=int(params.arg_dict["length"])
if "smcs" in params.arg_dict:
    NUM_SMC=int(params.arg_dict["smcs"])

if "off" in params.arg_dict:
    OFF_RATE=float(params.arg_dict["off"])
    OFF_2 = OFF_RATE
if "extend" in params.arg_dict:
    EXTEND_RATE=float(params.arg_dict["extend"])
    EXTEND_2 = EXTEND_RATE
if "shrink" in params.arg_dict:
    SHRINK_RATE=float(params.arg_dict["shrink"])
if "switch" in params.arg_dict:
    SWITCH_RATE=float(params.arg_dict["switch"])
if "rebind" in params.arg_dict:
    REBIND_TIME=float(params.arg_dict["rebind"])
if "emptybind" in params.arg_dict:
    EMPTY_SITE_REBIND=float(params.arg_dict["emptybind"])
if "loadoutward" in params.arg_dict:
    LOAD_OUTWARD=int(params.arg_dict["loadoutward"])

if "lock" in params.arg_dict:
    LOCK_LIFE=float(params.arg_dict["lock"])
if "blocked" in params.arg_dict:
    BLOCKED_LOCK=int(params.arg_dict["blocked"])

if "smcs2" in params.arg_dict:
    NUM_2 = int(params.arg_dict["smcs2"])
if "extend2" in params.arg_dict:
    EXTEND_2=float(params.arg_dict["extend2"])
if "off2" in params.arg_dict:
    OFF_2=float(params.arg_dict["off2"])

if "slide" in params.arg_dict:
    SLIDING_RATE=float(params.arg_dict["slide"])
    SLIDING_RATE2=SLIDING_RATE
if "slide2" in params.arg_dict:
    SLIDING_RATE2=float(params.arg_dict["slide2"])
if "asymcase" in params.arg_dict:
    ASYM_DISTRIB = params.arg_dict["asymcase"]

if "frac" in params.arg_dict:
    FRACTION_ONE_SIDED=float(params.arg_dict["frac"])
    #PROC_NAME=b"mixed extrusion"

if "paired" in params.arg_dict:
    if int(params.arg_dict["paired"]) == 1:
        PAIRED=True
if "pushers" in params.arg_dict:
    if int(params.arg_dict["pushers"]) == 1:
        PUSHERS=True

if "lifetimes" in params.arg_dict:
    LIFETIMES=float(params.arg_dict["lifetimes"])
if "snapshots" in params.arg_dict:
    NUM_SNAPS=int(params.arg_dict["snapshots"])

if "flag" in params.arg_dict:
    FLAG=params.arg_dict["flag"]



#actually create the params dict
p = {}
p['L'] = LENGTH
p['N'] = NUM_SMC
if NUM_2 == 0:
    p['R_OFF'] = OFF_RATE
    p['R_EXTEND'] = EXTEND_RATE
else:
    p['R_OFF'] = [OFF_RATE if i < NUM_SMC-NUM_2 else OFF_2 for i in range(NUM_SMC)]
    p['R_EXTEND'] = [EXTEND_RATE if i < NUM_SMC-NUM_2 else EXTEND_2 for i in range(NUM_SMC)]

p['R_SHRINK'] = SHRINK_RATE
p['R_SWITCH'] = SWITCH_RATE
p['REBINDING_TIME']=REBIND_TIME
p['ACTIVATION_TIMES']=np.random.exponential(p['REBINDING_TIME'],p['N'])
p['ACTIVATION_TIMES'][0]=0.

p['BINDING_PROB']=EMPTY_SITE_REBIND
p['LOAD_OUTWARD']=LOAD_OUTWARD

p['LOCK_LIFE']=LOCK_LIFE
p['BLOCKED_LOCK']=BLOCKED_LOCK

if not (ASYM_DISTRIB == ""):
    SLIDING_RATE = np.zeros(NUM_SMC)
    SLIDING_RATE2 = np.zeros(NUM_SMC)
    if ASYM_DISTRIB == "case1":
        print("asymmetry case 1")
        for i in range(NUM_SMC):
            if i < 0.5*NUM_SMC:
                asym_score = np.random.uniform(0.0,0.8)
            else:
                asym_score = np.random.uniform(0.8,1.0)
            if not type(p['R_EXTEND']) in [list, np.ndarray]:
                #print("single ext rate")
                SLIDING_RATE[i] = EXTEND_RATE*(1.-asym_score)/(1.+asym_score)
            else:
                #print("multiple ext rates")
                SLIDING_RATE[i] = p['R_EXTEND'][i]*(1.-asym_score)/(1.+asym_score)
    elif ASYM_DISTRIB == "case2":
        print("asymmetry case 2")
        for i in range(NUM_SMC):
            if i < 0.5*NUM_SMC:
                asym_score = np.random.uniform(0.0,0.8)
            else:
                asym_score = 1.0
            if not type(p['R_EXTEND']) in [list, np.ndarray]:
                #print("single ext rate")
                SLIDING_RATE[i] = EXTEND_RATE*(1.-asym_score)/(1.+asym_score)
            else:
                #print("multiple ext rates")
                SLIDING_RATE[i] = p['R_EXTEND'][i]*(1.-asym_score)/(1.+asym_score)
    elif ASYM_DISTRIB == "case3":
        print("asymmetry case 3")
        for i in range(NUM_SMC):
            if i < 0.4*NUM_SMC:
                asym_score = np.random.uniform(0.0,0.2)
            elif i < 0.85*NUM_SMC:
                asym_score = np.random.uniform(0.2,0.6)
            else:
                asym_score = np.random.uniform(0.6,1.0)
            if not type(p['R_EXTEND']) in [list, np.ndarray]:
                #print("single ext rate")
                SLIDING_RATE[i] = EXTEND_RATE*(1.-asym_score)/(1.+asym_score)
            else:
                #print("multiple ext rates")
                SLIDING_RATE[i] = p['R_EXTEND'][i]*(1.-asym_score)/(1.+asym_score)
    elif ASYM_DISTRIB == "case4":
        print("asymmetry case 4")
        for i in range(NUM_SMC):
            if i < 0.6*NUM_SMC:
                asym_score = np.random.uniform(0.0,0.9)
            else:
                asym_score = 1.0
            if not type(p['R_EXTEND']) in [list, np.ndarray]:
                #print("single ext rate")
                SLIDING_RATE[i] = EXTEND_RATE*(1.-asym_score)/(1.+asym_score)
            else:
                #print("multiple ext rates")
                SLIDING_RATE[i] = p['R_EXTEND'][i]*(1.-asym_score)/(1.+asym_score)
    elif ASYM_DISTRIB == "case5":
        ext_array = np.zeros(NUM_SMC)
        shrink_array = np.zeros(NUM_SMC)
        for i in range(NUM_SMC):
            if i < 0.1*NUM_SMC: #quasi-symmetric
                ext_array[i] = 0.69
                shrink_array[i] = 0.098
                SLIDING_RATE[i] = 0.44
                SLIDING_RATE2[i] = 0.16
            elif i < 0.2*NUM_SMC: #one-sided
                ext_array[i] = 1.2
                shrink_array[i] = 0.067
                SLIDING_RATE[i] = 0.28
                SLIDING_RATE2[i] = 0.22
            elif i < 0.3*NUM_SMC: # symmetric!
                ext_array[i] = 0.62
                shrink_array[i] = 0.14
                SLIDING_RATE[i] = 0.62
                SLIDING_RATE2[i] = 0.14
            elif i < 0.4*NUM_SMC: # mostly 1-sided
                ext_array[i] = 0.17
                shrink_array[i] = 0.019
                SLIDING_RATE[i] = 0.093
                SLIDING_RATE2[i] = 0.071
            elif i < 0.5*NUM_SMC: # mostly one-sided
                ext_array[i] = 0.24   
                shrink_array[i] = 0.050
                SLIDING_RATE[i] = 0.074
                SLIDING_RATE2[i] = 0.045
            elif i < 0.6*NUM_SMC: # mostly one-sided
                ext_array[i] = 0.17     
                shrink_array[i] = 0.027
                SLIDING_RATE[i] = 0.077 
                SLIDING_RATE2[i] = 0.069
            elif i < 0.7*NUM_SMC: # mostly one-sided
                ext_array[i] = 0.96     
                shrink_array[i] = 0.32
                SLIDING_RATE[i] = 0.38 
                SLIDING_RATE2[i] = 0.32
            elif i < 0.8*NUM_SMC: # mostly one-sided
                ext_array[i] = 0.73    
                shrink_array[i] = 0.17 
                SLIDING_RATE[i] = 0.38 
                SLIDING_RATE2[i] = 0.31
            elif i < 0.9*NUM_SMC: # mostly one-sided
                ext_array[i] = 0.92
                shrink_array[i] = 0.31 
                SLIDING_RATE[i] = 0.46 
                SLIDING_RATE2[i] = 0.4
            else:
                ext_array[i] = 0.75
                shrink_array[i] = 0.20 
                SLIDING_RATE[i] = 0.097
                SLIDING_RATE2[i] = 0.11
        p['R_EXTEND'] = ext_array
        p['R_SHRINK'] = shrink_array




p['R_SLIDE']=[SLIDING_RATE,SLIDING_RATE2]
p['FRACTION_ONE_SIDED']=FRACTION_ONE_SIDED

p['T_MAX_LIFETIMES'] = LIFETIMES
p['T_MAX'] = p['T_MAX_LIFETIMES']*(1. / min([OFF_RATE,OFF_2]) + p['REBINDING_TIME']) if OFF_RATE > 0  else p['T_MAX_LIFETIMES']*(p['L']/min([EXTEND_RATE,EXTEND_2])+p['REBINDING_TIME'])
p['N_SNAPSHOTS'] = NUM_SNAPS
PROC_NAME=b"extrusion asymcases"
p['PROCESS_NAME'] = PROC_NAME



#do the simulation
print("commencing simulation...")
#note: simlef_mix runs faster than simlef_onesided. In any case, 
# just use simlef_mix for all sims except for switch rate > 0
if PUSHERS:#180521: in principle, I think PUSHERS should work with slide nonzero and even regular two-sided, but don't worry for now
    l_sites, r_sites, lagging_legs, ts = simlef_pushers.simulate(p)
elif PAIRED:
    l_sites, r_sites, c_sites, ts = simlef_paired.simulate(p)
else:
    if type(SLIDING_RATE) in [list, np.ndarray]:
        l_sites, r_sites, lagging_legs, ts = simlef_mix_slidedistrib.simulate(p)
    elif (FRACTION_ONE_SIDED<1) or (SLIDING_RATE>0):
        l_sites, r_sites, lagging_legs, ts = simlef_mix.simulate(p) # historical accident that I return lagging legs here
    else:
        l_sites, r_sites, leading_legs, ts = simlef_onesided.simulate(p)





#calculate
compacted=[]
for t in range(p['N_SNAPSHOTS']//2,p['N_SNAPSHOTS'],1):
    #parents is an array listing -1 or a number for each LEF
    parents = looptools.get_parent_loops(l_sites[t], r_sites[t])
    #which entries of parents==-1, so identifying non-nested loops
    root_loops_idxs = np.where(parents == -1)[0]
    children = looptools.get_loop_branches(parents)#find children of each LEF, some arrays are empty.
    #################################################################
    in_loops=np.sum(r_sites[t][root_loops_idxs] - l_sites[t][root_loops_idxs])
    compacted.append(in_loops/p['L'])

comp_av=np.mean(compacted)
comp_stdev=np.std(compacted)

#output data
fname="output/smc_traj_L{l}_N{n}_off{off}_ext{ext}_shr{shr}_sw{sw}_bind{bind}".format(l=p['L'],n=p['N'],
                                                                                      off=OFF_RATE,ext=EXTEND_RATE,shr=SHRINK_RATE,
                                                                                      sw=p['R_SWITCH'],bind=p['REBINDING_TIME'])
#if FRACTION_ONE_SIDED<1:
fname=fname+"_frac{frac}".format(frac=FRACTION_ONE_SIDED)
if type(SLIDING_RATE) in [list, np.ndarray]:
   fname=fname+"_asym{case}".format(case=ASYM_DISTRIB)
elif SLIDING_RATE>0:
    fname=fname+"_sl{slide}".format(slide=SLIDING_RATE)
    if SLIDING_RATE2!=SLIDING_RATE:
        fname=fname+"_slII{slide2}".format(slide2=SLIDING_RATE2)
if PAIRED:
    fname=fname+"_paired"
if PUSHERS:
    fname=fname+"_push"

if EMPTY_SITE_REBIND<1.0:
    if LOAD_OUTWARD:
        fname=fname+"_emptybindLO{eb}".format(eb=EMPTY_SITE_REBIND)
    else:
        fname=fname+"_emptybind{eb}".format(eb=EMPTY_SITE_REBIND)

if not (LOCK_LIFE == 1.0):
    if BLOCKED_LOCK == 0:
        fname=fname+"_lock{ll}".format(ll=LOCK_LIFE)
    else:
        fname=fname+"_blockedlock{ll}".format(ll=LOCK_LIFE)

if not (NUM_2 == 0):
    fname=fname+"_twopop_NII{n}_extII{v}_offII{k}".format(n=NUM_2, v=EXTEND_2, k=OFF_2)

fname=fname+FLAG+".dat"

with open(fname, "w") as myfile:
    myfile.write(str(NUM_SMC)+" "+str(NUM_SNAPS)+" "+str(comp_av)+" "+str(comp_stdev)+"\n")
    for ii in range(NUM_SNAPS):
        myfile.write(str(ts[ii]))
        myfile.write(" ")
        for nn in range(NUM_SMC):
            myfile.write(str(l_sites[ii][nn])+" ")
        for nn in range(NUM_SMC):
            myfile.write(str(r_sites[ii][nn])+" ")
        if PAIRED:
            for nn in range(len(c_sites[ii])):
                myfile.write(str(c_sites[ii][nn])+" ")
        else:
            if (FRACTION_ONE_SIDED<1) or (type(SLIDING_RATE) in [list, np.ndarray]) or (SLIDING_RATE>0):
                for nn in range(len(lagging_legs)):
                    myfile.write(str(lagging_legs[nn])+" ") #oops... Just record lagging legs insted of leading...
            else:
                for nn in range(NUM_SMC):
                    myfile.write(str(leading_legs[ii][nn])+" ")
        myfile.write("\n")
    
    myfile.write(str(p))

