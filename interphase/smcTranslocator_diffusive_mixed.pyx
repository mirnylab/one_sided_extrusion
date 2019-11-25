#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=True

import numpy as np
cimport numpy as np 
import cython
cimport cython 
import sys
sys.path.insert(0, "/net/levsha/share/homes/aafke/libs/looplib/")
sys.path.insert(0, "/net/levsha/share/homes/aafke/libs/looplib/looplib/looptools_c")
#sys.path.insert(0, "/net/levsha/share/homes/aafke/libs/looplib/looplib/looptools.py")
if not "/net/levsha/share/homes/aafke/miniconda3/lib/python3.6/site-packages/looplib-0.1-py3.6-linux-x86_64.egg" in sys.path:
    sys.path.append("/net/levsha/share/homes/aafke/miniconda3/lib/python3.6/site-packages/looplib-0.1-py3.6-linux-x86_64.egg")
from looplib import looptools


cdef extern from "<stdlib.h>":
    double drand48()   

cdef cython.double randnum():
    return drand48()


cdef class smcTranslocatorDirectional(object):
    cdef int N
    cdef int M
    cdef cython.double [:] emission
    cdef cython.double [:] stallLeft
    cdef cython.double [:] stallRight
    cdef cython.double [:] stallFalloff
    cdef cython.double [:] falloff
    cdef cython.double [:] pause
    cdef cython.double [:] cumEmission
    cdef cython.long [:] SMCs1 #list of where the SMCs reside (left; SMCs2 is right); -1 for unbound SMC
    cdef cython.long [:] SMCs2
    cdef cython.long [:] SMCs3 #list of central site if dealing with case of "paired" SMCs
    cdef cython.long [:] stalled1 #list of whether or not each SMC is stalled
    cdef cython.long [:] stalled2
    cdef cython.long [:] occupied # list of whether or not site is occupied
    cdef cython.long [:] onesided # list of whether or not each SMC is one-sided
    cdef cython.long [:] inactive_side # list of which of the two sides is inactive. needed to differentiate between two sides in case where active side runs into a stall site
    cdef cython.int paired # whether or not SMCs are "paired" 
    cdef cython.int slide # whether or not one leg of the one-sided SMC diffuses passively instead of extruding
    cdef cython.double [:] slidepause # pause probability for diffusing arms, so we can regulate v_diffuse / v_active ratio
    cdef cython.double [:] slidepauseForward # pause probability for diffusing arms, so we can regulate v_diffuse / v_active ratio
    cdef cython.double [:] slidepauseBackward
    cdef cython.int [:] sliding_on # used to determine whether safety belt is attached or not
    cdef cython.double [:] belt_off_rate
    cdef cython.double [:] belt_on_rate
    cdef cython.double [:] belt_off_rate_original # since belt_off_rate can be set to 0 sometimes, I need to store this variable for reuse later.
    cdef cython.long [:] belt_attached
    #cdef cython.double [:] slidepauseSum
    cdef cython.double [:] switch
    cdef cython.long [:] switch_life
    cdef cython.int pushing

    cdef int maxss
    cdef int curss
    cdef cython.long [:] ssarray  
    
    cdef cython.double FULL_LOOP_ENT
    cdef cython.double loop_prefact
    cdef cython.double SLIDE_PAUSEP
 
    
    def __init__(self, emissionProb, deathProb, stallProbLeft, stallProbRight, pauseProb, stallFalloffProb,  numSmc, onesided, paired=0, slide=0, slidepauseForward=0,slidepauseBackward=0, switch=0, pushing=0, belt_on=0, belt_off=0, loop_prefactor=1.5, FULL_LOOP_ENTROPY=1,SLIDE_PAUSEPROB=0.9):
        #FULL_LOOP_ENTROPY should be 1 for entropy
        #for diffusive belt_off should be 1, belt_on=0
        #for one sided belt_on=1, belt_off=0
        #check:if slide=0, everything related to sliding (including entropy) is ignored.
        #onesided is array with length of nr SMCs
        emissionProb[0] = 0
        emissionProb[len(emissionProb)-1] = 0
        emissionProb[stallProbLeft > 0.9] = 0        
        emissionProb[stallProbRight > 0.9] = 0        
        
        self.N = len(emissionProb)
        self.M = numSmc
        self.emission = emissionProb
        self.stallLeft = stallProbLeft
        self.stallRight = stallProbRight
        self.falloff = deathProb
        self.pause = pauseProb
        cumem = np.cumsum(emissionProb)
        cumem = cumem / float(cumem[len(cumem)-1])
        self.cumEmission = np.array(cumem, np.double)
        self.SMCs1 = np.zeros((self.M), int)
        self.SMCs2 = np.zeros((self.M), int)
        self.SMCs3 = np.zeros((self.M), int)
        self.stalled1 = np.zeros(self.M, int)
        self.stalled2 = np.zeros(self.M, int)
        self.occupied = np.zeros(self.N, int)
        self.stallFalloff = stallFalloffProb
        self.occupied[0] = 1
        self.occupied[self.N - 1] = 1
        self.maxss = 1000000
        self.curss = 99999999

        self.onesided=onesided
        self.paired=paired
        self.slide=slide
        self.pushing=pushing
        self.belt_attached=np.ones(self.M, int)
        
        self.FULL_LOOP_ENT=FULL_LOOP_ENTROPY
        self.loop_prefact=loop_prefactor
        self.SLIDE_PAUSEP=SLIDE_PAUSEPROB

        if type(slidepauseForward) in [int,float, np.float64, np.double]: # just in case it's not initialized
            self.slidepauseForward=np.zeros(self.M, np.double)
            for i in range(self.M):
                self.slidepauseForward[i]=slidepauseForward
        else:
            self.slidepauseForward=slidepauseForward
        if type(slidepauseBackward) in [int,float, np.float64, np.double]: # just in case it's not initialized
            self.slidepauseBackward=np.zeros(self.M, np.double)
            for i in range(self.M):
                self.slidepauseBackward[i]=slidepauseBackward
        else:
            self.slidepauseBackward=slidepauseBackward
        #if type(slidepauseSum) in [int,float, np.float64, np.double]: # just in case it's not initialized
        #    self.slidepauseSum=np.zeros(self.M, np.double)
        #    for i in range(self.M):
        #        self.slidepauseSum[i]=slidepauseSum

        if type(switch) in [int,float,np.float64,np.double]:
            self.switch=np.zeros(self.M, np.double)
            for i in range(self.M):
                self.switch[i] = switch
        else:
            self.switch=switch

        if type(belt_on) in [int,float,np.float64,np.double]:
            self.belt_on_rate=np.zeros(self.M, np.double)
            for i in range(self.M):
                self.belt_on_rate[i] = belt_on
        else:
            self.belt_on_rate=belt_on
        if type(belt_off) in [int,float,np.float64,np.double]:
            self.belt_off_rate=np.zeros(self.M, np.double)
            for i in range(self.M):
                self.belt_off_rate[i] = belt_off
                self.belt_off_rate_original[i] = belt_off
        else:
            self.belt_off_rate=belt_off
            self.belt_off_rate_original = belt_off # 

        self.inactive_side= np.zeros(self.M,int)

        for ind in xrange(self.M):
            self.birth(ind)


    cdef birth(self, cython.int ind):
        cdef int pos,i 
  
        while True:
            pos = self.getss()
            if pos >= self.N - 1:
                print "bad value", pos, self.cumEmission[len(self.cumEmission)-1]
                continue 
            if pos <= 0:
                print "bad value", pos, self.cumEmission[0]
                continue  
            
            if self.occupied[pos] == 1:
                continue
            if self.occupied[pos+1] == 1:
                continue
            
            #want to avoid placing SMCs across chain breaks, so prohibit landing on stall sites
            if self.stallLeft[pos]==1.: # hmm. I think this works in cython...
                continue
            if self.stallRight[pos+1]==1.:#this one is probably not even necessary, due to the above line.
                continue

            self.SMCs1[ind] = pos
            self.SMCs2[ind] = pos+1 #don't let smcs start on single binding site
            if self.paired:
                self.SMCs3[ind]= pos + np.random.randint(0,2) #just put the center site on one of those two, don't allow shrinking 
            self.occupied[pos] = 1
            self.occupied[pos+1] = 1

            #this won't be necessary if I add stochasticity to stepping.
            #if (pos < (self.N - 3)) and (self.occupied[pos+1] == 0):
            #    if randnum() > 0.5:                  
            #        self.SMCs2[ind] = pos + 2
            #        self.occupied[pos+2] = 1
            #        self.occupied[pos+1] = 0

            if self.onesided[ind]:
                if randnum() < 0.5: #1-sided extrusion - stall 1 arm
                    self.stalled1[ind]=1
                    self.inactive_side[ind]=1
                    if self.paired:
                        self.SMCs3[ind]=self.SMCs2[ind]
                else:
                    self.stalled2[ind]=1
                    self.inactive_side[ind]=2
                    if self.paired:
                        self.SMCs3[ind]=self.SMCs1[ind]

                if self.switch[ind]>0:
                    self.set_switch_life(ind)

                self.belt_attached[ind] = 1
                self.belt_off_rate[ind] = self.belt_off_rate_original[ind]

            return


    cdef set_switch_life(self, cython.long ind):
        self.switch_life[ind] = np.long(-np.log(np.random.uniform())/self.switch[ind])
        return

    def set_slidepause(self, slidepauseForward, slidepauseBackward):#cython.double [:] slidepauseForward, cython.double [:] slidepauseBackward):
        self.slidepauseForward=slidepauseForward
        self.slidepauseBackward=slidepauseBackward


    cdef do_switch(self, cython.int ind):
        self.inactive_side[ind] = 1 + self.inactive_side[ind]%2
        # note that this allows repeat attempts against a stall site via directional switching
        if self.inactive_side[ind] == 1:
            self.stalled1[ind] = 1
            self.stalled2[ind] = 0
        else:
            self.stalled1[ind] = 0
            self.stalled2[ind] = 1
        self.set_switch_life(ind)

    cdef death(self):
        cdef int i 
        cdef double falloff1, falloff2 
        cdef double falloff 
         
        for i in xrange(self.M):
            if self.stalled1[i] == 0:
                falloff1 = self.falloff[self.SMCs1[i]]
            else: 
                falloff1 = self.stallFalloff[self.SMCs1[i]]
            if self.stalled2[i] == 0:
                falloff2 = self.falloff[self.SMCs2[i]]
            else:
                falloff2 = self.stallFalloff[self.SMCs2[i]]              
            
            falloff = max(falloff1, falloff2)
            if randnum() < falloff:                 
                self.occupied[self.SMCs1[i]] = 0
                self.occupied[self.SMCs2[i]] = 0
                if self.paired:
                    self.occupied[self.SMCs3[i]] = 0
                self.stalled1[i] = 0
                self.stalled2[i] = 0
                self.birth(i)
    
    cdef int getss(self):
    
        if self.curss >= self.maxss - 1:
            foundArray = np.array(np.searchsorted(self.cumEmission, np.random.random(self.maxss)), dtype = np.long)
            self.ssarray = foundArray
            #print np.array(self.ssarray).min(), np.array(self.ssarray).max()
            self.curss = -1
        
        self.curss += 1         
        return self.ssarray[self.curss]
        
    

    cdef step(self):
        cdef int i 
        cdef double pause
        cdef double stall1, stall2 
        cdef int cur1
        cdef int cur2 
        #cdef int stall_site1, stall_site2
        cdef cython.double rr
        cdef cython.double ratesum
        cdef long pushing_allowed, x
        cdef cython.long [:] y
        
        if self.slide:
            self.entropic_rates() # set the rates taking into account the entropic cost of enlarging loops.
        
        for i in range(self.M):            
            stall1 = self.stallLeft[self.SMCs1[i]]
            stall2 = self.stallRight[self.SMCs2[i]]

            #these variables will be used to determine whether stall is due to a stall size
            #or simply due to being the inactive side of the extruder
            #stall_site1=0
            #stall_site2=0
            
            if randnum() < stall1: 
                self.stalled1[i] = 1
                #stall_site1=1 # used for book keeping before I had safety belt and inactive_side variables
                if self.inactive_side[i]==1:
                    self.belt_off_rate[i] = 0.
                    self.belt_attached[i] = 1
            if randnum() < stall2: 
                self.stalled2[i] = 1
                #stall_site2=1
                if self.inactive_side[i]==2:
                    self.belt_off_rate[i] = 0.
                    self.belt_attached[i] = 1

            #note- smc can already be stalled if it is one-sided
     
            cur1 = self.SMCs1[i]
            cur2 = self.SMCs2[i]
            
            if self.stalled1[i] == 0: # not stalled, just go if possible 
                if self.occupied[cur1-1] == 0:
                    pause1 = self.pause[self.SMCs1[i]]
                    if randnum() > pause1: 
                        self.occupied[cur1-1] = 1
                        self.occupied[cur1] = 0
                        self.SMCs1[i] = cur1 - 1
                        if self.paired:#guarantee that center site remains occupied
                            self.occupied[self.SMCs3[i]] = 1
                elif self.pushing and self.occupied[cur1-1] == 1: # if site occupied, but in a sim where pushing is allowed
                    #first check if pushing is allowed
                    pushing_allowed=1 # start with pushing allowed
                    x = cur1-1 # consider position of LEF of occupied site
                    if x-1<0: # if it's already at 0, it can't move
                        pushing_allowed=0
                    while pushing_allowed and self.occupied[x]: # while we need to check obstacle lefs
                        if x-1<0: # if lef at x can't move, can't push
                            pushing_allowed=0
                            break # not even necessary to break, just insurance.
                        else:
                            y=np.where(np.array(self.SMCs1)==x)[0] # check SMCs1 to see if it has the leg at x
                            if len(y)>0: # if so...
                                #I think self.stallLeft[x]<0.99999 should be changed to not stall_site1, or else, maybe stallLeft<1e-8... not sure why I didn't just use stallSite.
                                #didn't use stall_site, because I didn't want to have to have randnum() < stall.  I want stalled & finite stall prob.
                                if self.stalled1[y[0]] and (not (self.stallLeft[x]>0.0)) and (not self.belt_attached[y[0]]): #check if stalled (inactive) and not due to stallLeft=1 (not sure what cython does with == for floats), this allows pushing over real stalls with stall prob <1, but i'm currently only using stall to identify inactive sites and the end of the chain. 
                                    x=x-1 # move on to check next position
                                else:# smc at x is an active leg ..break
                                    pushing_allowed=0
                                    break
                            else: # otherwise, check SMCs2
                                y=np.where(np.array(self.SMCs2)==x)[0]
                                #len y should necessarily be >0 (since occupied is a condition to be in the while loop)
                                if self.stalled2[y[0]] and (not (self.stallRight[x]>0.0)) and (not (self.stallLeft[x]>0.0)) and not self.belt_attached[y[0]]: # SMCs2 move rightward, so don't really care if stallLeft==1... but do want to make sure I'm not pushing an active leg. and don't want to push past a stall size
                                    x=x-1
                                else:
                                    pushing_allowed=0
                                    break
                    if pushing_allowed:
                        #now check if active site makes a step
                        pause1=self.pause[self.SMCs1[i]] 
                        if randnum() > pause1:
                            #x is now the site just past the last (leftmost) smc in the series of pushed sites
                            self.occupied[x]=1
                            x=x+1
                            while self.occupied[x] == 1 and not x==cur1: # go through chain, moving SMCs, note that self.occupied does not change except ends
                                y=np.where(np.array(self.SMCs1)==x)[0]
                                if len(y)>0:
                                    self.SMCs1[y[0]]=x-1
                                else:
                                    y=np.where(np.array(self.SMCs2)==x)[0]
                                    self.SMCs2[y[0]]=x-1
                                x=x+1
                            self.occupied[cur1]=0
                            self.SMCs1[i]=cur1-1
                            

                        #need to move all those LEFs now...
                #need to repeat for SMCs2 moving rightward with cur+1, x+1 and x out of bounds when x>=L
 
            elif self.slide and (not self.belt_attached[i]) and (self.inactive_side[i]==1):# and (stall_site1==0): #if stall is not due to a stall site, then it is b/c we are looking at inactive side of LEF
                #HERE IS WHERE I MAKE CHANGES AND PUT IN slidepauseforward and slidepausebackward
                #inactive side 1, slide pause forward goes with cur-1, i.e., first if
                #inactive side 2, slide pause forward goes with cur+1 i.e., second if
                #so if-else-if scheme might need to be changed
                if (self.occupied[cur1-1]==0) and (self.occupied[cur1+1]==0):
                    rr=randnum()
                    if rr < 2.-self.slidepauseForward[i]-self.slidepauseBackward[i]:
                        #probability of sliding one way or the other
                        if rr < 1.-self.slidepauseForward[i]:
                            #slide forward
                            self.occupied[cur1-1]=1
                            self.SMCs1[i]=cur1-1
                        else:
                            #slide backward
                            self.occupied[cur1+1]=1
                            self.SMCs1[i]=cur1+1
                        self.occupied[cur1]=0
                        slidestep_taken=1
                elif (self.occupied[cur1-1]==0) and (self.occupied[cur1+1]==1):
                    if randnum() > self.slidepauseForward[i]:
                        #slide forward
                        self.occupied[cur1-1]=1
                        self.SMCs1[i]=cur1-1
                        self.occupied[cur1]=0
                        slidestep_taken=1
                elif (self.occupied[cur1-1]==1) and (self.occupied[cur1+1]==0):
                    if randnum() > self.slidepauseBackward[i]:
                        #slide backward
                        self.occupied[cur1+1]=1
                        self.SMCs1[i]=cur1+1
                        self.occupied[cur1]=0
                        slidestep_taken=1

            if self.stalled2[i] == 0:
                if self.occupied[cur2 + 1] == 0:
                    pause2 = self.pause[self.SMCs2[i]]
                    if randnum() > pause2: 
                        self.occupied[cur2 + 1] = 1
                        self.occupied[cur2] = 0
                        self.SMCs2[i] = cur2 + 1
                        if self.paired:#guarantee that center site remains occupied
                            self.occupied[self.SMCs3[i]] = 1
                elif self.pushing and self.occupied[cur2+1] == 1: # if site occupied, but in a push sim
                    pushing_allowed=1 # start with pushing allowed
                    x = cur2+1 # consider position of LEF of occupied site
                    if x+1>=self.N: # if it's already at N-1, it can't move
                        pushing_allowed=0
                    while pushing_allowed and self.occupied[x]: # while we need to check obstacle lefs
                        if x+1>=self.N: # if lef at x can't move, can't push
                            pushing_allowed=0
                            break # not even necessary to break, just insurance.
                        else:
                            y=np.where(np.array(self.SMCs1)==x)[0] # check SMCs1 to see if it has the leg at x
                            if len(y)>0: # if so...
                                if self.stalled1[y[0]] and (not (self.stallRight[x]>0.0)) and (not (self.stallLeft[x]>0.0)) and not self.belt_attached[y[0]]: #check if stalled (inactive) and not due to stall site, verify not active
                                    x=x+1 # move on to check next position
                                else:# smc at x is an active leg ..break
                                    pushing_allowed=0
                                    break
                            else: # otherwise, check SMCs2
                                y=np.where(np.array(self.SMCs2)==x)[0]
                                #len y should necessarily be >0 (since occupied is a condition to be in the while loop)
                                if self.stalled2[y[0]] and (not (self.stallRight[x]>0.0)) and not self.belt_attached[y[0]]: # SMCs2 move rightward, so don't really care if stallLeft==1... but do want to make sure I'm not pushing an active leg. and don't want to push past a stall size
                                    x=x+1
                                else:
                                    pushing_allowed=0
                                    break
                    if pushing_allowed: 
                        #now check if active site makes a step
                        pause2=self.pause[self.SMCs2[i]] 
                        if randnum() > pause2:            
                            #x is now the site of the last smc in the chain 
                            self.occupied[x]=1
                            x=x-1
                            while self.occupied[x] == 1 and not x==cur2: # go through chain, moving SMCs, note that self.occupied does not change except ends                                                                             
                                y=np.where(np.array(self.SMCs1)==x)[0]                                                         
                                if len(y)>0:              
                                    self.SMCs1[y[0]]=x+1     
                                else:                     
                                    y=np.where(np.array(self.SMCs2)==x)[0]                                                     
                                    self.SMCs2[y[0]]=x+1
                                x=x-1
                            self.occupied[cur2]=0         
                            self.SMCs2[i]=cur2+1 

            elif self.slide and (not self.belt_attached[i]) and (self.inactive_side[i]==2):# and (stall_site2==0):
                if (self.occupied[cur2-1]==0) and (self.occupied[cur2+1]==0):
                    rr=randnum()
                    #print(rr)
                    #ratesum = self.slidepauseForward[i]
                    #ratesum += self.slidepauseBackward[i]
                    if rr < 2.-self.slidepauseForward[i]-self.slidepauseBackward[i]: #ratesum:
                        #probability of sliding one way or the other
                        if rr < 1.-self.slidepauseForward[i]:
                            #slide forward
                            self.occupied[cur2+1]=1
                            self.SMCs2[i]=cur2+1
                        else:
                            #slide backward
                            self.occupied[cur2-1]=1
                            self.SMCs2[i]=cur2-1
                        self.occupied[cur2]=0
                        slidestep_taken=1
                elif (self.occupied[cur2-1]==1) and (self.occupied[cur2+1]==0):
                    if randnum() > self.slidepauseForward[i]:
                        #slide forward
                        self.occupied[cur2+1]=1
                        self.SMCs2[i]=cur2+1
                        self.occupied[cur2]=0
                        slidestep_taken=1
                elif (self.occupied[cur2-1]==0) and (self.occupied[cur2+1]==1):
                    if randnum() > self.slidepauseBackward[i]:
                        #slide backward
                        self.occupied[cur2-1]=1
                        self.SMCs2[i]=cur2-1
                        self.occupied[cur2]=0
                        slidestep_taken=1

            if self.switch[i]>0:
                if self.switch_life[i] == 0:
                    self.do_switch(i)
                else:
                    self.switch_life[i] -= 1

            if self.belt_attached[i]:
                if randnum() < self.belt_off_rate[i]:
                    self.belt_attached[i]=0
            else:
                if randnum() < self.belt_on_rate[i]:
                    self.belt_attached[i]=1

 

        
    def steps(self,N):
        cdef int i, slidestep_taken, remaining_steps
        remaining_steps=0
        for i in xrange(N):
            self.death()
            self.step()
            
    def getOccupied(self):
        return np.array(self.occupied)
    
    def getSMCs(self):
        if not self.paired:
            return np.array(self.SMCs1), np.array(self.SMCs2)
        else:
            return np.array(self.SMCs1), np.array(self.SMCs2), np.array(self.SMCs3)
        
    
    ##
    #these next two functions seem to be for updating matrices that keep track of:
    #smc-mediated contacts between two sites and
    #whether or not a particular site at a particular time (index) has an SMC.
    #these aren't used in the milker-based code that uses this class
    ##

    def updateMap(self, cmap):
        cmap[self.SMCs1, self.SMCs2] += 1
        cmap[self.SMCs2, self.SMCs1] += 1

    def updatePos(self, pos, ind):
        pos[ind, self.SMCs1] = 1
        pos[ind, self.SMCs2] = 1


    def entropic_rates(self):
        left, right = self.getSMCs() #SMCTran.getSMCs()
        lefsites= [[left[k],k] for k in range(len(left))]
        lefsites.extend([[right[k],k] for k in range(len(right))])
        # allBonds.append(bonds)

        #if nn>4:
        #    print("left=")
        #    print(left)
        #    print("right=")
        #    print(right)

        #bonds.sort()
        lefsites.sort()

        #code to calculate how to adjust sliding rates
        #i=bonds[0]#min(left) # 0 skip some empty sites.
        #lef_list=[]
        loop_len=np.zeros(self.M)
        parent_loop = np.zeros(self.M,dtype=np.int64)-1 # i^th component is the index of the smc that contains smc i. or -1 if none.
        loop_entropy_forward=np.zeros(self.M,dtype=np.float64)
        loop_entropy_backward=np.zeros(self.M,dtype=np.float64)

        k=0
        i=lefsites[k][0]
        while k < len(lefsites):
            lef_list=[]
            #need to find which LEF occupies site 
            lef_id=lefsites[k][1]#np.where((left==i))[0]#|(right==i) # actually, this should only catch left sides.
            lef_list.append(lef_id)
            j=k+1

            #NEED TO CHECK K+1<LEN(ARRAY)
            if j < len(lefsites)-1:
                while not (right[lef_id]==lefsites[j][0]):
                    lef_at_j=lefsites[j][1]#np.where((left==j)|(right==j))[0]

                    #print('lefatj and list type',type(lef_at_j), print(type(lef_list)), "lefatj", lef_at_j)
                    if lef_at_j in lef_list:
                        sum_loop_lens=0
                        for l in lef_list[lef_list.index(lef_at_j):]:
                            sum_loop_lens += loop_len[l]
                        loop_len[lef_at_j]+= right[lef_at_j]-left[lef_at_j] - sum_loop_lens

                        #also need to add contribution of containing loop.
                    else:
                        
                        lef_list.append(lef_at_j)
                    j+=1 # no more checking j<len since it should always be enclosed by lef_id

            if not right[lef_id] == lefsites[j][0]:
                print("Right side of LEF inconsistency! STOP!")
                #exit()

            sum_loop_lens=0
            if not lef_list.index(lef_id)==0:
                print("Last LEF index wrong? STOP!")
                #exit()
            for l in lef_list[1:]:
                sum_loop_lens += loop_len[l]
            loop_len[lef_id]+= right[lef_id] - left[lef_id] - sum_loop_lens

            k= j
            del lef_list

            k+=1

        #not really the entropy - instead, the derivative of entropy

        loop_entropy_forward[loop_len>0]= -1.*self.loop_prefact / loop_len[loop_len>0] 
        if self.FULL_LOOP_ENT:
            parents = looptools.get_parent_loops(left, right)
            loop_entropy_forward[parents>=0] = loop_entropy_forward[parents>=0] + self.loop_prefact / loop_len[parents[parents>=0]]
        loop_entropy_backward=-1.*loop_entropy_forward
        
        #adjust rates based on loop_entropy
        slidePauseArrayForward=np.zeros(self.M) + 1.-(1.-self.SLIDE_PAUSEP)*np.exp(loop_entropy_forward)
        slidePauseArrayBackward=np.zeros(self.M) + 1.-(1.-self.SLIDE_PAUSEP)*np.exp(loop_entropy_backward)

        #feed in new rates to smctranslocator
        #slidePauseSum=slidePauseArrayForward+slidePauseArrayBackward
        #SMCTran
        self.set_slidepause(slidePauseArrayForward, slidePauseArrayBackward)#, slidePauseSum)
        