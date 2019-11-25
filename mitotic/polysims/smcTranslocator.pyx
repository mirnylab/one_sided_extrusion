#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=True

import numpy as np
cimport numpy as np 
import cython
cimport cython 


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
    cdef cython.double [:] slidepause # pause probability for diffusing arms, so we can regulate v_diffuse / v_active ratio. note that this does not deal with entropy due to loops
    cdef cython.double [:] switch
    cdef cython.long [:] switch_life

    cdef int maxss
    cdef int curss
    cdef cython.long [:] ssarray  
    cdef int transparent 
    
    def __init__(self, emissionProb, deathProb, stallProbLeft, stallProbRight, pauseProb, stallFalloffProb,  numSmc, onesided, paired=0, slide=0, slidepause=0, switch=0, transparent=0):
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
        self.transparent=transparent

        cdef int i
        if type(slidepause) in [int,float]: # just in case it's not initialized
            self.slidepause=np.zeros(self.M, np.double)
            for i in range(self.M):
                self.slidepause[i]=slidepause
        else:
            self.slidepause=slidepause

        self.switch_life=np.zeros(self.M,long) # time remaining until directional switch
        if type(switch) in [int,float,np.float64,np.double]: # just in case it's not initialized
            self.switch=np.zeros(self.M, np.double)
            for i in range(self.M):
                self.switch[i]=switch
        else:
            self.switch=switch
        
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
            
            if not self.transparent == 1:
                if self.occupied[pos] == 1:
                    continue
                if self.occupied[pos+1] == 1:
                    continue
            
            #want to avoid placing SMCs across chain breaks, so prohibit landing on stall sites
            if self.stallLeft[pos]==1.:
                continue
            if self.stallRight[pos+1]==1.:#this one is probably not even necessary, due to the above line.
                continue

            self.SMCs1[ind] = pos
            self.SMCs2[ind] = pos+1 #don't let smcs start on single binding site
            if self.paired:
                self.SMCs3[ind]= pos + np.random.randint(0,2) #just put the center site on one of those two, don't allow shrinking 
            self.occupied[pos] = 1
            self.occupied[pos+1] = 1

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
 
            return


    cdef set_switch_life(self, cython.int ind):
        self.switch_life[ind] = int(-np.log(np.random.uniform())/self.switch[ind])
        return

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
                self.occupied[self.SMCs1[i]] = 0 # for "transparent" LEF sims, occupied variable doesn't work. For functionality though, this doesn't matter.
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
            self.curss = -1
        
        self.curss += 1         
        return self.ssarray[self.curss]
        
        

    cdef step(self):
        cdef int i 
        cdef double pause
        cdef double stall1, stall2 
        cdef int cur1
        cdef int cur2 
        cdef int stall_site1, stall_site2

        for i in range(self.M):            
            stall1 = self.stallLeft[self.SMCs1[i]]
            stall2 = self.stallRight[self.SMCs2[i]]

            #these variables will be used to determine whether stall is due to a stall size
            #or simply due to being the inactive side of the extruder
            stall_site1=0
            stall_site2=0
            
            if randnum() < stall1: 
                self.stalled1[i] = 1
                stall_site1=1
            if randnum() < stall2: 
                self.stalled2[i] = 1
                stall_site2=1
                         
            cur1 = self.SMCs1[i]
            cur2 = self.SMCs2[i]
            
            if self.stalled1[i] == 0: 
                if (self.occupied[cur1-1] == 0) or (self.transparent==1):
                    pause1 = self.pause[self.SMCs1[i]]
                    if randnum() > pause1: 
                        self.occupied[cur1 - 1] = 1
                        self.occupied[cur1] = 0 # need to fix for transparent
                        self.SMCs1[i] = cur1 - 1
                        if self.paired:#guarantee that center site remains occupied
                            self.occupied[self.SMCs3[i]] = 1
            elif self.slide and (self.inactive_side[i]==1) and (stall_site1==0): #if stall is not due to a stall site, then it is b/c we are looking at inactive side of LEF
                if randnum() > self.slidepause[i]:
                    if randnum() < 0.5:
                        if (self.occupied[cur1-1]==0) or (self.transparent==1):
                            self.occupied[cur1-1]=1
                            self.occupied[cur1]=0 #not used for transparent
                            self.SMCs1[i]=cur1-1
                    else:
                        if (self.occupied[cur1+1]==0) or (self.transparent==1):
                            self.occupied[cur1+1]=1
                            self.occupied[cur1]=0
                            self.SMCs1[i]=cur1+1

            if self.stalled2[i] == 0:                
                if (self.occupied[cur2 + 1] == 0) or (self.transparent==1):                    
                    pause2 = self.pause[self.SMCs2[i]]
                    if randnum() > pause2: 
                        self.occupied[cur2 + 1] = 1
                        self.occupied[cur2] = 0 #not used for transparent
                        self.SMCs2[i] = cur2 + 1
                        if self.paired:#guarantee that center site remains occupied
                            self.occupied[self.SMCs3[i]] = 1
            elif self.slide and (self.inactive_side[i]==2) and (stall_site2==0):
                if randnum() > self.slidepause[i]:
                    if randnum()<0.5:
                        if (self.occupied[cur2-1]==0) or (self.transparent==1):
                            self.occupied[cur2-1]=1
                            self.occupied[cur2]=0 #transparent needs doesn't use this
                            self.SMCs2[i]=cur2-1
                    else:
                        if (self.occupied[cur2+1]==0) or (self.transparent==1):
                            self.occupied[cur2+1]=1
                            self.occupied[cur2]=0 #not used for transparent
                            self.SMCs2[i]=cur2+1

            if self.switch[i]>0:
                if self.switch_life[i] == 0:
                    self.do_switch(i)
                else:
                    self.switch_life[i] -= 1

        
    def steps(self,N):
        cdef int i, slidestep_taken, remaining_steps
        remaining_steps=0
        for i in xrange(N):
            self.death()
            self.step()
            
    def getOccupied(self): #does not work for transparent... don't use with 3d LEF interactions.
        return np.array(self.occupied)
    
    def getSMCs(self):
        if np.any(self.onesided):
            return np.array(self.SMCs1), np.array(self.SMCs2), np.array(self.inactive_side)
        elif not self.paired:
            return np.array(self.SMCs1), np.array(self.SMCs2)
        else:
            return np.array(self.SMCs1), np.array(self.SMCs2), np.array(self.SMCs3)
        
    
    ##
    #these next two function are for updating matrices that keep track of:
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



