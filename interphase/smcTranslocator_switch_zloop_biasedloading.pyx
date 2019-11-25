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


cdef class smcTranslocatorDirectional(object): #Compare to smcTransloc
    cdef int N
    cdef int M
    cdef cython.double [:] emission
    cdef cython.double [:] stall
    cdef cython.double [:] stallFalloff
    cdef cython.double [:] falloff
    cdef cython.double [:] pause
    cdef cython.double [:] cumEmission
    cdef cython.long [:] SMCs1
    cdef cython.long [:] SMCs2
    cdef cython.long [:] stalled1 
    cdef cython.long [:] stalled2
    cdef cython.long [:] stallarray
    cdef cython.long [:] occupied 
    
    cdef double SwitchRate
    cdef int zloops
    cdef cython.long [:] MovingArm
    
    cdef int numCTCF
    cdef cython.long [:] CTCFoccupied
    cdef double CTCFprob
    cdef double CTCFbinding
    cdef double CTCFunbinding
    cdef cython.long [:] CTCFcoords
    cdef double CTCFCohesinEnergy
    
    cdef cython.long [:] loadleft
    cdef cython.long [:] loadright
    cdef int dirLoading
    
    cdef int maxss
    cdef int curss
    cdef cython.long [:] ssarray 
 
    
    def __init__(self, emissionProb, deathProb, stallProb, pauseProb, stallFalloffProb,numSmc, CTCFcoord,CTCFon,CTCFoff,zloop=1,CTCFCohesin=0,switch=0,dirLoading=0,lleft=np.array([ 399,  499,  699, 1099, 1199, 1399, 1799, 1899, 2099, 2499, 2599,
       2799, 3199, 3299, 3499, 3899, 3999, 4199, 4599, 4699, 4899, 5299,
       5399, 5599]),lright=np.array([ 401,  501,  701, 1101, 1201, 1401, 1801, 1901, 2101, 2501, 2601,
       2801, 3201, 3301, 3501, 3901, 4001, 4201, 4601, 4701, 4901, 5301,
       5401])):
        
        
        emissionProb[0] = 0
        emissionProb[len(emissionProb)-1] = 0
        emissionProb[stallProb > 0.9] = 0        
        
        self.N = len(emissionProb)
        self.M = numSmc
        self.emission = emissionProb
        self.stall = stallProb #Permanently occupied CTCF sites
        self.falloff = deathProb
        self.pause = pauseProb
        cumem = np.cumsum(emissionProb)
        cumem = cumem / float(cumem[len(cumem)-1])
        self.cumEmission = np.array(cumem, np.double)
        self.SMCs1 = np.zeros((self.M), int)
        self.SMCs2 = np.zeros((self.M), int)
        self.stalled1 = np.zeros(self.M, int)
        self.stalled2 = np.zeros(self.M, int)
        self.stallarray = np.zeros(self.N,int)
        self.occupied = np.zeros(self.N, int)
        self.stallFalloff = stallFalloffProb
        self.occupied[0] = 1
        self.occupied[self.N - 1] = 1
        self.maxss = 1000000
        self.curss = 99999999
        
        self.CTCFcoords=CTCFcoord
        self.numCTCF = len(self.CTCFcoords) # CTCFcoord is an array with the coordinates of CTCF binding sites
        self.SwitchRate = switch
        self.MovingArm = np.zeros(self.M,int)
        self.zloops=zloop
        
        self.CTCFbinding = CTCFon     # CTCF binding rate
        self.CTCFunbinding = CTCFoff  # CTCF unbinding rate
        self.CTCFprob = CTCFon/(CTCFon+CTCFoff) # Probability for CTCF to be bound, !!!check if computation is correct
        self.CTCFoccupied = np.zeros(self.N,int) # Occupation of CTCF binding sites
        self.CTCFCohesinEnergy = CTCFCohesin # Binding energy of CTCF and cohesin in units of kT
        self.loadleft=lleft
        self.loadright=lright
        self.dirLoading=dirLoading
        #print "CTCF Cohesin EN", self.CTCFCohesinEnergy
        
        for i in self.CTCFcoords: # Let CTCF sites equilibrate before starting the simulation
            if randnum()<self.CTCFprob:
                self.CTCFoccupied[i]=1

        for ind in xrange(self.M):
            self.birth(ind)


    cdef birth(self, cython.int ind):
        cdef int pos,i,n,l,r
        
        while True:
            pos = self.getss()#Get position for binding SMC
            if pos >= self.N - 2:
                #print "bad value", pos, self.cumEmission[len(self.cumEmission)-1]
                continue #Pick another position, this one falls of the lattice
            if pos <= 0:
                print "bad value", pos, self.cumEmission[0]
                continue #Pick another position, this one falls of the lattice
 
            if self.zloops == 0:
                if self.occupied[pos] == 1:
                    continue #Pick another position, this one is already occupied

                if self.occupied[pos+1] == 1:
                    continue
            
            self.SMCs1[ind] = pos
            self.SMCs2[ind] = pos+1
            self.occupied[pos] = 1
            self.occupied[pos+1] = 1
            
            #if (pos < (self.N - 3)) and (self.occupied[pos+2] == 0):
            #    if randnum() > 0.5:                  
            #        self.SMCs2[ind] = pos + 2
            #        self.occupied[pos+2] = 1
            #       self.occupied[pos+1] = 0
            n=0
            l=0
            r=0
            #print(str(pos))
            if self.dirLoading==1:
                while n<1:
                    for i in self.loadleft:# randnum()>0.5: # One sided extrusion, stall one of the two arms
                        if i==pos:
                            self.MovingArm[ind] = 0 # Left arm is moving
                            #print('loadleft')
                            l=1
                            n=2
                    for i in self.loadright:
                        if i==pos:
                            self.MovingArm[ind] = 1 # Right arm is moving
                            #print('loadright')
                            r=1
                            n=2
                    if (l<1)&(r<1):
                        if (randnum()>0.5):
                            #print('rndl')
                            self.MovingArm[ind] = 0
                            n=2
                        else: 
                            #print('rndr')
                            self.MovingArm[ind] = 1
                            n=2
                           
            else:
                if (randnum()>0.5):
                    #print('rndl')
                    self.MovingArm[ind] = 0
                else: 
                    #print('rndr')
                    self.MovingArm[ind] = 1
                    
            return

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
                self.stallarray[self.SMCs1[i]] = 0
                self.stallarray[self.SMCs2[i]] = 0
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
        
        # Each cohesin can switch and the two arms make steps in a time step
        # A cohesin stuck on one CTCF site can switch and continue to the other site.
        
        cdef int i 
        cdef double pause
        cdef double stall1, stall2 
        cdef int cur1
        cdef int cur2
        
        
        for i in range(self.M):  
            
            if randnum()<self.SwitchRate:
                #self.stalled1[i] = 0
                #self.stalled2[i] = 0
                if self.MovingArm[i]==0:
                    self.MovingArm[i]=1
                else: 
                    self.MovingArm[i]=0         
            
            stall1 = self.stall[self.SMCs1[i]] 
            stall2 = self.stall[self.SMCs2[i]]
                                    
            if randnum() < stall1: 
                self.stalled1[i] = 1
                self.stallarray[self.SMCs1[i]] =1 
            if randnum() < stall2: 
                self.stalled2[i] = 1
                self.stallarray[self.SMCs2[i]] =1 

                         
            cur1 = self.SMCs1[i] # Treat occupied CTCF site as inpenetrable barrier, add the possibility that residence time CTCF increases when bound by cohesin
            cur2 = self.SMCs2[i]                      
            
            if self.MovingArm[i] == 0: # Left arm is active
                if self.stalled1[i] == 0:
                        if ((self.occupied[cur1-1]==0) or  (self.zloops==1)) and (cur1>1):
                            if self.CTCFoccupied[cur1-1]==0: 
                                pause1 = self.pause[self.SMCs1[i]]
                                if randnum() < pause1:#If not stalled, paused or CTCF bound, make a step to the left
                                    self.occupied[cur1 - 1] = 1
                                    self.occupied[cur1] = 0
                                    self.SMCs1[i] = cur1 - 1
                                              
            if self.MovingArm[i] == 1:       # Right arm is active         
                if self.stalled2[i] == 0:
                        if ((self.occupied[cur2+1]==0) or (self.zloops==1)) and (cur2<self.N-2):
                            if self.CTCFoccupied[cur2+1]==0:
                                pause2 = self.pause[self.SMCs2[i]]
                                if randnum() < pause2: #If not stalled, paused or CTCF bound, make a step to the right
                                    self.occupied[cur2 + 1] = 1
                                    self.occupied[cur2] = 0
                                    self.SMCs2[i] = cur2 + 1
                  
                
    cdef CTCFkinetics(self):
        cdef int i
        cdef double CTCFbound
        cdef double on,off
        
        for i in self.CTCFcoords:
            CTCFbound = self.CTCFoccupied[i] 
            if CTCFbound == 1:
                if ((self.occupied[i-1]==1)|(self.occupied[i+1]==1)):
                    unbinding=self.CTCFunbinding*np.exp(-self.CTCFCohesinEnergy)#If cohesin and CTCF interact, CTCF unbinding rate is reduced.
                    if ((randnum() < unbinding)): 
                        self.CTCFoccupied[i]=0
                else:
                    if ((randnum() < self.CTCFunbinding)): 
                        self.CTCFoccupied[i]=0
            if ((CTCFbound == 0)):#Bind (even if occupied by cohesin as each monomer represents many basepairs)
                if randnum() < self.CTCFbinding:
                    self.CTCFoccupied[i]=1
      
    
    def steps(self,N): #Right now, cohesin and CTCF are always updated in the same order, should actually used random sequential update...
        cdef int i 
        for i in xrange(N):
            self.death()
            self.step()
            self.CTCFkinetics()
            
    def getOccupied(self):
        return np.array(self.occupied)
    
    def getSMCs(self):
        return np.array(self.SMCs1), np.array(self.SMCs2)
        
        
    def updateMap(self, cmap):
        cmap[self.SMCs1, self.SMCs2] += 1
        cmap[self.SMCs2, self.SMCs1] += 1

    def updatePos(self, pos, ind):
        pos[ind, self.SMCs1] = 1
        pos[ind, self.SMCs2] = 1