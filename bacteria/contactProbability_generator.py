# Written by Hugo Brandao
# Harvard University, 2018

# ChainLayout is used to generate theoretical contact probability maps for a given set 
# of input SMC positions assuming Gaussian chain statistics

# Input:
#      N: length of the chain 
#      SMCs: a list of tuples representing the SMC positions e.g. [(start1, end1), (start2,end2),...]

# Usage:
# ChainLayout.get_dist(pos1,pos2) returns an effective distance, d_eff, between any two monomers 
# at pos1, pos2 on the chain. Contact probability may be computed as 1/sqrt(d_eff)**x where x = 1,2,3,... 
# is the dimension of the real space

import numpy as np
import matplotlib.pyplot as plt
import looplib as lp
from looplib import looptools as lt

class ChainLayout:
    def __init__(self,SMCs,N):
        self.SMCs = sorted(SMCs)
        self.N = N
        # loops
        lsites, rsites = lt.convert_loops_to_sites(self.SMCs)
        self.lsites = lsites
        self.rsites = rsites
        self.dists = rsites-lsites
        self.parents = lt.get_parent_loops(lsites,rsites)
        self.children = lt.get_loop_branches(self.parents,lsites)
        self.isroot = lt.get_roots(lsites,rsites)
        self.depth = self.get_depth()
        self.loop_idx_dict = None
        
        # main chain
        self.main_chain, self.main_chain_L = self.make_main_chain()
        self.cum_MCL = np.cumsum(self.main_chain_L)
        
    def make_main_chain(self):
        # from a given set of SMCs, create a structure for the main chain (backbone)
        main_chain = []
        main_chain_L = []
        rsites = self.rsites
        lsites = self.lsites
        isroot = self.isroot
        N = self.N
        for l,r in zip(np.r_[0,rsites[isroot]],np.r_[lsites[isroot],N]):
            main_chain.append((l,r))
            main_chain_L.append(r-l)
        return main_chain, main_chain_L
    
    
    def get_main_chain_idx(self,pos): 
        # returns the index of the main chain segment containing position pos
        mainchain = self.main_chain
        for x in range(len(mainchain)):
            if (pos >= mainchain[x][0] ) & (pos <= mainchain[x][1] ):
                return x
        return -1 # out of bounds..
    
    def main_chain_dist(self,pos1,pos2):
        # for a given pair of monomer positions pos1, pos2, this function returns the distance
        # along the main chain between the monomers excluding the loops between them       
        pos1, pos2 = sorted([pos1,pos2])
        mc1_idx = self.get_main_chain_idx(pos1) 
        mc2_idx = self.get_main_chain_idx(pos2)
        # assert mc1_idx >=0
        # assert mc2_idx >=0
        
        # if both positions on the same chain segment
        if mc1_idx == mc2_idx:
            return np.abs(pos2-pos1)
        else:
            L_right = pos2-self.main_chain[mc2_idx][0]
            L_left =  self.main_chain[mc1_idx][1]-pos1
            L_between = self.cum_MCL[mc2_idx-1]-self.cum_MCL[mc1_idx]
            # assert( L_right >=0)
            # assert( L_left >=0 )
            return L_right + L_left + L_between

        
    def get_depth(self):   
        # for each SMC position, this function returns the depth/nestedness of the loop
        # parents is the list of parent indices returned from get_parent_loops()
        parents = self.parents
        def get_depth_recursive(idx,parents,count):
            if parents[idx] != -1:
                count = get_depth_recursive(parents[idx],parents,count+1)
            return count

        depth = np.zeros(len(parents),dtype=int) 
        for p in range(len(parents)):
            depth[p] = get_depth_recursive(p,parents,0)
        return depth   
    
    def get_loop_idx(self,pos):
        # for a given monomer position (pos) of a chain of monomers, find the smallest loop in 
        # which pos resides; the loop size is determined by the total linear monomeric distance 
        # between SMC left-right positions

        if self.loop_idx_dict is None: 
            # pre-compute loop_idx
            self.loop_idx_dict = {}
            lsites = self.lsites
            rsites = self.rsites
            dists = self.dists            
            
            for p in range(self.N):
                idx = np.where((lsites<=p) & (rsites>=p))[0]
                if len(idx)>0:
                    idx2 = np.argmin(dists[idx])
                    self.loop_idx_dict[p] = (idx[idx2], idx)
                else:
                    self.loop_idx_dict[p] = (-2 , idx) 

        return self.loop_idx_dict[pos]


    def get_circle_distance(self,pos,lp_idx=None,dist=0,desired_depth=0):
        # this function returns the effective distance of a position pos to 
        # the root of a series of nested loops
        SMCs = self.SMCs
        
        if lp_idx is None:
            lp_idx, _ = self.get_loop_idx(pos)

        children_idx = self.children[lp_idx]
        parent_idx = self.parents[lp_idx]
        new_pos = SMCs[lp_idx][1] 
        
        # if there are no children
        if len(children_idx) == 0:
            N_circ =  SMCs[lp_idx][1]-SMCs[lp_idx][0] # np.diff(SMCs[lp_idx])[0] #
            s = pos - SMCs[lp_idx][0]

            dist = dist + s*(1-s/N_circ)

        else: 
            # split children in to "left" and "right" of pos, compute loop size
            left_children = 0
            right_children = 0
            
            s_left = pos - SMCs[lp_idx][0]
            s_right = SMCs[lp_idx][1] - pos
            
            #  assert(s_left>=0)
            #  assert(s_right>=0)

            for c in [SMCs[x] for x in children_idx]:
                if c[1]<=pos:
                    left_children +=   c[1]-c[0] #
                else:
                    right_children +=  c[1]-c[0] #
                    
            # assert(left_children>=0)
            # assert(right_children>=0)
            
            s_left = s_left-left_children
            s_right = s_right-right_children
            
            dist = s_left*(1-s_left/(s_left+s_right))

            
        if (self.depth[lp_idx] != desired_depth) and (parent_idx != -1):
            d_temp, new_pos = self.get_circle_distance(new_pos,parent_idx,dist,desired_depth)
            dist += d_temp
            
        return dist, new_pos
    
    
    def get_circle_distance2(self,pos1,pos2,lp_idx=None):
        # gets the circular distance between two points if they are on the same loop
        # returns  -3 if the two positions are on different loops
        if pos1 > pos2:
            pos1, pos2 = (pos2,pos1)

        SMCs = self.SMCs
        
        # get index of the loop for calculation
        # make sure that both positions are on the same loop
        if lp_idx is None:
            lp_idx, lp_idx_all = self.get_loop_idx(pos1)
            lp_idx2, lp2_idx_all = self.get_loop_idx(pos2)

            # if the loops are not identified on the same branch check
            # if the larger of the two loops contains the smaller loop as a child
            # check also if the two positions are different direct children of a larger, parent loop
            if lp_idx != lp_idx2:
                sameBranch = False
                
                # identify overlap depth
                overlap = list(set(lp_idx_all).intersection(lp2_idx_all))#np.intersect1d(lp_idx_all,lp2_idx_all)

                # if there is no overlap, then the loops are on different branches
                if len(overlap) == 0:
                    return -3, -3
                
                meeting_idx = overlap[np.argmin(self.dists[overlap])]
                meeting_depth = self.depth[meeting_idx] 
                
                # if the loop at which they meet too distant, can't use this function
                if (self.depth[lp_idx]- meeting_depth)>=2 or (self.depth[lp_idx2]- meeting_depth)>=2 :
                    return -3, -3
                
                # if the loop at which they meet equals one of the two depths:
                elif (self.depth[lp_idx]== meeting_depth)  or (self.depth[lp_idx2] == meeting_depth):

                    # get larger loop
                    loop1_len = self.dists[lp_idx]
                    loop2_len = self.dists[lp_idx2]
            
                    if loop1_len > loop2_len:
                        pos = pos2
                        children_idx = self.children[lp_idx]
                    else:

                        pos = pos1
                        children_idx = self.children[lp_idx2]
                        lp_idx = lp_idx2

                    # test if larger loop contains the smaller loop as of its children
                    for c in [SMCs[x] for x in children_idx]:
                        if (c[0] == pos) or (c[1]==pos):
                            sameBranch = True
                            continue
                    if sameBranch == False:
                        return -3, -3
                
                # if the parent loop contains both smaller loops
                elif ((self.depth[lp_idx] - meeting_depth)==1)  and ((self.depth[lp_idx2] - meeting_depth)==1):

                    # test if position of  "smaller" loops positions are at roots of the loop
                    if ((SMCs[lp_idx][0] == pos1) or (SMCs[lp_idx][1] == pos1)) and \
                        ((SMCs[lp_idx2][0] == pos2) or (SMCs[lp_idx2][1] == pos2)):
                            lp_idx = meeting_idx
                    else:
                        return -3,-3
                
                # something strange happened
                else:
                    assert 1==0
                    
        children_idx = self.children[lp_idx]
        parent_idx = self.parents[lp_idx]
        new_pos = SMCs[lp_idx][1] 

        # if there are no children
        if len(children_idx) == 0:
            N_circ =  SMCs[lp_idx][1] - SMCs[lp_idx][0] # np.diff(SMCs[lp_idx])[0] #
            s = pos2-pos1 

            dist = s*(1-s/N_circ)

        else: 
            # split children in to "left", "middle and "right" of pos, compute loop size
            left_children = 0
            middle_children = 0
            right_children = 0

            s_left = pos1 - SMCs[lp_idx][0]
            s_right = SMCs[lp_idx][1] - pos2
            s_tot =  SMCs[lp_idx][1]- SMCs[lp_idx][0]
            
            #print(s_left,s_right,s_tot)
            #assert(s_left>=0)
            #assert(s_right>=0)

            for c in [SMCs[x] for x in children_idx]:
                if c[1]<=pos1:
                    left_children += c[1]-c[0]#
                elif (c[0]>= pos1) and (c[1]<=pos2):
                    middle_children +=  c[1]-c[0]#
                else:
                    right_children += c[1]-c[0]#

            # assert(left_children>=0)
            # assert(right_children>=0)

            s_left = s_left-left_children
            s_right = s_right-right_children
            s_tot = s_tot-left_children-right_children-middle_children
            
            # assert(s_left>=0)
            # assert(s_right>=0)
            # assert(s_tot>=s_right+s_left)

            s = s_left + s_right

            dist = s*(1-s/s_tot)

        return dist, new_pos    
    
    
    def get_dist(self,pos1,pos2,dist=0):
        if pos1 > pos2:
            pos1, pos2 = (pos2,pos1)
        
        # are pos1, pos2 in loops or main chain?
        branch1_idx, branch1_all = self.get_loop_idx(pos1)
        branch2_idx, branch2_all = self.get_loop_idx(pos2)
    
         # if both positions are on the main chain
        if branch1_idx == branch2_idx == -2:
            dist += self.main_chain_dist(pos1,pos2)

            
        # if left position is on the main chain
        elif branch1_idx == -2:
            # compute distance of branch2 to root
            dist_circ, root_pos = self.get_circle_distance(pos2)           
            # compute distance of branch1 point to root
            dist_lin = self.main_chain_dist(pos1,root_pos)
            # return sum
            dist += dist_circ + dist_lin

            
        # if right position is on the main chain
        elif branch2_idx == -2:
            # compute distance of branch2 to root
            dist_circ, root_pos = self.get_circle_distance(pos1)           
            # compute distance of branch1 point to root
            dist_lin = self.main_chain_dist(pos2,root_pos)
            # return sum
            dist += dist_circ + dist_lin
            
        # if both positions are on a loop
        else: 
            # if loops are nested, find the branch in which they meet
            overlap = list(set(branch1_all).intersection(branch2_all)) # find the intersection of the two lists
            if len(overlap)>0:
                meeting_idx = overlap[np.argmin(self.dists[overlap])]
                meeting_depth = self.depth[meeting_idx]                    
                
                # if both loops are on different branches
                # this means that the overlap indices do not contain the branch indices
                if (branch1_idx not in overlap) and (branch2_idx not in overlap):

                    d1, np1 = self.get_circle_distance(pos1,lp_idx=branch1_idx,\
                                                       desired_depth=meeting_depth+1)
                    d2, np2 = self.get_circle_distance(pos2,lp_idx=branch2_idx,\
                                                       desired_depth=meeting_depth+1) 
                    d3, np3 = self.get_circle_distance2(np1,np2)

                    dist = d1+d2+d3
                
                elif branch1_idx == branch2_idx:
                    dist, _ = self.get_circle_distance2(pos1,pos2)
                    
                # if one loops is nested within the other
                else:

                    # find which is the bigger loop 
                    
                    # get larger loop
                    loop1_len = self.dists[branch1_idx]
                    loop2_len = self.dists[branch2_idx]

                    if loop1_len > loop2_len:
                        pos_small = pos2
                        pos_big = pos1
                    else:
                        pos_small = pos1
                        pos_big = pos2

                    #vals = np.asarray([[loop1_len,branch1_idx,pos1],[loop2_len,branch2_idx,pos2]])
                    #vals = vals[np.argsort(vals[:,0])]
                    #pos_small = vals[0,2] # smaller loop
                    #pos_big  = vals[1,2]
                    
                    d1, np1 = self.get_circle_distance(pos_small,desired_depth=meeting_depth+1)
                    d2, np2 = self.get_circle_distance2(pos_big,np1)
                    
                    dist = d1+d2
            
            else:
                # if there is no overlap, calculate the loop distances to main chain
                d1, np1 = self.get_circle_distance(pos1,lp_idx=branch1_idx)
                d2, np2 = self.get_circle_distance(pos2,lp_idx=branch2_idx)
                d3 = self.main_chain_dist(np1,np2)
                
                dist = d1+d2+d3 
                #dist = self.get_dist(np1,np2,dist=(d1+d2)) 

        return dist
