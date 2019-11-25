# Written by Hugo Brandao (hbrandao@g.harvard.edu)
# (c) copyright Harvard University, 2017

import os
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, ConvexHull
import numpy as np
from itertools import product
from scipy.spatial import cKDTree 
import mpl_toolkits.mplot3d as a3
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D


""" Algorithm to create an alpha shape triangulation of a set of points

Delaunay triangulate my points:

Store a set of edges, list of edge points

Loop over Delaunay triangles
For all vertices of each triangle, get spatial coordinate + compute:
1) Length of sides
2) Perimiter of triangle/2
3) Area of triangle
4) Circumradius of the triangle

Apply radius length cutoff: if R < 1/alpha
If criterion met, then add the edge, store the volume (Area)

To calculate if it's a boundary point, get the two circle (sphere) centers of radius 1/alpha for each edge
From circle centers, calculate the nearest neighbour distance. If it exceeds 1/alpha, then this is a boundary edge
If not, then store it as a boundary
"""

#####################################################################################
##                            Codes in 2D                                          ## 
#####################################################################################

# Draws the simplices provided a Delaunay triangulation in 2D
def drawSimplices(simplices,colour='k'):
    for s in simplices:
        for (p1,p2) in [x for x in product(s, s) if x[0]!=x[1] and x[0]> x[1]]:
            X = [points[p1][0], points[p2][0]]
            Y = [points[p1][1], points[p2][1]]
            plt.plot(X,Y,colour)   

# Given a radius r, get the location of the circles that pass through 
# The two points p1 and p2 (in 2D)
def getCircleCenter(p1,p2,r=1):
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    pc = (p1+p2)/2 # midpoint of p1, p2
    q = np.sqrt(np.sum((p2-pc)**2)) # half-distance
    A = r**2-(q)**2 # determinant
    if A>0:
        d1 = (p2-p1)*[1,-1]
        d1 = d1[::-1]
        d1 = d1/np.sqrt(np.sum(d1**2)) 
        d2 = -d1
        c1 = pc + np.sqrt(A)*d1 # first solution
        c2 = pc + np.sqrt(A)*d2 # second solution
        return [c1,c2]
    else:
        return []   

# Given 3 points p1, p2, p3, find the radius of the circle that passes through them    
def circumCircle_radius(p1,p2,p3):
    # Lengths of sides of triangle given by points:
    # p1 = (x1,y1), p2=(x2,y2), p3=(x3,y3)
    a = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    b = np.sqrt((p2[0]-p3[0])**2 + (p2[1]-p3[1])**2)
    c = np.sqrt((p3[0]-p1[0])**2 + (p3[1]-p1[1])**2)
    # 1/2 the perimiter of triangle
    s = (a + b + c)/2.0
    # area of triangle (using Heron's formula)
    area = np.sqrt(s*(s-a)*(s-b)*(s-c))
    circum_radius = a*b*c/(4.0*area)
    return circum_radius 
## http://mathworld.wolfram.com/Circumsphere.html 

# Add an edge to the edges set, and append provided coordinates to
# edge_coordinates if points i, j are not already in edges set 
def addEdge(edges,edge_coords,coords,i,j):
    if (i,j) in edges or (j,i) in edges:
        return False
    edges.add((i,j))
    edge_coords.append(coords)
    return True

# Determines whether an edge of the belongs to the boundary is an interior point
def getBoundaryCode(tree,edge_points,r,tol = 0.00001):
    # boundary code returns:
    # 1 if the p1,p2 are boundary edges of the shape
    # 0 if in the interior 
    # 2 if the circle does not exist
    p1 = edge_points[0]
    p2 = edge_points[1]
    circles = getCircleCenter(p1,p2,r)
    
    if len(circles) != 2:
        return 2
    
    bq = tree.query(circles[0])
    bq2 = tree.query(circles[1])
    if (bq[0] > r-tol) or (bq2[0] > r-tol):
        return 1
    else:
        return 0

# Compues the alpha shape in 2D
def alphaShape2D(points, alpha=1):
    shp = Delaunay(points)
    tree = cKDTree(points)
    edges = set()
    edge_coords = []
    boundary_code = []
    area = 0
    perimiter = 0
    for si,s in enumerate(shp.simplices):
        pa,pb,pc =[points[x] for x in s]
        R = circumCircle_radius(pa,pb,pc)
        if R<1/alpha:
            area += getArea3pt(pa,pb,pc)
            e1 = addEdge(edges,edge_coords,(pa,pb), s[0],s[1])
            e2 = addEdge(edges,edge_coords,(pb,pc), s[1],s[2])
            e3 = addEdge(edges,edge_coords,(pc,pa), s[2],s[0])
            
            if e1 == True:
                bc = getBoundaryCode(tree,(pa,pb),1/alpha)
                boundary_code.append(bc)
                if bc == 1:
                    perimiter += v_length(pa,pb)
            if e2 == True:
                bc = getBoundaryCode(tree,(pb,pc),1/alpha)
                boundary_code.append(bc)
                if bc == 1:
                    perimiter += v_length(pb,pc)
                
            if e3 == True:
                bc = getBoundaryCode(tree,(pc,pa),1/alpha)
                boundary_code.append(bc)           
                if bc == 1:
                    perimiter += v_length(pa,pc)
            
    return edges, edge_coords, boundary_code, area, perimiter

def v_length(p1,p2):
    return np.sqrt(np.sum((np.asarray(p1)-np.asarray(p2) )**2))

#####################################################################################
##                            Codes in 3D                                          ## 
#####################################################################################

def mydet2(A):
    return A[0][0]*A[1][1]-A[0][1]*A[1][0]

def mydet3(A):
    return A[0][0]* (A[1][1]*A[2][2]-A[1][2]*A[2][1]) - \
           A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0]) + \
           A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0])

#def mydet4(A):
#    A = np.asarray(A)
#    return A[0][0]*mydet3(A[1:,1:]) \
#         - A[0][1]*mydet3(A[1:,[0,2,3]]) \
#         + A[0][2]*mydet3(A[1:,[0,1,3]]) \
#         - A[0][3]*mydet3(A[1:,[0,1,2]])
    
def mydet4(A):
    d1 = A[1][1]* (A[2][2]*A[3][3]-A[2][3]*A[3][2]) - \
        A[1][2]*(A[2][1]*A[3][3]-A[2][3]*A[3][1]) + \
        A[1][3]*(A[2][1]*A[3][2]-A[2][2]*A[3][1])  # all values get +1
    d2 = A[1][0]* (A[2][2]*A[3][3]-A[2][3]*A[3][2]) - \
        A[1][2]*(A[2][0]*A[3][3]-A[2][3]*A[3][0]) + \
        A[1][3]*(A[2][0]*A[3][2]-A[2][2]*A[3][0])# all y values +1, all x >0 get +1
    d3 = A[1][0]* (A[2][1]*A[3][3]-A[2][3]*A[3][1]) - \
        A[1][1]*(A[2][0]*A[3][3]-A[2][3]*A[3][0]) + \
        A[1][3]*(A[2][0]*A[3][1]-A[2][1]*A[3][0]) # all y values +1,all x >1 get +1
    d4 = A[1][0]* (A[2][1]*A[3][2]-A[2][2]*A[3][1]) - \
        A[1][1]*(A[2][0]*A[3][2]-A[2][2]*A[3][0]) + \
        A[1][2]*(A[2][0]*A[3][1]-A[2][1]*A[3][0]) # all y values +1, all x>2 get +1
    return A[0][0]*(d1) \
         - A[0][1]*(d2) \
         + A[0][2]*(d3) \
         - A[0][3]*(d4)    
        

# Get the radius of the sphere which passes through the 4 points p1, p2, p3, p4
def circumSphere_radius(p1,p2,p3,p4):
    p1 = np.asarray(p1)
    p1sq = p1[0]*p1[0] + p1[1]*p1[1] + p1[2]*p1[2]
    p2 = np.asarray(p2)
    p2sq = p2[0]*p2[0] + p2[1]*p2[1] + p2[2]*p2[2]
    p3 = np.asarray(p3)
    p3sq = p3[0]*p3[0] + p3[1]*p3[1] + p3[2]*p3[2]
    p4 = np.asarray(p4)
    p4sq = p4[0]*p4[0] + p4[1]*p4[1] + p4[2]*p4[2]
    A = [[p1[0],p1[1],p1[2],1] ,\
         [p2[0],p2[1],p2[2],1] ,\
         [p3[0],p3[1],p3[2],1] ,\
         [p4[0],p4[1],p4[2],1] ]
    Dx = [[p1sq,p1[1],p1[2],1] ,\
         [p2sq,p2[1],p2[2],1] ,\
         [p3sq,p3[1],p3[2],1] ,\
         [p4sq,p4[1],p4[2],1] ]
    Dy = [[p1sq,p1[0],p1[2],1] ,\
         [p2sq,p2[0],p2[2],1] ,\
         [p3sq,p3[0],p3[2],1] ,\
         [p4sq,p4[0],p4[2],1] ]
    Dz = [[p1sq,p1[0],p1[1],1] ,\
         [p2sq,p2[0],p2[1],1] ,\
         [p3sq,p3[0],p3[1],1] ,\
         [p4sq,p4[0],p4[1],1] ]
    Dc = [[p1sq,p1[0],p1[1],p1[2]] ,\
         [p2sq,p2[0],p2[1],p2[2]] ,\
         [p3sq,p3[0],p3[1],p3[2]] ,\
         [p4sq,p4[0],p4[1],p4[2]] ]
    
    a = mydet4(A)
    Dx = mydet4(Dx)
    Dy = -mydet4(Dy)
    Dz = mydet4(Dz)
    c = mydet4(Dc)     
    delta = Dx*Dx+Dy*Dy+Dz*Dz- 4*a*c
    if delta< 0:
        return np.nan   
    circumSphere_r = np.sqrt(delta)/2/np.abs(a)
    return circumSphere_r

""" Slow version
def circumSphere_radius(p1,p2,p3,p4):
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    p3 = np.asarray(p3)
    p4 = np.asarray(p4)
    A = [[p1[0],p1[1],p1[2],1] ,\
         [p2[0],p2[1],p2[2],1] ,\
         [p3[0],p3[1],p3[2],1] ,\
         [p4[0],p4[1],p4[2],1] ]
    Dx = [[np.sum(p1**2),p1[1],p1[2],1] ,\
         [np.sum(p2**2),p2[1],p2[2],1] ,\
         [np.sum(p3**2),p3[1],p3[2],1] ,\
         [np.sum(p4**2),p4[1],p4[2],1] ]
    Dy = [[np.sum(p1**2),p1[0],p1[2],1] ,\
         [np.sum(p2**2),p2[0],p2[2],1] ,\
         [np.sum(p3**2),p3[0],p3[2],1] ,\
         [np.sum(p4**2),p4[0],p4[2],1] ]
    Dz = [[np.sum(p1**2),p1[0],p1[1],1] ,\
         [np.sum(p2**2),p2[0],p2[1],1] ,\
         [np.sum(p3**2),p3[0],p3[1],1] ,\
         [np.sum(p4**2),p4[0],p4[1],1] ]
    Dc = [[np.sum(p1**2),p1[0],p1[1],p1[2]] ,\
         [np.sum(p2**2),p2[0],p2[1],p2[2]] ,\
         [np.sum(p3**2),p3[0],p3[1],p3[2]] ,\
         [np.sum(p4**2),p4[0],p4[1],p4[2]] ]
    a = np.linalg.det(A)
    Dx = np.linalg.det(Dx)
    Dy = -np.linalg.det(Dy)
    Dz = np.linalg.det(Dz)
    c = np.linalg.det(Dc)
    delta = Dx**2+Dy**2+Dz**2 - 4*a*c
    if delta< 0:
        return np.nan   
    circumSphere_r = np.sqrt(delta)/2/np.abs(a)
    return circumSphere_r
"""
def mycross(b,c):
    return np.asarray([b[1]*c[2]-b[2]*c[1] , b[2]*c[0]-b[0]*c[2], b[0]*c[1]-b[1]*c[0]])

# Find a coordinate which is equidistant from all three points p1, p2, p3
def getCircumCenter3D(p1,p2,p3):
#    p1 = np.asarray(p1)
#    p2 = np.asarray(p2)
#    p3 = np.asarray(p3)
    ac = p3-p1
    ab = p2-p1
    #abXac = mycross(ac,ab)
    abXac = [ac[1]*ab[2]-ac[2]*ab[1] , ac[2]*ab[0]-ac[0]*ab[2], ac[0]*ab[1]-ac[1]*ab[0]]
    lac = ac[0]*ac[0] + ac[1]*ac[1] + ac[2]*ac[2] 
    lab = ab[0]*ab[0] + ab[1]*ab[1] + ab[2]*ab[2]
    labXac = abXac[0]*abXac[0] + abXac[1]*abXac[1] + abXac[2]*abXac[2]   #(np.sum(abXac**2))
    
    if labXac == 0:
        print("Warning: labXac = 0")
    #center = p1 + (np.cross(ab,abXac)*lac + np.cross(abXac,ac)*lab)/(2*labXac)
    # center = p1 + (mycross(ab,abXac)*lac + mycross(abXac,ac)*lab)/(2*labXac)
    
    v1 = lac/(2*labXac)
    v2 = lab/(2*labXac)

#    return p1 + (np.asarray([ab[1]*abXac[2]-ab[2]*abXac[1] , ab[2]*abXac[0]-ab[0]*abXac[2], ab[0]*abXac[1]-ab[1]*abXac[0]])*lac \
#                 + np.asarray([abXac[1]*ac[2]-abXac[2]*ac[1] , abXac[2]*ac[0]-abXac[0]*ac[2], abXac[0]*ac[1]-abXac[1]*ac[0]])*lab) \
#                  /(2*labXac) #center
    
#    return p1 + (np.asarray([ab[1]*abXac[2]-ab[2]*abXac[1] , ab[2]*abXac[0]-ab[0]*abXac[2], ab[0]*abXac[1]-ab[1]*abXac[0]])*lac \
#                 + np.asarray([abXac[1]*ac[2]-abXac[2]*ac[1] , abXac[2]*ac[0]-abXac[0]*ac[2], abXac[0]*ac[1]-abXac[1]*ac[0]])*lab) \
#                  /(2*labXac) #center

    return p1 + np.asarray([v1*(ab[1]*abXac[2]-ab[2]*abXac[1])+v2*(abXac[1]*ac[2]-abXac[2]*ac[1]),\
                            v1*( ab[2]*abXac[0]-ab[0]*abXac[2])+v2*(abXac[2]*ac[0]-abXac[0]*ac[2]),\
                            v1*(ab[0]*abXac[1]-ab[1]*abXac[0])+v2*(abXac[0]*ac[1]-abXac[1]*ac[0])])

    
# Given a radius r, get the coordinates of the centers of the two spheres that 
# pass through points p1, p2, p3
def getSphereCenter(p1,p2,p3,r=1,tol=1e-10):
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    p3 = np.asarray(p3)
    r = r+tol
    pc = getCircumCenter3D(p1,p2,p3)# circumcenter of triangle formed by p1, p2, p3
    #q = np.sqrt(np.sum((p2-pc)**2)) # distance of centroid to one corner
    p2pc = p2-pc
    qsq = p2pc[0]*p2pc[0] + p2pc[1]*p2pc[1] + p2pc[2]*p2pc[2]  # distance sq of centroid to one corner
    A = r*r-qsq # determinant
    if A>0:
        #d1 = np.cross(p1-p3,p2-p3)
        d1 = mycross(p1-p3,p2-p3)
        d1msq = d1[0]*d1[0] + d1[1]*d1[1] + d1[2]*d1[2]     #np.sum(d1**2)      
        if d1msq ==0:
            print("getSphereCenter division by zero")        
        d1 = np.sqrt(A/d1msq)*d1
        return [pc + d1,pc - d1]
    else:
        return []
    
# adds the edge to set, and appends coordinates
def addEdge3(edges,edge_coords,coords,edge_indices):
    edge_indices = tuple(sorted(edge_indices))
    if edge_indices in edges:
        return False
    edges.add(edge_indices)
    edge_coords.append(coords)
    return True

# Test whether the edges in question are at the boundary of the alpha shape
def getBoundaryCode3(tree,edge_points,r,tol = 0.001):
    # boundary code returns:
    # 1 if the p1,p2 are boundary edges of the shape
    # 0 if in the interior 
    # 2 if the circle does not exist
    p1 = edge_points[0]
    p2 = edge_points[1]
    p3 = edge_points[2]
    spheres = getSphereCenter(p1,p2,p3,r)
    
    if len(spheres) != 2:
        return 2
    
    bq = tree.query(spheres[0])
    if (bq[0] > r-tol):
        return 1    
    bq2 = tree.query(spheres[1])
    if (bq2[0] > r-tol):
        return 1
    return 0

def getArea3pt(p1,p2,p3):
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    p3 = np.asarray(p3)
    ab = p2-p1
    bc = p2-p3
    v = mycross(ab,bc)
    return np.sqrt(v[0]*v[0] +v[1]*v[1]+v[2]*v[2])/2

def getVolume4pt(p1,p2,p3,p4):
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    p3 = np.asarray(p3)
    p4 = np.asarray(p4)
    ab = p2-p1
    bc = p2-p3
    cd = p4-p3
    A = np.asarray([ab,bc,cd])
    #return np.abs(np.linalg.det(A)/6)
    return np.abs(mydet3(A)/6)



# Compute the alphaShape for a set of points in 3D    
def alphaShape3D(points, alpha=1):
    shp = Delaunay(points)
    tree = cKDTree(points)
    edges = set()
    edge_coords = []
    boundary_code = []
    volume = 0
    sa = 0
    for si,s in enumerate(shp.simplices):
        pa,pb,pc,pd =[points[x] for x in s]
        # for each triangulation tetrahedron, get the circumsphere
        R = circumSphere_radius(pa,pb,pc,pd)
        # check if the circumpshere is within the acceptable range
        if R<1/alpha:
            volume += getVolume4pt(pa,pb,pc,pd)
            e1 = addEdge3(edges,edge_coords,(pa,pb,pc), (s[0],s[1],s[2]) )
            e2 = addEdge3(edges,edge_coords,(pb,pc,pd), (s[1],s[2],s[3]) )
            e3 = addEdge3(edges,edge_coords,(pc,pd,pa), (s[2],s[3],s[0]) )
            e4 = addEdge3(edges,edge_coords,(pd,pa,pb), (s[3],s[0],s[1]) )
            if e1 == True:
                bc = getBoundaryCode3(tree,(pa,pb,pc),1/alpha)
                boundary_code.append(bc)
                if bc == 1:
                    sa += getArea3pt(pa,pb,pc)
            if e2 == True:
                bc = getBoundaryCode3(tree,(pb,pc,pd),1/alpha)
                boundary_code.append(bc)
                if bc == 1:
                    sa += getArea3pt(pb,pc,pd)                
                
            if e3 == True:
                bc = getBoundaryCode3(tree,(pc,pd,pa),1/alpha)
                boundary_code.append(bc) 
                if bc == 1:
                    sa += getArea3pt(pc,pd,pa)                
            if e4 == True:
                bc = getBoundaryCode3(tree,(pd,pa,pb),1/alpha)
                boundary_code.append(bc)  
                if bc == 1:
                    sa += getArea3pt(pd,pa,pb)                  
    return edges, edge_coords, boundary_code, volume, sa


def createSphere(r,center, N=10):
    lst = []
    thetas = [(2*np.pi*i)/N for i in range(N)]
    phis = [(np.pi*i)/N for i in range(N)]
    for theta in thetas:
        for phi in phis:
            x = r * np.sin(phi) * np.cos(theta) + center[0]
            y = r * np.sin(phi) * np.sin(theta) + center[1]
            z = r * np.cos(phi) + center[2]
            lst.append((x, y, z))
    return np.asarray(lst)


def drawWireFrame3D(edge_coords,boundaryCode,drawBoundary=True,drawInner=False,boundaryStyle='k-',innerStyle='m-'):
    fig= plt.figure()
    ax = fig.gca(projection='3d')
    for c in range(len(edge_coords)):
        for perm in [[0,1],[0,2],[1,2]]:
            X = []; Y = []; Z = []
            for p1,p2,p3 in [edge_coords[c][x] for x in perm]:
                X.append(p1); Y.append(p2); Z.append(p3)

            if boundaryCode[c] == 1 and drawBoundary:
                ax.plot(X,Y,Z,boundaryStyle)
            if boundaryCode[c] == 0 and drawInner:
                ax.plot(X,Y,Z,innerStyle)               
    plt.show()
    

def drawFaces3D(edge_coords,boundaryCode,alpha=0.5,colour='skyblue'):
    fig= plt.figure()
    ax = fig.gca(projection='3d')
    patches = []
    X, Y, Z = [], [], []
    tri_list = []
    for c in range(len(edge_coords)):
        if boundaryCode[c] == 1:
            tri = a3.art3d.Poly3DCollection([np.asarray(edge_coords[c])])
            tri.set_color('skyblue')
            tri.set_edgecolor('k')
            ax.add_collection3d(tri)


