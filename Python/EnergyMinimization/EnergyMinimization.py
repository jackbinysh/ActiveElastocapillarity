import meshio
import pygalmesh
import numpy as np
import copy
from mshr import *
from dolfin import *
from collections import Counter
import matplotlib.pyplot as plt
import os
import json
import shutil
import scipy.optimize as opt

####### 2D STUFF #########

def MakeBondAngleLists(mesh):
    # make list of:
    #interior bonds : interiorbonds
    # edge bonds :edgepoints
    # bonds : interiorbonds+edgebonds
    # angle triples: angletriples
    triangles=mesh.cells[0].data
    x = [[[triangle[0],triangle[1]],[triangle[0],triangle[2]],[triangle[1],triangle[2]] ]   for triangle in triangles]
    flattenedx = [val for sublist in x for val in sublist]
    bonds = [[x[0],x[1]] if x[0]<x[1] else [x[1],x[0]] for x in flattenedx]

    # get a list of the bonds on the edge, and in the interior
    edgebonds=[]
    interiorbonds=[]
    for elem in bonds:
        if 1==bonds.count(elem):
            edgebonds.append(elem)
        elif 2==bonds.count(elem) and elem not in interiorbonds:
            interiorbonds.append(elem)

    bonds=interiorbonds+edgebonds

    # for the edge bonds, get the angle triples
    EdgeVertices = list(set([val for sublist in edgebonds for val in sublist]))
    angletriples=[]

    for vertex in EdgeVertices:
        Neighbors=[x for x in edgebonds if vertex in x]
        NeighborVertices = [val for sublist in Neighbors for val in sublist if val!=vertex]
        angletriples.append([NeighborVertices[0],vertex,NeighborVertices[1]])

    return interiorbonds,edgebonds,angletriples

def MakeDolfinMesh(a, edgepoints):
    # make the mesh. Lets have a unit circle. It seems, from trial and error, that
    # res = 1.5*Radius/mesh_size,
    domain = Ellipse(Point(0, 0, 0),1.0,1.0, edgepoints)
    mesh = generate_mesh(domain, 1.5/a)
    mesh.init()

    # need to add a 3rd dimension 0 coordinate here
    points = np.insert(mesh.coordinates(),2,0,axis=1)
    cells = cells = [("triangle",mesh.cells() )]
    MeshioMesh = meshio.Mesh(points,cells)

    return MeshioMesh 


def MakeBondHist(Coordinates,bondlist):
    lengths=[np.linalg.norm(Coordinates[bond[1]] -Coordinates[bond[0]]) for bond in bondlist]
    plt.hist(lengths)
    return lengths

# take the positions, return vector of bending energy
def vBending(P_ij,angletriples,kd,theta0):
    
    BendingEnergies=np.zeros(len(angletriples))
    for i, angletriple in enumerate(angletriples):  
        r10 = P_ij[angletriple[0],:]-P_ij[angletriple[1],:]
        r12 = P_ij[angletriple[2],:]-P_ij[angletriple[1],:]
        theta = np.arccos(  np.dot(r10,r12)/( (np.linalg.norm(r10))*(np.linalg.norm(r12)) )  )
        BendingEnergies[i] = (1/2)*kd*(theta-theta0)**2
    return BendingEnergies

def dist(P):
    return np.sqrt((P[:,0]-P[:,0][:,np.newaxis])**2 +
                   (P[:,1]-P[:,1][:,np.newaxis])**2)

def vNeoHookean(r_ij,r0_ij,khook):
    kneo_ij = (r0_ij**2)*khook/3  
    # the diagonal is irrelevant, just fill it with 1's
    np.fill_diagonal(r_ij,1)
    np.fill_diagonal(r0_ij,1)
    lam_ij=r_ij/r0_ij
    V_ij=(kneo_ij/2)*((2/lam_ij) + lam_ij**2)
    return V_ij
 
def TotalArea(P_ij,triangles):
    TotalArea=0
    for triangle in triangles:
        v1 = P_ij[triangle[1]]-P_ij[triangle[0]]
        v2 = P_ij[triangle[2]]-P_ij[triangle[0]]
        TriArea= 0.5*np.linalg.norm( (np.cross(v1,v2)))
        TotalArea = TotalArea+TriArea
    return TotalArea   

def vTotalArea(pts,tri):
    AB=pts[tri[:,0:2]]
    t1 = np.subtract(AB[:,0,:],AB[:,1,:])
    BC=pts[tri[:,1:3]]
    t2 = np.subtract(BC[:,0,:],BC[:,1,:])
    return np.absolute(0.5*np.cross(t1,t2)).sum()   
    
# The argument P is a vector (flattened matrix), needed for scipy
# A : connectivity matrix
# r0_ij: Bond rest lengths
# angletriples:
# k : Hookean spring constant
# kd : the discrete bond bending energy
# theta0: preferred bond angle

def energy(P,A,r0_ij,angletriples,triangles,k,kd,theta0,B,TargetArea): 
    # We convert it to a matrix here.
    P_ij = P.reshape((-1, 2))
    # We compute the distance matrix.
    r_ij = dist(P_ij)
    # NeoHookean Spring bond energies
    # 0.5 to account for double counting
    SpringEnergy = (0.5*A*vNeoHookean(r_ij,r0_ij,k)).sum()   
    #bond bending energy
    BendingEnergy = vBending(P_ij,angletriples,kd,theta0).sum()
    # Energetic penalty on volume change
    VolumeConstraintEnergy = B*(vTotalArea(P_ij,triangles)-TargetArea)**2
    
    return SpringEnergy+BendingEnergy+VolumeConstraintEnergy    

############ 3D STUFF ####################


def MakeDolfinMesh3D(a, edgepoints):
    
    # Make the mesh, a unit sphere: 
    domain = Sphere(Point(0, 0, 0),1.0,10)
    mesh = generate_mesh(domain,1.5/a)
    
    # make the cube
    #mesh = UnitCubeMesh(1,1,1)
    
    points = mesh.coordinates()
    cells = [("tetra",mesh.cells() )]
    MeshioMesh = meshio.Mesh(points,cells)
    return MeshioMesh 

# Note: this code assumes all triangle and bond specifications are in ascending vertex order, i.e
# bond = [0,3], but never [3,0]
# triangle  =[1,4,6], never [4, 1,6] or [6,4,1] etc.
def MakeBondAngleLists3D(InputMesh):
    
    tetras=InputMesh.cells[0].data
    
    trilist=[]
    for tetra in tetras:
        for (i,v) in enumerate(tetra):
            # make it a python list for ease
            tetra = list(tetra)
            # the triangle made from removing the ith element of the tetra list
            tri = (tetra[:i]+tetra[i+1:])
            # add to the list of all our triangles
            trilist.append(tri)
          
    # we now have a list of all the triangles in the mesh. the duplicates are in the interior, the unique ones
    # form the boundary
    boundarytris=[]
    for tri in trilist:
        if 1==trilist.count(tri):
             boundarytris.append(tri)
                
                
    # Now lets make bond lists. First, all the bonds
    bondlist=[]
    for t in tetras:
        for (i,v1) in enumerate(t):
            for(j,v2) in enumerate(t):
                if(j>i and [v1,v2] not in bondlist):
                    bondlist.append([v1,v2])

    # Now just the bonds on the edge                
    edgebondlist=[]
    for t in boundarytris:
        for (i,v1) in enumerate(t):
            for(j,v2) in enumerate(t):
                if(j>i and [v1,v2] not in edgebondlist):
                    edgebondlist.append([v1,v2])

    # and by a diff, the interior bonds
    interiorbondlist=[]
    for bond in bondlist:
        if(bond not in edgebondlist):
            interiorbondlist.append(bond)

    # these should all be numpy lists going forward
    return np.array(interiorbondlist), np.array(edgebondlist), np.array(boundarytris)


#r_ij: numpy list of bond lengths
# r0_ij: list of rest lengths
#khook: the spring constant
#returns V_ij, a list of the bond energies
def NeoHookean3D(r_ij,r0_ij,khook):
    kneo_ij = (r0_ij**2)*khook/3  
    lam_ij=r_ij/r0_ij
    V_ij=(kneo_ij/2)*((2/lam_ij) + lam_ij**2)
    return V_ij


def energy3D(P,bondlist,r0_ij,khook): 
    # We convert it to a matrix here.
    P_ij = P.reshape((-1, 3))
    # from the bond list, work out what the current bond lengths are:
    AB=P_ij[bondlist]
    t1 = np.subtract(AB[:,0,:],AB[:,1,:])
    rij=np.linalg.norm(t1,axis=1)
    # NeoHookean Spring bond energies
    SpringEnergy = vNeoHookean(r_ij,r0_ij,k).sum()   
    #bond bending energy
    #BendingEnergy = vBending(P_ij,angletriples,kd,theta0).sum()
    # Energetic penalty on volume change
    #VolumeConstraintEnergy = B*(vTotalArea(P_ij,triangles)-TargetArea)**2
    return SpringEnergy
