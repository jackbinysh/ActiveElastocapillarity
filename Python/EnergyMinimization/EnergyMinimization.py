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

def ReadMesh(filename):
    InputMesh = meshio.read(filename)
    
    # make list of:
    #interior bonds : interiorbonds
    # edge bonds :edgepoints
    # bonds : interiorbonds+edgebonds
    # angle triples: angletriples
    triangles=mesh.cells()
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
        
    return InputMesh, OutputMesh, interiorbonds,edgebonds,angletriples



def MakeDolfinMesh(a, edgepoints):
    # make the mesh. Lets have a unit circle. It seems, from trial and error, that
    # res = 1.5*Radius/mesh_size,
    domain = Ellipse(Point(0, 0, 0),1.0,1.0, edgepoints)
    mesh = generate_mesh(domain, 1.5/a)
    
    mesh.init()
    
    # need to add a 3rd dimension 0 coordinate here
    points = np.insert(mesh.coordinates(),2,0,axis=1)
    cells = cells = [("triangle",mesh.cells() )]
    InputMesh = meshio.Mesh(points,cells)
    #copy for modifying at output
    OutputMesh=copy.deepcopy(InputMesh)
    
    # make list of:
    #interior bonds : interiorbonds
    # edge bonds :edgepoints
    # bonds : interiorbonds+edgebonds
    # angle triples: angletriples
    triangles=mesh.cells()
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
        
    return InputMesh, OutputMesh, interiorbonds,edgebonds,angletriples

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
