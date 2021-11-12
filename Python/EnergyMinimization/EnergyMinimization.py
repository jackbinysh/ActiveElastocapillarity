import meshio
#import pygalmesh
import numpy as np
import copy
#from mshr import *
#from dolfin import *
from collections import Counter
#import matplotlib.pyplot as plt
import os
import sys
import json
import shutil
import scipy.optimize as opt
from numba import jit

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
    domain = Sphere(Point(0, 0, 0),1.0,edgepoints)
    mesh = generate_mesh(domain,1.5/a)
    
    # make the cube
    #mesh = UnitCubeMesh(1,1,1)
    
    points = mesh.coordinates()
    cells = [("tetra",mesh.cells() )]
    MeshioMesh = meshio.Mesh(points,cells)
    return MeshioMesh 

def MakeMeshData3D(InputMesh):
    
    for block in InputMesh.cells:
        if 'tetra'==block.type:
            tetras=block.data.copy()
    
    # Im going to sort the tetrahedra so their vertices appear always in ascending order
    tetras=np.sort(tetras,axis=1)
    
    trilist=[]
    for tetra in tetras:
        for (i,v) in enumerate(tetra):
            # make it a python list for ease
            tetra = list(tetra)
            # the triangle made from removing the ith element of the tetra list
            tri = (tetra[:i]+tetra[i+1:])
            # add to the list of all our triangles
            trilist.append(tri)
    trilist=np.array(trilist) 
     
 
    # we now have a list of all the triangles in the mesh. the duplicates are in the interior, the unique ones
    # form the boundary     
    unique_trilist,idx,inv, count = np.unique(trilist, axis=0,return_index=True,return_inverse=True,return_counts=True)
    boundarytris=unique_trilist[1==count]
                                       
    # Now lets make bond lists. First, all the bonds                  
    bonds=np.vstack((tetras[:,[0,1]],
           tetras[:,[0,2]],
           tetras[:,[0,3]],
           tetras[:,[1,2]],
           tetras[:,[1,3]],
           tetras[:,[2,3]]
          ))
    bondlist= np.unique(bonds, axis=0)                                  
                  
    # Now just the bonds on the edge                                   
    edgebonds=np.vstack((boundarytris[:,[0,1]],
                     boundarytris[:,[0,2]],
                     boundarytris[:,[1,2]],
          ))  
    edgebondlist,idx,inv= np.unique(edgebonds, axis=0,return_index=True,return_inverse=True)  
    
    # inv maps from the long list of edges to the unique list. We invert this map to get back to the long list. Now, 
    # we know that list goes (tri0,tri1,tri2,tri3... tri_N, tri0,tri1,...  tri_N, tri0,tri1,... ) so we mod out to 
    # get which triangle we came from
    Nt=len(boundarytris)
    x=np.empty((len(edgebondlist),2),dtype='uint64')
    for i in np.arange(0,len(edgebondlist)):
        x[i,:]=np.where(inv==i)[0]
    bidxTotidx=(x%Nt)
                    

    # By a diff, get the interior bonds. See:
    # https://stackoverflow.com/questions/11903083/find-the-set-difference-between-two-large-arrays-matrices-in-python/11903368#11903368
    a1=bondlist
    a2=edgebondlist
    a1_rows = a1.view([('', a1.dtype)] * a1.shape[1])
    a2_rows = a2.view([('', a2.dtype)] * a2.shape[1])
    interiorbondlist=np.setdiff1d(a1_rows, a2_rows).view(a1.dtype).reshape(-1, a1.shape[1])
                   
    return interiorbondlist, edgebondlist, boundarytris, bidxTotidx, tetras


# assuming a sphere, orient triangles, given an interior points
def OrientTriangles(points, boundarytris,interiorpoint):
    
    AB=points[boundarytris[:,0:2]]
    t1 = np.subtract(AB[:,0,:],AB[:,1,:])
    BC=points[boundarytris[:,1:3]]
    t2 = np.subtract(BC[:,0,:],BC[:,1,:])
    # the normal vectors
    sizes = np.linalg.norm(np.cross(t1,t2),axis=1)
    normals=np.cross(t1,t2)/sizes[:,None]
    
    # barycentre of each triangle
    
    barys=(points[boundarytris[:,0]]+points[boundarytris[:,1]]+points[boundarytris[:,2]])/3
    
    # vec from interior point to barycentres
    v = barys-interiorpoint
    # should we flip a pair of bonds? 
    flip = (np.multiply(v, normals).sum(axis=1) <0) 
    
    # make the flips on a copy, not touching the input
    orientedboundarytris=boundarytris.copy()
    for (tidx, t) in enumerate(orientedboundarytris):
        if True==flip[tidx]:
            t[[0,1]]=t[[1,0]]
    
    return orientedboundarytris
    
    
#r_ij: numpy list of bond lengths
# r0_ij: list of rest lengths
#khook: the spring constant
#returns V_ij, a list of the bond energies
def NeoHookean3D(r_ij,r0_ij,khook,MatNon):
    kneo_ij = (r0_ij**2)*khook/3  
    lam_ij=r_ij/r0_ij
    #V_ij=(kneo_ij/2)*((2/lam_ij) + lam_ij**2)
    V_ij=kneo_ij*(  ((1-MatNon)/2)*((2/lam_ij) + lam_ij**2)+ (MatNon/2)*((1/lam_ij)**2 + 2*lam_ij)  )
    V_ij = V_ij -1.5*kneo_ij
    return V_ij

@jit(nopython=True)
def NumbaNeoHookean3D(r_ij,r0_ij,khook,MatNon):
    kneo_ij = (r0_ij**2)*khook/3  
    lam_ij=r_ij/r0_ij
    V_ij=kneo_ij*(  ((1-MatNon)/2)*((2/lam_ij) + lam_ij**2)+ (MatNon/2)*((1/lam_ij)**2 + 2*lam_ij)  )
    # shift so zero extension is 0 energy
    V_ij = V_ij -1.5*kneo_ij
    return V_ij

@jit(nopython=True)
def NumbaSurfaceEnergy3D(r_ij,r0_ij,khook,MatNon,gamma):

    # the surface bonds have a prestretch by a factor gamma:
    r0_ij=gamma*r0_ij

    kneo_ij = (r0_ij**2)*khook/3  
    lam_ij=r_ij/r0_ij
    V_ij=kneo_ij*(  ((1-MatNon)/2)*((2/lam_ij) + lam_ij**2)+ (MatNon/2)*((1/lam_ij)**2 + 2*lam_ij)  )
    # shift so zero extension is 0 energy
    V_ij = V_ij -1.5*kneo_ij
    return V_ij

#@jit(nopython=True)
#def NumbaSurfaceEnergy3D(r_ij,gamma):
#
#    ksurf = 1/(2*np.sqrt(3))*gamma
#    return ksurf*r_ij**2

# Here I implement the bending energy found in, e.g.:
# "Spectrin-Level modelling of the cytoskeleton and optical tweezers stretching of the Erythrocyte", Li, Dao, Lim, Suresh 2005.
# and references therin, in particular:
# "Topology changes in fluid membranes" Boal and Rao 1992
# The formula to be implemented is
# F_b = k_bend*Sum_{a,b}(1-cos(theta_ab - theta_0)). The sum is over tri's sharing an edge on the surface. theta_ab is the angle between their normals. 
def BendingEnergytheta0(P,boundarytris,bidxTotidx,kbend,theta_0):
    
    # first, compute list of normals to the triangles:
    AB=P[boundarytris[:,0:2]]
    t1 = np.subtract(AB[:,0,:],AB[:,1,:])
    BC=P[boundarytris[:,1:3]]
    t2 = np.subtract(BC[:,0,:],BC[:,1,:])
    
    normals= np.cross(t1,t2)
    sizes = np.linalg.norm(normals,axis=1)
    normals=normals/sizes[:,None]
    
    # now, run over the bonds, get the (a,b) pairs of neighboring triangles, and
    # compute the bending energy for each
  
    # first set of triangles, "a",  in the pairings across bonds
    tris_a=boundarytris[bidxTotidx[:,0]]
    #x_a, barycentres:
    x_a=(P[tris_a[:,0]]+P[tris_a[:,1]]+P[tris_a[:,2]])/3   
    # the normals
    n_a = normals[bidxTotidx[:,0]]
    
   
    # second set of triangles, "b",  in the pairings across bonds
    tris_b=boundarytris[bidxTotidx[:,1]]
    #x_b, barycentres:
    x_b=(P[tris_b[:,0]]+P[tris_b[:,1]]+P[tris_b[:,2]])/3   
    # the normals
    n_b = normals[bidxTotidx[:,1]]
    
    # cosines
    costheta_ab = np.multiply(n_a, n_b).sum(axis=1) 
    
    # sines, signed accoring to (x_a-x_b).(n_a-n_b)
    sintheta_ab_unsigned= np.linalg.norm( np.cross(n_a,n_b) ,axis=1)
    signs= np.multiply((n_a-n_b), (x_a-x_b)).sum(axis=1)>0
    # turn it from 0's and 1's to -1's and 1's
    signs = 2*(signs-0.5)
   
    sintheta_ab = signs*sintheta_ab_unsigned
    
    return kbend*( 1-(np.cos(theta_0)*costheta_ab+np.sin(theta_0)*sintheta_ab) )


def BendingEnergy(P,boundarytris,bidxTotidx,kbend):
    
    # first, compute list of normals to the triangles:
    AB=P[boundarytris[:,0:2]]
    t1 = np.subtract(AB[:,0,:],AB[:,1,:])
    BC=P[boundarytris[:,1:3]]
    t2 = np.subtract(BC[:,0,:],BC[:,1,:])
    
    normals= np.cross(t1,t2)
    sizes = np.linalg.norm(normals,axis=1)
    normals=normals/sizes[:,None]
    
    # now, run over the bonds, get the (a,b) pairs of neighboring triangles, and
    # compute the bending energy for each
  
    # first set of triangles, "a",  in the pairings across bonds
    tris_a=boundarytris[bidxTotidx[:,0]]
    # the normals
    n_a = normals[bidxTotidx[:,0]]
    
   
    # second set of triangles, "b",  in the pairings across bonds
    tris_b=boundarytris[bidxTotidx[:,1]]
    # the normals
    n_b = normals[bidxTotidx[:,1]]
    
    # cosines
    costheta_ab = np.multiply(n_a, n_b).sum(axis=1) 
    
    return kbend*(1-costheta_ab)

@jit(nopython=True)
def NumbaBendingEnergy(P,boundarytris,bidxTotidx,kbend):
    
    # first, compute list of normals to the triangles:     
    A=P[boundarytris[:,0]]
    B=P[boundarytris[:,1]]
    t1=A-B
    
    B=P[boundarytris[:,1]]
    C=P[boundarytris[:,2]]
    t2=B-C
    
    normals= np.cross(t1,t2)
    sizes=np.sqrt(np.multiply(normals, normals).sum(axis=1))
    normals=(1/sizes.reshape(-1,1))*normals
    
    # now, run over the bonds, get the (a,b) pairs of neighboring triangles, and
    # compute the bending energy for each
  
    # first set of triangles, "a",  in the pairings across bonds
    tris_a=boundarytris[bidxTotidx[:,0]]
    # the normals
    n_a = normals[bidxTotidx[:,0]]
    
   
    # second set of triangles, "b",  in the pairings across bonds
    tris_b=boundarytris[bidxTotidx[:,1]]
    # the normals
    n_b = normals[bidxTotidx[:,1]]
    
    # cosines
    costheta_ab = np.multiply(n_a, n_b).sum(axis=1) 
    
    return kbend*(1-costheta_ab)

@jit(nopython=True)
def NumbaBendingEnergy_2(P,boundarytris,bidxTotidx,kbend):
    
    # first, compute list of normals to the triangles:     
    normals=np.zeros( (len(boundarytris),3) )
    for i in range(len(boundarytris)):
        
        P0=P[boundarytris[i,0]]
        P1=P[boundarytris[i,1]]
        P2=P[boundarytris[i,2]]
        
        t0x=P1[0]-P0[0]
        t0y=P1[1]-P0[1]
        t0z=P1[2]-P0[2]
        
        t1x=P2[0]-P0[0]
        t1y=P2[1]-P0[1]
        t1z=P2[2]-P0[2]
        
        nx = t0y*t1z- t0z*t1y
        ny = t0z*t1x- t0x*t1z
        nz = t0x*t1y- t0y*t1x
        
        size=np.sqrt(nx*nx+ny*ny+nz*nz)
        
        normals[i,0]=(nx/size)
        normals[i,1]=(ny/size)
        normals[i,2]=(nz/size)
        
    
    costheta_ab=np.zeros(len(bidxTotidx))
    for i in range(len(bidxTotidx)):
        n_a=normals[bidxTotidx[i,0]]
        n_b=normals[bidxTotidx[i,1]]
        costheta_ab[i]=n_a[0]*n_b[0]+n_a[1]*n_b[1]+n_a[2]*n_b[2]
        
    return kbend*(1-costheta_ab)   

@jit(nopython=True)
def NumbaBendingEnergy_theta0(P,boundarytris,bidxTotidx,kbend,theta0):
    
    # first, compute list of normals to the triangles:     
    normals=np.zeros( (len(boundarytris),3) )
    # triangle barycentres
    barys=np.zeros( (len(boundarytris),3) )

    for i in range(len(boundarytris)):
        
        P0=P[boundarytris[i,0]]
        P1=P[boundarytris[i,1]]
        P2=P[boundarytris[i,2]]

        barys[i,:]= (P0+P1+P2)/3
        
        t0x=P1[0]-P0[0]
        t0y=P1[1]-P0[1]
        t0z=P1[2]-P0[2]
        
        t1x=P2[0]-P0[0]
        t1y=P2[1]-P0[1]
        t1z=P2[2]-P0[2]
        
        nx = t0y*t1z- t0z*t1y
        ny = t0z*t1x- t0x*t1z
        nz = t0x*t1y- t0y*t1x
        
        size=np.sqrt(nx*nx+ny*ny+nz*nz)
        
        normals[i,0]=(nx/size)
        normals[i,1]=(ny/size)
        normals[i,2]=(nz/size)
        
   
    # now loop over the bonds, getting cos(theta-theta0)
    cosdeltatheta_ab=np.zeros(len(bidxTotidx))
    for i in range(len(bidxTotidx)):
        n_a=normals[bidxTotidx[i,0]]
        n_b=normals[bidxTotidx[i,1]]
        x_a=barys[bidxTotidx[i,0]]
        x_b=barys[bidxTotidx[i,1]]

        # cosines
        costheta_ab=n_a[0]*n_b[0]+n_a[1]*n_b[1]+n_a[2]*n_b[2]
        # unsignedd sines
        sintheta_ab_unsigned= np.linalg.norm(np.cross(n_a,n_b) ,axis=1)
        # sines, signed accoring to (x_a-x_b).(n_a-n_b)
        signs= np.multiply((n_a-n_b), (x_a-x_b)).sum(axis=1)>0
        # turn it from 0's and 1's to -1's and 1's
        signs = 2*(signs-0.5)
        sintheta_ab = signs*sintheta_ab_unsigned

        cosdeltatheta_ab[i]=costheta_ab*costheta_0[i]+sintheta_ab*sintheta0[i]
    
    return kbend*(1-cosdeltatheta_ab)
        

# use the divergence theorem
def Volume3D(P,boundarytris,bidxTotidx):
    
    # Barycentres:
    x_a=(P[boundarytris[:,0]]+P[boundarytris[:,1]]+P[boundarytris[:,2]])/3   
    
    # first, compute list of normals to the triangles:
    AB=P[boundarytris[:,0:2]]
    t1 = np.subtract(AB[:,0,:],AB[:,1,:])
    BC=P[boundarytris[:,1:3]]
    t2 = np.subtract(BC[:,0,:],BC[:,1,:])
    
    dA= 0.5*np.cross(t1,t2)

    return (np.multiply(x_a,dA).sum(axis=1)/3).sum()

# directly sum the triple product over all tetrahedra
def Volume3D_tetras(P,tetras):
   
    AB=P[tetras[:,[0,1]]]
    t1 = np.subtract(AB[:,0,:],AB[:,1,:])

    BC=P[tetras[:,[0,2]]]
    t2 = np.subtract(BC[:,0,:],BC[:,1,:]) 

    CD=P[tetras[:,[0,3]]]
    t3 = np.subtract(CD[:,0,:],CD[:,1,:])   

    t1ct2=np.cross(t1,t2)
    t3dott1ct2=np.multiply(t3,t1ct2).sum(axis=1)

    return (np.abs(t3dott1ct2)/6)

# directly sum the triple product over all tetrahedra
@jit(nopython=True)
def NumbaVolume3D_tetras(P,tetras):
   
    A=P[tetras[:,0]]
    B=P[tetras[:,1]]
    t1=A-B

    B=P[tetras[:,0]]
    C=P[tetras[:,2]]
    t2=B-C
  
    C=P[tetras[:,0]]
    D=P[tetras[:,3]]
    t3=C-D

    t1ct2=np.cross(t1,t2)
    t3dott1ct2=np.multiply(t3,t1ct2).sum(axis=1)

    return (np.abs(t3dott1ct2)/6)

# directly sum the triple product over all tetrahedra
@jit(nopython=True)
def NumbaVolume3D_tetras_2(P,tetras):
    
    Tot=np.zeros(len(tetras))
    for i in range(len(tetras)):
        
        P0= P[tetras[i,0]]
        P1= P[tetras[i,1]] 
        P2= P[tetras[i,2]] 
        P3= P[tetras[i,3]] 
              
        t0x=P1[0]-P0[0]
        t0y=P1[1]-P0[1]
        t0z=P1[2]-P0[2]
        
        t1x=P2[0]-P0[0]
        t1y=P2[1]-P0[1]
        t1z=P2[2]-P0[2]
        
        t2x=P3[0]-P0[0]
        t2y=P3[1]-P0[1]
        t2z=P3[2]-P0[2]
        
        
        t0ct1x = t0y*t1z- t0z*t1y
        t0ct1y = t0z*t1x- t0x*t1z
        t0ct1z = t0x*t1y- t0y*t1x
        
        t2dott0ct1=t2x*t0ct1x+t2y*t0ct1y+t2z*t0ct1z
        
        Tot[i]=np.abs(t2dott0ct1/6)
      
    return Tot

@jit(nopython=True)
def NumbaMakeBondLengths(P,bonds):
    r_ij=np.zeros(len(bonds))
    
    for i in range(len(bonds)):
        P0 = P[bonds[i,0]]
        P1 = P[bonds[i,1]]
        
        tx=P1[0]-P0[0]
        ty=P1[1]-P0[1]
        tz=P1[2]-P0[2]       
        r_ij[i]=np.sqrt(tx*tx+ty*ty+tz*tz)  
        
    return r_ij

def vTotalArea3D(pts,tri):
    AB=pts[tri[:,0:2]]
    t1 = np.subtract(AB[:,0,:],AB[:,1,:])
    BC=pts[tri[:,1:3]]
    t2 = np.subtract(BC[:,0,:],BC[:,1,:])
    return np.linalg.norm(0.5*np.cross(t1,t2),axis=1).sum()

@jit(nopython=True)
def PointConstraintEnergy(P_ij,P0_ij,pidx,E):
    Energy=0
    for p in pidx:
        Energy+=E*((P_ij[p,0]-P0_ij[p,0])**2)
        Energy+=E*((P_ij[p,1]-P0_ij[p,1])**2)
        Energy+=E*((P_ij[p,2]-P0_ij[p,2])**2)
    return Energy  

def energy3D(P,bondlist,orientedboundarytris,bidxTotidx,tetras,r0_ij,khook,kbend,theta0,B,MatNon,TargetVolumes): 
    # We convert it to a matrix here.
    P_ij = P.reshape((-1, 3))
    # from the bond list, work out what the current bond lengths are:
    AB=P_ij[bondlist]
    t1 = np.subtract(AB[:,0,:],AB[:,1,:])
    r_ij=np.linalg.norm(t1,axis=1)
    # NeoHookean Spring bond energies
    SpringEnergy = NeoHookean3D(r_ij,r0_ij,khook,MatNon).sum()   
    #bond bending energy
    BendingEnergyvar = BendingEnergy(P_ij,orientedboundarytris,bidxTotidx,kbend).sum()
    # Energetic penalty on volume change
    #VolumeConstraintEnergy = B*(Volume3D(P_ij,orientedboundarytris,bidxTotidx)-TargetVolume)**2
    VolumeConstraintEnergy = (B*(Volume3D_tetras(P_ij,tetras)-TargetVolumes)**2).sum()
    return SpringEnergy+BendingEnergyvar+VolumeConstraintEnergy

#P - array of points
# InteriorBonds, SurfaceBonds - lists of bonds on the interior/surface of the objects
# orientedboundarytris - list of (i,j,k) tuples for surface triangles, correctly oriented.
# bidxTotidx - which surface bonds belong to which triangle
# tetras - (i,j,k,l) list of the tetrahedra in the mesh
# rinterior0_ij - list of initial lengths of the interior bonds. 
# rsurface0_ij - list of initial lengths of the surface bonds. 
#@jit(nopython=True)
def Numbaenergy3D(P,InteriorBonds,SurfaceBonds,orientedboundarytris,bidxTotidx,tetras,rinterior0_ij,rsurface0_ij,khook,kbend,gamma,theta0,B,MatNon,TargetVolumes,ConstraintPidx,P0_ij):     
    # We convert it to a matrix here.
    P_ij = P.reshape((-1, 3))

    # Do the interior bonds, Neo Hookean elasticity
    rinterior_ij=NumbaMakeBondLengths(P_ij,InteriorBonds)
    InteriorSpringEnergy = NumbaNeoHookean3D(rinterior_ij,rinterior0_ij,khook,MatNon).sum()   

    # Do the surface
    rsurface_ij=NumbaMakeBondLengths(P_ij,SurfaceBonds)
    SurfaceSpringEnergy=NumbaSurfaceEnergy3D(rsurface_ij,rsurface0_ij,khook,MatNon,gamma).sum()

    #bond bending energy
    BendingEnergyvar = NumbaBendingEnergy_2(P_ij,orientedboundarytris,bidxTotidx,kbend).sum()

    # Energetic penalty on volume change
    VolumeConstraintEnergy = (B*(NumbaVolume3D_tetras_2(P_ij,tetras)-TargetVolumes)**2).sum() 

    #If we are fixing some points in 3D space, apply our constaint
    PointConstraintEnergyvar=PointConstraintEnergy(P_ij,P0_ij,ConstraintPidx,0.01*B)

    return InteriorSpringEnergy+SurfaceSpringEnergy+BendingEnergyvar+VolumeConstraintEnergy+PointConstraintEnergyvar

#@jit(nopython=True)
#def Numbaenergy3D(P,Bonds,orientedboundarytris,bidxTotidx,tetras,r0_ij,khook,kbend,theta0,B,MatNon,TargetVolumes):     
#    # We convert it to a matrix here.
#    P_ij = P.reshape((-1, 3))
#
#    # Do the interior bonds, Neo Hookean elasticity
#    r_ij=NumbaMakeBondLengths(P_ij,Bonds)
#    SpringEnergy = NumbaNeoHookean3D(r_ij,r0_ij,khook,MatNon).sum()   
#
#    # Do the surface
#    #rsurface_ij=NumbaMakeBondLengths(P_ij,SurfaceBonds)
#    #SurfaceSpringEnergy=NumbaSurfaceEnergy3D(rsurface_ij,rsurface0_ij,khook,MatNon,gamma).sum()
#
#    #bond bending energy
#    BendingEnergyvar = NumbaBendingEnergy_2(P_ij,orientedboundarytris,bidxTotidx,kbend).sum()
#
#    # Energetic penalty on volume change
#    VolumeConstraintEnergy = (B*(NumbaVolume3D_tetras_2(P_ij,tetras)-TargetVolumes)**2).sum() 
#
#    return SpringEnergy+BendingEnergyvar+VolumeConstraintEnergy


def  Output3D(Name,DataFolder,OutputMesh,P_ij,bondlist,orientedboundarytris,bidxTotidx,tetras,r0_ij,khook,kbend,theta0,B,MatNon,TargetVolumes,g0): 
    
    # from the bond list, work out what the current bond lengths are:
    AB=P_ij[bondlist]
    t1 = np.subtract(AB[:,0,:],AB[:,1,:])
    r_ij=np.linalg.norm(t1,axis=1)
    # NeoHookean Spring bond energies
    SpringEnergy = NeoHookean3D(r_ij,r0_ij,khook,MatNon)   
    #bond bending energy
    BendingEnergyvar = BendingEnergy(P_ij,orientedboundarytris,bidxTotidx,kbend)
    # Energetic penalty on volume change
    VolumeConstraintEnergy = (B*(Volume3D_tetras(P_ij,tetras)-TargetVolumes)**2)
    
    # write summary stats
    TVolume=Volume3D_tetras(P_ij,tetras).sum()
    TBending=BendingEnergyvar.sum()
    TVolumeConstraint=VolumeConstraintEnergy.sum()
    TSpringEnergy=SpringEnergy.sum()
    TEnergy=SpringEnergy.sum()+BendingEnergyvar.sum()+VolumeConstraintEnergy.sum()
        
    filepath=DataFolder+"OutputSummary.log"
    f=open(filepath,"a")
    if os.stat(filepath).st_size == 0:
        f.write('g0 Volume VolumeConstraint Bending SpringEnergy TotalEnergy \n')

    outputlist=["{:0.5f}".format(x) for x in [g0,TVolume,TVolumeConstraint,TBending,TSpringEnergy,TEnergy]] 
    outputlist.append("\n") 
    f.write(" ".join(outputlist))
    f.close()
      
    # write point data to the meshio object
    OutputMesh.points= P_ij
    
    #write cell data
    bondzeros=np.zeros(len(bondlist))
    interiorbondzeros=np.zeros(len(bondlist)-len(bidxTotidx))
    tetrazeros=np.zeros(len(tetras))
    trizeros=np.zeros(len(orientedboundarytris))
    
    OutputMesh.cell_data['VolumeEnergy']=[bondzeros,trizeros,VolumeConstraintEnergy]
    OutputMesh.cell_data['SpringEnergy']=[SpringEnergy,trizeros,tetrazeros]
    OutputMesh.cell_data['BendingEnergy']=[np.concatenate(( interiorbondzeros,BendingEnergyvar )),trizeros,tetrazeros]
    
    OutputMesh.write(DataFolder+Name,binary=True)  

    


################### FUNCTIONS FOR CALIBRATING THE ELASTIC MODULII ######################

# same as above, but shifted to put the energy minimum at 0
@jit(nopython=True)
def SpringCalibrationEnergy(r_ij,r0_ij,khook,MatNon):
    kneo_ij = (r0_ij**2)*khook/3  
    lam_ij=r_ij/r0_ij
    #V_ij=kneo_ij*((1-MatNon)*lam_ij**2+MatNon*(1/lam_ij)**2)
    V_ij=kneo_ij*(  ((1-MatNon)/2)*((2/lam_ij) + lam_ij**2)+ (MatNon/2)*((1/lam_ij)**2 + 2*lam_ij)  )
    # shift so zero extension is 0 energy
    V_ij = V_ij -1.5*kneo_ij   
    # shift so zero extension is 0 energy
    return V_ij

@jit(nopython=True)
def zavg(Pout_ij,Layer):
    Zavg=0
    for pidx in Layer:
        Zavg+=Pout_ij[pidx,2]
    Zavg/= len(Layer)
    return Zavg


@jit(nopython=True)
def f(theta,r0,R,alpha,beta):
    return (alpha**2-beta**2)*R*np.sin(theta)*np.cos(theta)- alpha*r0[0]*np.sin(theta)+beta*r0[1]*np.cos(theta)
@jit(nopython=True)
def Df(theta,r0,R,alpha,beta):
    return (alpha**2-beta**2)*R*(np.cos(theta)**2-np.sin(theta)**2)- alpha*r0[0]*np.cos(theta)-beta*r0[1]*np.sin(theta)

@jit(nopython=True)
def DistanceToEllipse(r0,R,alpha,beta):
    
    # Initial guess
    theta0=np.arctan2((alpha*r0[1]),(beta*r0[0]))

    # run newtons method
    max_iter=5
    theta = theta0
    for n in range(0,max_iter):
        fxn = f(theta,r0,R,alpha,beta)
        Dfxn = Df(theta,r0,R,alpha,beta)
        theta = theta - fxn/Dfxn
    
    thetafinal=theta 
    
    xellipse=R*alpha*np.cos(thetafinal)
    yellipse=R*beta*np.sin(thetafinal)
    
    deltax= r0[0]-xellipse
    deltay= r0[1]-yellipse
    
    return (thetafinal,xellipse,yellipse,np.sqrt(deltax**2+deltay**2))

@jit(nopython=True)
def EllipseConstraintEnergy(P_ij,lam,E,InputMeshPoints,BoundaryPoints):
    
    Energy=0
    for pidx in BoundaryPoints:
        
        r0=np.array([np.sqrt(P_ij[pidx,0]**2+P_ij[pidx,1]**2),P_ij[pidx,2]])
        distance=0
        thetafinal, xellipse,yellipse,distance= DistanceToEllipse(r0,1,lam[0],lam[2]) 
        Energy += E*distance**2

        #Energy += E*(P_ij[pidx,0]-lam[0]*InputMeshPoints[pidx,0])**2
        #Energy += E*(P_ij[pidx,1]-lam[1]*InputMeshPoints[pidx,1])**2
        #Energy += E*(P_ij[pidx,2]-lam[2]*InputMeshPoints[pidx,2])**2                     
    return Energy


@jit(nopython=True)
def SurfaceConstraintEnergy(P_ij,TopLayer,BottomLayer,z0,E):
    TopEnergy=0
    for pidx in TopLayer:
        TopEnergy+=E*(P_ij[pidx,2]-z0)**2
    BottomEnergy=0
    for pidx in BottomLayer:
        BottomEnergy+=E*(P_ij[pidx,2])**2
        
    return TopEnergy+BottomEnergy  

# apply a linear potential, i.e. constant force, to top and bottom layers.
@jit(nopython=True)
def SurfaceForceEnergy(P_ij,TopLayer,BottomLayer,Fz,Topz0=0,Bottomz0=0):
    TopEnergy=0
    for pidx in TopLayer:
        TopEnergy+=-Fz*(P_ij[pidx,2]-Topz0)
    BottomEnergy=0
    for pidx in BottomLayer:
        BottomEnergy+=Fz*(P_ij[pidx,2]-Bottomz0)
        
    return TopEnergy+BottomEnergy  

@jit
def ModuliiEnergyEllipse(P,bondlist,tetras,r0_ij,khook,B,MatNon,TargetVolumes,lam,E,InputMeshPoints,BoundaryPoints):     
    # We convert it to a matrix here.
    P_ij = P.reshape((-1, 3))
    r_ij=NumbaMakeBondLengths(P_ij,bondlist)
    # NeoHookean Spring bond energies
    SpringEnergy = SpringCalibrationEnergy(r_ij,r0_ij,khook,MatNon).sum()   
    # Energetic penalty on volume change
    VolumeConstraintEnergy = (B*(NumbaVolume3D_tetras_2(P_ij,tetras)-TargetVolumes)**2).sum()
    # top and bottom constraints:
    
    SurfaceConstraintEnergyvar =EllipseConstraintEnergy(P_ij,lam,E,InputMeshPoints,BoundaryPoints)
    return SpringEnergy+VolumeConstraintEnergy+SurfaceConstraintEnergyvar


@jit(nopython=True)
def ModuliiEnergyDisplacement(P,TopLayer,BottomLayer,bondlist,tetras,r0_ij,khook,B,MatNon,TargetVolumes,z0,E):     
    # We convert it to a matrix here.
    P_ij = P.reshape((-1, 3))
    r_ij=NumbaMakeBondLengths(P_ij,bondlist)
    # NeoHookean Spring bond energies
    SpringEnergy = NeoHookeanShifted(r_ij,r0_ij,khook,MatNon).sum()   
    # Energetic penalty on volume change
    VolumeConstraintEnergy = (B*(NumbaVolume3D_tetras_2(P_ij,tetras)-TargetVolumes)**2).sum()
    # top and bottom constraints:
    SurfaceConstraintEnergyvar =SurfaceConstraintEnergy(P_ij,TopLayer,BottomLayer,z0,E)
    return SpringEnergy+VolumeConstraintEnergy+SurfaceConstraintEnergyvar

@jit(nopython=True)
def ModuliiEnergyForce(P,TopLayer,BottomLayer,bondlist,tetras,r0_ij,khook,B,MatNon,TargetVolumes,Fz):     
    # We convert it to a matrix here.
    P_ij = P.reshape((-1, 3))
    r_ij=NumbaMakeBondLengths(P_ij,bondlist)
    # NeoHookean Spring bond energies
    SpringEnergy = NeoHookeanShifted(r_ij,r0_ij,khook,MatNon).sum()   
    # Energetic penalty on volume change
    VolumeConstraintEnergy = (B*(NumbaVolume3D_tetras_2(P_ij,tetras)-TargetVolumes)**2).sum()
    # top and bottom constraints:
    SurfaceConstraintEnergyvar =SurfaceForceEnergy(P_ij,TopLayer,BottomLayer,Fz)
    return SpringEnergy+VolumeConstraintEnergy+SurfaceConstraintEnergyvar


def CalibrationOutput3D(Name,DataFolder,OutputMesh,P_ij,interiorbonds,edgebonds,orientedboundarytris,tetras,r0_ij,khook,B,MatNon,TargetVolumes,TopLayer=None,BottomLayer=None,z0=None,lam=None,E=None,Fz=None,BoundaryPoints=None,InputMeshPoints=None):    
    # from the bond list, work out what the current bond lengths are:
    
    bondlist=np.concatenate((interiorbonds,edgebonds))
    AB=P_ij[bondlist]
    t1 = np.subtract(AB[:,0,:],AB[:,1,:])
    r_ij=np.linalg.norm(t1,axis=1)
    
    Ninterior=len(interiorbonds)
    NExterior=len(edgebonds)
    
    #Spring bond energies, for both the exterior and interior bonds:
    InteriorSpringEnergy = SpringCalibrationEnergy(r_ij[0:Ninterior],r0_ij[0:Ninterior],khook,MatNon)  
    ExteriorSpringEnergy = SpringCalibrationEnergy(r_ij[Ninterior:],r0_ij[Ninterior:],khook,MatNon)  
    SpringEnergy = SpringCalibrationEnergy(r_ij,r0_ij,khook,MatNon)  
    
 
    # Energetic penalty on volume change
    VolumeConstraintEnergy = (B*(Volume3D_tetras(P_ij,tetras)-TargetVolumes)**2)

    
    # write point data to the meshio object
    OutputMesh.points= P_ij

    #write cell data
    bondzeros=np.zeros(len(bondlist))
    tetrazeros=np.zeros(len(tetras))
    trizeros=np.zeros(len(orientedboundarytris))

    OutputMesh.cell_data['VolumeEnergy']=[bondzeros,trizeros,VolumeConstraintEnergy]
    OutputMesh.cell_data['SpringEnergy']=[SpringEnergy,trizeros,tetrazeros]

    OutputMesh.write(DataFolder+Name,binary=True) 

    # write summary stats.
    TVolume=Volume3D_tetras(P_ij,tetras).sum()
    TVolumeConstraint=VolumeConstraintEnergy.sum()
    TInteriorSpringEnergy=InteriorSpringEnergy.sum()
    TExteriorSpringEnergy=ExteriorSpringEnergy.sum()
    TSpringEnergy=SpringEnergy.sum()
    #TSurfaceConstraint =SurfaceConstraintEnergy(P_ij,TopLayer,BottomLayer,z0,E)
    TSurfaceConstraint =EllipseConstraintEnergy(P_ij,lam,E,InputMeshPoints,BoundaryPoints)
       
    #topZavg=zavg(P_ij,TopLayer)
    #bottomZavg=zavg(P_ij,BottomLayer)
    #lam0=1
    #lam=(topZavg-bottomZavg)/lam0
    
   
    filepath=DataFolder+"OutputSummary.log"
    f=open(filepath,"a")
    
    if type(lam)!=type(None):
        if os.stat(filepath).st_size == 0:
            f.write('lambdax lambday lambdaz Volume VolumeConstraint SurfaceConstraint InteriorSpringEnergy, ExteriorSpringEnergy TotalSpringEnergy \n')

        outputlist=["{:0.5f}".format(x) for x in [lam[0],lam[1],lam[2],TVolume,TVolumeConstraint,TSurfaceConstraint,TInteriorSpringEnergy,TExteriorSpringEnergy,TSpringEnergy]] 
        outputlist.append("\n") 
        f.write(" ".join(outputlist)) 
    
    elif z0!=None:
        if os.stat(filepath).st_size == 0:
            f.write('z0 Volume VolumeConstraint SurfaceConstraint lambda SpringEnergy \n')

        outputlist=["{:0.5f}".format(x) for x in [z0,TVolume,TVolumeConstraint,TSurfaceConstraint, lam,TSpringEnergy]] 
        outputlist.append("\n") 
        f.write(" ".join(outputlist))
    else:
        if os.stat(filepath).st_size == 0:
            f.write('Fz Volume VolumeConstraint lambda SpringEnergy \n')

        outputlist=["{:0.5f}".format(x) for x in [Fz,TVolume,TVolumeConstraint,lam,TSpringEnergy]] 
        outputlist.append("\n") 
        f.write(" ".join(outputlist))
    
    f.close()

 
    
    
    
