import meshio
import numpy as np
import copy
from collections import Counter
#import matplotlib.pyplot as plt
import os
import sys
import json
import shutil
import scipy.optimize as opt
from numba import jit

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
# return signed cosines and sines of plaquette bending angles
def getCosSintheta(P,boundarytris,bidxTotidx):
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
        
    # now loop over the bonds, getting cos(theta) and sin(theta)
    costheta_ab=np.zeros(len(bidxTotidx))
    sintheta_ab=np.zeros(len(bidxTotidx))
    for i in range(len(bidxTotidx)):
        n_a=normals[bidxTotidx[i,0]]
        n_b=normals[bidxTotidx[i,1]]
        x_a=barys[bidxTotidx[i,0]]
        x_b=barys[bidxTotidx[i,1]]

        # cosines
        costheta_ab[i]=n_a[0]*n_b[0]+n_a[1]*n_b[1]+n_a[2]*n_b[2]
        # unsignedd sines
        sintheta_ab_unsigned= np.linalg.norm(np.cross(n_a,n_b))
        # sines, signed accoring to (x_a-x_b).(n_a-n_b)
        signs= np.dot((n_a-n_b), (x_a-x_b))>0
        # turn it from 0's and 1's to -1's and 1's
        signs = 2*(signs-0.5)
        sintheta_ab[i] = signs*sintheta_ab_unsigned

    return costheta_ab,sintheta_ab

@jit(nopython=True)
def NumbaBendingEnergy_theta0(P,boundarytris,bidxTotidx,kbend,costheta0,sintheta0):

    costheta_ab,sintheta_ab=getCosSintheta(P,boundarytris,bidxTotidx)
    cosdeltatheta_ab=np.zeros(len(bidxTotidx))
    for i in range(len(bidxTotidx)):
        cosdeltatheta_ab[i]=costheta_ab[i]*costheta0[i]+sintheta_ab[i]*sintheta0[i]
    
    return kbend*(1-cosdeltatheta_ab)

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

@jit(nopython=True)
def PointConstraintEnergy(P_ij,P0_ij,pidx,E):
    Energy=0
    for p in pidx:
        Energy+=E*((P_ij[p,0]-P0_ij[p,0])**2)
        Energy+=E*((P_ij[p,1]-P0_ij[p,1])**2)
        Energy+=E*((P_ij[p,2]-P0_ij[p,2])**2)
    return Energy  

#P - array of points
# InteriorBonds, SurfaceBonds - lists of bonds on the interior/surface of the objects
# orientedboundarytris - list of (i,j,k) tuples for surface triangles, correctly oriented.
# bidxTotidx - which surface bonds belong to which triangle
# tetras - (i,j,k,l) list of the tetrahedra in the mesh
# rinterior0_ij - list of initial lengths of the interior bonds. 
# rsurface0_ij - list of initial lengths of the surface bonds. 
#@jit(nopython=True)
def Numbaenergy3D(P,InteriorBonds,SurfaceBonds,orientedboundarytris,bidxTotidx,tetras,rinterior0_ij,rsurface0_ij,costheta0,sintheta0,khook,kbend,gamma,theta0,B,MatNon,TargetVolumes,ConstraintPidx,P0_ij):     
    # We convert it to a matrix here.
    P_ij = P.reshape((-1, 3))

    # Do the interior bonds, Neo Hookean elasticity
    rinterior_ij=NumbaMakeBondLengths(P_ij,InteriorBonds)
    InteriorSpringEnergy = NumbaNeoHookean3D(rinterior_ij,rinterior0_ij,khook,MatNon).sum()   

    # Do the surface
    rsurface_ij=NumbaMakeBondLengths(P_ij,SurfaceBonds)
    SurfaceSpringEnergy=NumbaSurfaceEnergy3D(rsurface_ij,rsurface0_ij,khook,MatNon,gamma).sum()

    #bond bending energy
    BendingEnergyvar = NumbaBendingEnergy_theta0(P_ij,orientedboundarytris,bidxTotidx,kbend,costheta0,sintheta0).sum()

    # Energetic penalty on volume change
    VolumeConstraintEnergy = (B*(NumbaVolume3D_tetras_2(P_ij,tetras)-TargetVolumes)**2).sum() 

    #If we are fixing some points in 3D space, apply our constaint
    PointConstraintEnergyvar=PointConstraintEnergy(P_ij,P0_ij,ConstraintPidx,0.01*B)

    return InteriorSpringEnergy+SurfaceSpringEnergy+BendingEnergyvar+VolumeConstraintEnergy+PointConstraintEnergyvar

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


