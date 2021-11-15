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

