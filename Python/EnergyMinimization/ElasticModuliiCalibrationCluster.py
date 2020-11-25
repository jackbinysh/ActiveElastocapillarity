#!/usr/bin/env python
# coding: utf-8
import meshio
import pygmsh
import numpy as np
import copy
import glob
from collections import Counter
import os
import json
import shutil
import scipy.optimize as opt
from EnergyMinimization import *
import numba
# which line of input file defines me?
line=int(sys.argv[1])

# read in arguments from file
reader=open("Parameters.txt","r")
parameters=reader.readlines()[line].split()

# Target mesh size:
target_a = 0.2
# continuum shear modulus:
mu=1
# Energetic penalty for volume change
#B=1000000
B=50000
# Surface Constraint Energy
E=100
# The Material Nonlinearity parameter, between 0 and 1
MatNon=float(parameters[0])
axis=int(parameters[1])
khook = mu

# root folder for data
DataFolder='/mnt/jacb23-XDrive/Physics/ResearchProjects/ASouslov/RC-PH1229/ActiveElastocapillarity/2020-11-18-ModuliiCalibration/'+"alpha_"+"{0:0.2f}".format(MatNon)+"axis_"+"{0:d}".format(axis)+"/"
# Name of the current file
ScriptName="ElasticModuliiCalibrationCluster.py"

try:
    os.mkdir(DataFolder)
except OSError:
    print ("Creation of the directory %s failed" % DataFolder)
else:
    print ("Successfully created the directory %s " % DataFolder)
    
# try and clear out the folder of vtk files and log files, if there was a previous run in it
for filename in glob.glob(DataFolder+'*.vtk')+glob.glob(DataFolder+'*.log'):
    file_path = os.path.join(DataFolder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))
                
#Dump all the parameters to a file in the run folder        
f=open(DataFolder+"Parameters.log","w+")
datadict= { 
        "a":target_a, 
        "B":B,
        "mu":mu,
        "alpha": MatNon
}
json.dump(datadict,f)
f.close()

# and for good measure, dump a copy of this code into the data file too
shutil.copyfile(ScriptName,DataFolder+ScriptName)

with pygmsh.occ.Geometry() as geom:
    geom.characteristic_length_max = target_a
    ellipsoid = geom.add_ball([0.0, 0.0, 0.0], 1)
    InputMesh = geom.generate_mesh()
    
interiorbonds,edgebonds,boundarytris, bidxTotidx, tetras= MakeMeshData3D(InputMesh)
bonds=np.concatenate((interiorbonds,edgebonds))
orientedboundarytris=OrientTriangles(InputMesh.points,boundarytris,np.array([0,0,0]))
boundarytris=orientedboundarytris


BoundaryPoints= np.unique(edgebonds.ravel())

cells=[ ("line", bonds ), ("triangle",boundarytris ), ("tetra",tetras)]
isbond=  np.ones(len(bonds))
isedgebond= np.concatenate( ( np.zeros(len(interiorbonds)),np.ones(len(edgebonds)) ) )
CellDataDict={'isedgebond':[isedgebond,np.zeros(len(boundarytris)),np.zeros(len(tetras))]
              ,'isbond':[isbond,np.zeros(len(boundarytris)),np.zeros(len(tetras))]}

OutputMesh=meshio.Mesh(InputMesh.points, cells, {},CellDataDict)
OutputMesh.write(DataFolder+"InitialMesh.vtk",binary=True) 
      
# make the preferred rest lengths of the interior springs
interiorpairs=InputMesh.points[interiorbonds]
interiorvecs = np.subtract(interiorpairs[:,0,:],interiorpairs[:,1,:])
InteriorBondRestLengths=np.linalg.norm(interiorvecs,axis=1)

# make the preferred rest lengths of the edge springs. Initially have the at g0=1, but then
#update them in the loop
edgepairs=InputMesh.points[edgebonds]
edgevecs = np.subtract(edgepairs[:,0,:],edgepairs[:,1,:])
EdgeBondRestLengths=np.linalg.norm(edgevecs,axis=1)
  
r0_ij=np.concatenate((InteriorBondRestLengths,EdgeBondRestLengths))

# The volume constraint is simply that the target volume should be the initial volume
TargetVolumes=Volume3D_tetras(InputMesh.points,tetras)

for mode in ("Compression","Extension"):
    Pout_ij =InputMesh.points
    if mode=="Extension":
        z0range=np.arange(1,1.6,0.05)
    elif mode=="Compression":
        z0range=np.arange(1,0.4,-0.05)
        
    for z0 in z0range:

        print("Current z0"+"{0:0.3f}".format(z0))  
        if axis==0:
            lam=np.array([z0,1/np.sqrt(z0),1/np.sqrt(z0)])
        elif axis==1:
            lam=np.array([1/np.sqrt(z0),z0,1/np.sqrt(z0)])
        elif axis==2:
            lam=np.array([1/np.sqrt(z0),1/np.sqrt(z0),z0])
            
        
        # minimize
        history=[]
        #def ModuliiEnergyEllipse(P,bondlist,tetras,r0_ij,khook,B,MatNon,TargetVolumes,lam,E,InputMesh,BoundaryPoints): 
        Pout_ij = opt.minimize(ModuliiEnergyEllipse, Pout_ij.ravel()
                                #,callback=mycallback
                                ,options={'gtol':1e-03,'disp': True}  
                                ,args=(bonds
                                      ,tetras
                                      ,r0_ij
                                      ,khook
                                      ,B
                                      ,MatNon
                                      ,TargetVolumes
                                      ,lam
                                      ,E
                                      ,InputMesh.points
                                      ,BoundaryPoints)
                               ).x.reshape((-1, 3))


        Name="z0_"+"{0:0.3f}".format(z0)+".vtk"
               
 #CalibrationOutput3D(Name,DataFolder,OutputMesh,P_ij,bondlist,orientedboundarytris,tetras,r0_ij,khook,B,MatNon,TargetVolumes,TopLayer=None,BottomLayer=None,z0=None,E=None,Fz=None,BoundaryPoints=None,InputMeshPoints=None):          
        CalibrationOutput3D(Name
                            ,DataFolder= DataFolder
                            ,OutputMesh=OutputMesh
                            ,P_ij=Pout_ij
                            ,bondlist=bonds
                            ,orientedboundarytris=orientedboundarytris
                            ,tetras=tetras
                            ,r0_ij=r0_ij
                            ,khook=khook
                            ,B=B
                            ,MatNon=MatNon
                            ,TargetVolumes=TargetVolumes
                            ,z0=z0
                            ,lam=lam
                            ,E=E
                            ,BoundaryPoints=BoundaryPoints
                            ,InputMeshPoints=InputMesh.points)    
    
    
    
    
    
    
    