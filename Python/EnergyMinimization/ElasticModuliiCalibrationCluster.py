import meshio
import pygalmesh
import pygmsh
import numpy as np
import copy
import glob
from collections import Counter
import matplotlib.pyplot as plt
import os
import json
import shutil
import scipy.optimize as opt
from EnergyMinimization import *
import numba

# Target mesh size:
target_a = 0.2
# continuum shear modulus:
mu=1
# Energetic penalty for volume change
#B=1000000
B=50000
# The Material Nonlinearity parameter, between 0 and 1
MatNon=0.0
khook = mu

# root folder for data
DataFolder=os.getcwd()+'/Data/Scratch/'
# Name of the current file
ScriptName="ElasticModuliiCalibration.ipynb"

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
    ellipsoid =geom.add_cylinder([0, 0.0, 0.0], [0.0, 0.0, 1], 1),
    InputMesh = geom.generate_mesh()
    
interiorbonds,edgebonds,boundarytris, bidxTotidx, tetras= MakeMeshData3D(InputMesh)
bonds=np.concatenate((interiorbonds,edgebonds))
orientedboundarytris=OrientTriangles(InputMesh.points,boundarytris,np.array([0,0,0]))
boundarytris=orientedboundarytris

TopLayer= np.where((InputMesh.points[:,2]>0.99))[0]
BottomLayer= np.where((InputMesh.points[:,2]<0.01))[0]

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

for mode in ("Extension","Compression"):
    Pout_ij =InputMesh.points
    if mode=="Extension":
        Frange=np.arange(0,0.05,0.005)
    elif mode=="Compression":
        Frange=np.arange(0,-0.05,-0.005)
        
    for F in Frange:

        print("Current Fz"+"{0:0.3f}".format(F))
        
        # where are the current top and bottom surfaces, pre minimization? Use this for working out external work
        topZavg=zavg(Pout_ij,TopLayer)
        bottomZavg=zavg(Pout_ij,BottomLayer)
        print(topZavg)
        print(bottomZavg)

        # minimize
        history=[]

        #ModuliiEnergy(P,TopLayer,BottomLayer,bondlist,tetras,r0_ij,z0,khook,B,E,MatNon,TargetVolumes): 
        Pout_ij = opt.minimize(ModuliiEnergy, Pout_ij.ravel()
                               # ,callback=mycallback
                                ,options={'gtol':1e-03,'disp': True}  
                                ,args=(TopLayer
                                      ,BottomLayer
                                      ,bonds
                                      ,tetras
                                      ,r0_ij
                                      ,khook
                                      ,B
                                      ,F
                                      ,MatNon
                                      ,TargetVolumes)
                               ).x.reshape((-1, 3))


        Name="Fz_"+"{0:0.3f}".format(F)+".vtk"
        CalibrationOutput3D(Name
                            ,DataFolder
                            ,OutputMesh
                            ,Pout_ij
                            ,bonds
                            ,orientedboundarytris
                            ,tetras
                            ,r0_ij
                            ,khook
                            ,B
                            ,MatNon
                            ,TargetVolumes
                            ,TopLayer
                            ,BottomLayer
                            ,F)



