#!/usr/bin/env python
# coding: utf-8
import meshio
import numpy as np
import copy
import glob
from collections import Counter
import matplotlib.pyplot as plt
import os
import sys
import json
import shutil
import scipy.optimize as opt
from EnergyMinimization import *

# which line of input file defines me?
line=int(sys.argv[1])

# read in arguments from file
reader=open("Parameters.txt","r")
parameters=reader.readlines()[line].split()

kc=float(parameters[0])
B=float(parameters[1])
MatNon=float(parameters[2])

# Target mesh size:
target_a = 0.2
# continuum bending modulus: READ IN FROM COMMAND LINE
kc=float(sys.argv[1])
# continuum shear modulus:
mu=1
# Energetic penalty for volume change , READ IN FROM COMMAND LINE
B=float(sys.argv[2])
# The Material Nonlinearity parameter, between 0 and 1. READ IN FROM COMMAND LINE
MatNon=float(sys.argv[3])
# the spring prestress values 
g0start=1.5
g0end=3
g0step=0.1

# The microscopic values
kbend=kc/target_a
khook = mu
theta0=0

# root folder for data
DataFolder='/mnt/jacb23-XDrive/Physics/ResearchProjects/ASouslov/RC-PH1229/ActiveElastocapillarity/2020-10-23-EnergyMinimization/'
# Folder for the run data
RunFolder="alpha_"+"{0:0.2f}".format(MatNon)+"_B_"+"{0:0.1f}".format(B)+"/"
# Name of the run
RunName=""
# Name of the current file
ScriptName="EnergyMinimizationScript3DCluster.ipynb"

path = DataFolder+RunFolder
# make the folder 
try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)
    
# try and clear out the folder of vtk files and log files, if there was a previous run in it
for filename in glob.glob(path+'*.vtk')+glob.glob(path+'*.log'):
    file_path = os.path.join(path, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))
                
#Dump all the parameters to a file in the run folder        
f=open(DataFolder+RunFolder+"Parameters.log","w+")
datadict= { 
        "a":target_a,
        "kc":kc, 
        "B":B,
        "mu":mu,
        "alpha":MatNon,
        "g0start":g0start,
        "g0end":g0end,
}
json.dump(datadict,f)
f.close()

# and for good measure, dump a copy of this code into the data file too
shutil.copyfile(ScriptName,DataFolder+RunFolder+ScriptName)

# Read in the Mesh
InputMesh=meshio.read("InputMesh.vtk")
OutputMesh = copy.deepcopy(InputMesh)    
InputMesh.write(DataFolder+RunFolder+RunName+"InputMesh.vtk") 

#Make the bond lists, make the oriented boundary triangles list, make the mapping from bonds to boundary triangles
interiorbonds,edgebonds,boundarytris, bidxTotidx, tetras= MakeMeshData3D(InputMesh)
bonds=np.concatenate((interiorbonds,edgebonds))
orientedboundarytris=OrientTriangles(InputMesh.points,boundarytris,np.array([0,0,0]))
boundarytris=orientedboundarytris

# make the preferred rest lengths of the interior springs
interiorpairs=InputMesh.points[interiorbonds]
interiorvecs = np.subtract(interiorpairs[:,0,:],interiorpairs[:,1,:])
InteriorBondRestLengths=np.linalg.norm(interiorvecs,axis=1)

# make the preferred rest lengths of the edge springs. Initially have the at g0=1, but then
#update them in the loop
edgepairs=InputMesh.points[edgebonds]
edgevecs = np.subtract(edgepairs[:,0,:],edgepairs[:,1,:])
InitialEdgeBondRestLengths=np.linalg.norm(edgevecs,axis=1)

# The volume constraint is simply that the target volume should be the initial volume
TargetVolumes=Volume3D_tetras(InputMesh.points,tetras)

# initial input points. Pout changes over time
Pout_ij =InputMesh.points


for g0 in np.arange(g0start,g0end,g0step):
    
    print("Current g0"+"{0:0.2f}".format(g0))

    # the important bit! Giving it the prestress
    EdgeBondRestLengths= g0*InitialEdgeBondRestLengths
    r0_ij=np.concatenate((InteriorBondRestLengths,EdgeBondRestLengths))

    #energy3D(P,bondlist,orientedboundarytris,bidxTotidx,tetras,r0_ij,khook,kbend,theta0,B,MatNon,TargetVolumes): 
    Pout_ij = opt.minimize(energy3D, Pout_ij.ravel()
                            ,options={'gtol':1e-02,'disp': True}  
                            ,args=(bonds
                                  ,orientedboundarytris
                                  ,bidxTotidx
                                  ,tetras
                                  ,r0_ij
                                  ,khook
                                  ,kbend
                                  ,theta0
                                  ,B
                                  ,MatNon
                                  ,TargetVolumes)
                           ).x.reshape((-1, 3))
   

    # write the output 
    OutputMesh.points= Pout_ij  
    OutputMesh.write(DataFolder+RunFolder+RunName+"g0_"+"{0:0.2f}".format(g0)+".vtk",binary=True)  
