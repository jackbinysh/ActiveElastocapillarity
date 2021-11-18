#!/usr/bin/env python
# coding: utf-8
import os
import sys
# fix the thread numbers for numba and scripy; apparently this is done before import
#https://stackoverflow.com/questions/30791550/limit-number-of-threads-in-numpy
nthreads=2
os.environ["OMP_NUM_THREADS"] = str(nthreads) # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)  # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = str(nthreads)  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] =str(nthreads) # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] =str(nthreads)  # export NUMEXPR_NUM_THREADS=6
import numpy as np
# fix the thread numbers for numba
# https://numba.pydata.org/numba-doc/latest/user/threading-layer.html
import numba 
numba.set_num_threads(nthreads)
# remainder of the imports
import meshio
import pygmsh
import copy
import glob
from collections import Counter
import json
import shutil
import scipy.optimize as opt
from EnergyMinimization import *

### DATA READ IN ###

# which line of input file defines me?
line=int(sys.argv[1])
# read in arguments from file
reader=open("Parameters.txt","r")
parameters=reader.readlines()[line].split()

### SETTING ALL PARAMETERS. NO MORE HARD CODED NUMBERS AFTER THIS

###  Define the cone geometry ###
target_a=float(parameters[0])
cone_base=[0,0,0]
cone_tip=[0,0,3]
bottomradius=1
topradius=0
interiorpoint=np.array([0,0,0.1]) # needed to orient the mesh below
z_thresh=0.01 #Below this z plane, we constaint the points to not move

### define the run parameters ### 
kbend=float(parameters[0]) # discrete bending modulus: READ IN FROM COMMAND LINE
khook=1 # hookean spring constant:
B=100000 # Energetic penalty for volume change 
MatNon=0 # Material Nonlinearity
g0range=np.arange(1,1.6,0.1) # the spring prestress values 

### IO names ###
ExperimentFolder="/mnt/jacb23-XDrive/Physics/ResearchProjects/ASouslov/RC-PH1229/ActiveElastocapillarity/2021-11-16-ConeEnergyMinimization/" # root folder for data
#DataFolder="/home/jackbinysh/Code/ActiveElastocapillarity/Python/EnergyMinimization/Data/Scratch/"
DataFolder=ExperimentFolder+"kbend_"+"{0:0.1f}".format(kbend)+"/"
ScriptName="EnergyMinimizationScriptCone.py" # Name of the current file
FunctionFileName="EnergyMinimization.py" # Name of the file of functions used for this run

### I/O SETUP ###

try:
    os.mkdir(ExperimentFolder)
except OSError:
    print ("Creation of the directory %s failed" % ExperimentFolder)
else:
    print ("Successfully created the directory %s " % ExperimentFolder)

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
        "kbend":kbend, 
        "B":B,
        "khook":khook,
        "alpha": MatNon
}
json.dump(datadict,f)
f.close()

# Dump an exact copy of this code into the data file
shutil.copyfile(ScriptName,DataFolder+ScriptName)
shutil.copyfile(FunctionFileName,DataFolder+FunctionFileName)

#redirect stdout and stderr to a file in the output folder
sys.stdout = open(DataFolder+"stdout.log", 'w+')
sys.stderr = open(DataFolder+"stderr.log", 'w+')

### MESH GENERATION ###

# Make the Mesh
with pygmsh.occ.Geometry() as geom:
    geom.characteristic_length_max = target_a
    cone = geom.add_cone(cone_base,cone_tip, bottomradius,topradius)
    InputMesh = geom.generate_mesh()

#Make the bond lists, make the oriented boundary triangles list, make the mapping from bonds to boundary triangles
interiorbonds,edgebonds,boundarytris, bidxTotidx, tetras= MakeMeshData3D(InputMesh)
bonds=np.concatenate((interiorbonds,edgebonds))
orientedboundarytris=OrientTriangles(InputMesh.points,boundarytris,)
boundarytris=orientedboundarytris

# For the cone, the bottom layer is constrained not to move
ConstrainedPidx=(InputMesh.points[:,2]<z_thresh).nonzero()[0]

# Write a copy of the input Mesh, for visualisation
cells=[ ("line", bonds ), ("triangle",boundarytris ), ("tetra",tetras)]
isbond=  np.ones(len(bonds))
isedgebond= np.concatenate( ( np.zeros(len(interiorbonds)),np.ones(len(edgebonds)) ) )
CellDataDict={'isedgebond':[isedgebond,np.zeros(len(boundarytris)),np.zeros(len(tetras))]
              ,'isbond':[isbond,np.zeros(len(boundarytris)),np.zeros(len(tetras))]}

OutputMesh=meshio.Mesh(InputMesh.points, cells, {},CellDataDict)
OutputMesh.write(DataFolder+"InitialMesh.vtk",binary=True)

### ENERGY MINIMIIZATION ###

# Make the preferred rest lengths of the interior springs
interiorpairs=InputMesh.points[interiorbonds]
interiorvecs = np.subtract(interiorpairs[:,0,:],interiorpairs[:,1,:])
rinterior0_ij=np.linalg.norm(interiorvecs,axis=1)

# Make the preferred rest lengths of the edge springs. 
edgepairs=InputMesh.points[edgebonds]
edgevecs = np.subtract(edgepairs[:,0,:],edgepairs[:,1,:])
rsurface0_ij=np.linalg.norm(edgevecs,axis=1)

# Make the target volumes. The constraint is simply that the target volume should be the initial volume
TargetVolumes=NumbaVolume3D_tetras_2(InputMesh.points,tetras)

# Make the intial angles of the mesh
costheta0,sintheta0=getCosSintheta(InputMesh.points,boundarytris,bidxTotidx)

# Make the vector of material Nonlinearity values. Here we have all zeros
MatNonvec = np.concatenate((np.repeat(MatNon,len(interiorbonds)), np.repeat(0,len(edgebonds))))

# initial input points. Pout changes over time
Pout_ij =InputMesh.points
P0_ij =InputMesh.points

for g0 in g0range:

    Pout_ij = opt.minimize(Numbaenergy3D, Pout_ij.ravel()
                            ,callback=mycallback
                            ,options={'gtol':1e-2,'disp': True}  
                            ,args=(interiorbonds
                                  ,edgebonds
                                  ,orientedboundarytris
                                  ,bidxTotidx
                                  ,tetras
                                  ,rinterior0_ij
                                  ,rsurface0_ij
                                  ,costheta0
                                  ,sintheta0
                                  ,khook
                                  ,kbend
                                  ,g0
                                  ,B
                                  ,MatNon
                                  ,TargetVolumes
                                  ,ConstrainedPidx
                                  ,P0_ij)
                           ).x.reshape((-1, 3))


    # output the resulting shape
    Name="g0_"+"{0:0.4f}".format(g0)+".vtk"
    r0_ij=np.concatenate((rinterior0_ij,g0*rsurface0_ij))
    Output3D(Name
             ,DataFolder
             ,OutputMesh
             ,Pout_ij
             ,interiorbonds
            ,edgebonds
            ,orientedboundarytris
            ,bidxTotidx
            ,tetras
            ,rinterior0_ij
            ,rsurface0_ij
            ,costheta0
            ,sintheta0
            ,khook
            ,kbend
            ,g0
            ,B
            ,MatNon
            ,TargetVolumes)

sys.stdout.close()
sys.stderr.close()
reader.close()
