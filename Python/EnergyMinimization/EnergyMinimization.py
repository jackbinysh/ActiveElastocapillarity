#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pygmsh
import meshio
import pygalmesh
import hoomd
import hoomd.md
import numpy as np
import copy
import optimesh
from mshr import *
from dolfin import *
from collections import Counter
import matplotlib.pyplot as plt
import os
import json
import shutil
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Helper Functions

# ## Dolfin

# In[2]:


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


# In[3]:


def MakeBondHist(Coordinates,bondlist):
    lengths=[np.linalg.norm(Coordinates[bond[1]] -Coordinates[bond[0]]) for bond in bondlist]
    plt.hist(lengths)
    return lengths


# # Initialisation

# ## Run setup

# User settings: What are the continuum parameters we want? 

# In[4]:


# Target mesh size:
target_a = 0.2
# continuum bending modulus:
kc=1
# continuum shear modulus:
mu=1
# particle mass
m = 1
# damping coefficient
alpha=2

### dynamical data ###
# dt
mydt = 0.003
# Total Number of inflations
NumInflations=250
# the value of g0 (this changes over time)
g0 = 1
#steps to inflate g0 by
deltag=0.1
# Runtime for each inflation
RunStepsPerInflation=200000
def RunStepsPerInflation(g0):
    if g0<5:
        RunSteps=200000
    else:
        RunSteps=1000000
   
    return RunSteps

#logging interval
LogInterval=50000


# Right, lets define the bond type and parameters for each bond. In 2D, we know that the elastic modulii are proportional to the microscopic spring constant. We also know that the continuum and microscopic momdulii are related by a lattice space: $\mu = O(1) k$, $k_c = k_d a$. Since I dont know any better, for know I will just set k to mu.

# In[5]:


kd=kc/target_a
k = mu
theta0=np.pi


# Set up the experiment

# In[6]:


# root folder for data
DataFolder=os.getcwd()+'/Data/'
# Folder for the run data
RunFolder="Sweep/"
# Name of the run
RunName="Disk"
# Name of the current file
ScriptName="EnergyMinimization.ipynb"

path = DataFolder+RunFolder
# make the folder 
try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)
    
# try and clear out the folder if there was a previous run in it
for filename in os.listdir(path):
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
        "mu":mu,
        "m":m,
        "alpha":alpha,
        "go":g0,
        "mydt":mydt,
}
json.dump(datadict,f)
f.close()

# and for good measure, dump a copy of this code into the data file too
shutil.copyfile(ScriptName,DataFolder+RunFolder+ScriptName)


# Make the mesh, write it out to the folder

# In[7]:


InputMesh, OutputMesh, interiorbonds,edgebonds,angletriples = MakeDolfinMesh(target_a,40)
InputMesh.write(DataFolder+RunFolder+RunName+"InputMesh.vtk")
#InputMesh, OutputMesh, interiorbonds,edgebonds,angletriples = MakePygMesh()


# Check out the Mesh. One of the lessons learnt is that you shouldnt have much of a spread in the intial edge lengths

# In[8]:


edgelengths= MakeBondHist(InputMesh.points,edgebonds)
np.mean(edgelengths)


# # Energy Minimization

# In[9]:


def dist(P):
    return np.sqrt((P[:,0]-P[:,0][:,np.newaxis])**2 +
                   (P[:,1]-P[:,1][:,np.newaxis])**2)


# In[10]:


def vNeoHookean(r_ij,khook,r0_ij):
    kneo_ij = (r0_ij**2)*khook/3  
    # the diagonal is irrelevant, just fill it with 1's
    np.fill_diagonal(r_ij,1)
    np.fill_diagonal(r0_ij,1)
    lam_ij=r_ij/r0_ij
    V_ij=(kneo_ij/2)*((2/lam_ij) + lam_ij**2)
    return V_ij
    


# In[11]:


def NeoHookean(r,khook,r0):
    kneo = (r0**2)*khook/3
    lam=r/r0
    V=(kneo/2)*((2/lam) + lam**2)
    return V


# In[12]:


# The argument P is a vector (flattened matrix), needed for scipy
def energy(P):
  
    # We convert it to a matrix here.
    P = P.reshape((-1, 2))
    # We compute the distance matrix.
    r_ij = dist(P)
    # NeoHookean Spring bond energies
    # 0.5 to account for double counting
    return (0.5*A*vNeoHookean(r_ij,1,r0_ij)).sum()    


# In[17]:


# initial points and distance matrix
P0_ij =InputMesh.points[:,0:2] 
r0_ij = dist(2*P0_ij)
# the connectivity matrix
A = np.zeros( (len(P0_ij),len(P0_ij)) )
for bond in edgebonds+interiorbonds:
    A[bond[0],bond[1]]=1
    A[bond[1],bond[0]]=1


# In[18]:


history = []
def mycallback(xi):
    counter=len(history)
    history.append(xi)
    print(counter)
    tempP = xi.reshape((-1, 2))
    #output for visualisation
    OutputMesh.points[:,0:2] = tempP           
    OutputMesh.write(DataFolder+RunFolder+RunName+str(counter)+"Output.vtk",binary=True)


# In[19]:


# the energy minisation:
P1 = opt.minimize(energy, P0_ij.ravel(),method='L-BFGS-B',callback=mycallback).x.reshape((-1, 2))


# In[185]:


#output for visualisation
OutputMesh.points[:,0:2] = P1            
OutputMesh.write(DataFolder+RunFolder+RunName+"Output.vtk",binary=True)


#  ## Running the HOOMD sim

# In[ ]:


hoomd.context.initialize("");


# Define the snapshot. We will have a unique bond id for every bond in the system, as they will all have different rest lengths. We also want to make a distinction between surface and bulk bonds.

# In[ ]:


# number of points and bonds
Npts=len(InputMesh.points);
bonds = interiorbonds+edgebonds
NBonds = len(bonds);
#indices
bondindices = list(range(0,NBonds))
#surface bond or not
bondclassification = [0]*len(interiorbonds)+[1]*len(edgebonds)


# In[ ]:


snapshot = hoomd.data.make_snapshot(N=Npts
                                    ,box=hoomd.data.boxdim(Lx=200, Ly=200,dimensions=2)
                                    ,particle_types=['A']
                                    ,bond_types=[str(i) for i in  bondindices]
                                    ,angle_types=['0']   
                                   );


# Read in the points, bonds and angles

# In[ ]:


# points
snapshot.particles.position[:] = InputMesh.points;
snapshot.particles.typeid[0:Npts]=0
# mass
snapshot.particles.mass[:]=m
# bonds
snapshot.bonds.resize(NBonds)
snapshot.bonds.group[:] = bonds
snapshot.bonds.typeid[:] = bondindices
#angle triples
snapshot.angles.resize(len(angletriples))
snapshot.angles.group[:] = angletriples
snapshot.angles.typeid[:len(angletriples)] =0


# make the initialisation

# In[ ]:


system=hoomd.init.read_snapshot(snapshot);


# ## Bond definitions

# In[ ]:


def NeoHookean(r,rmin,rmax,khook,r0):
    kneo = (r0**2)*khook/3
    lam=r/r0
    V=(kneo/2)*((2/lam) + lam**2)
    F = -(kneo/r0)*(  lam - (1/lam**2)  )
    return (V,F)


# In[ ]:


NeoHookeanbond = hoomd.md.bond.table(width=300)

for i in snapshot.bonds.typeid:
    p1,p2 = snapshot.bonds.group[i]
    
    if(0==bondclassification[i]):
 
        InitLength=np.linalg.norm(InputMesh.points[p2] - InputMesh.points[p1])
        NeoHookeanbond.bond_coeff.set(str(i), func=NeoHookean, rmin=0.5*InitLength,rmax=15*InitLength, coeff=dict(khook=k,r0=InitLength));

    if(1==bondclassification[i]):

        restlength=np.linalg.norm(InputMesh.points[p2] - InputMesh.points[p1])
        NeoHookeanbond.bond_coeff.set(str(i), func=NeoHookean, rmin=0.5*InitLength,rmax=15*InitLength, coeff=dict(khook=k,r0=InitLength));


angle = hoomd.md.angle.harmonic();
angle.angle_coeff.set('0', k=kd, t0=theta0 );


# Define the integrator. In this case, a langevin dynamics. For the damping, particle diameters are set to 1 by default. The $\lambda$ parameter below is such that $\gamma = \lambda d$, so I just have $\gamma=\lambda$. The overdamped assumption is that $m/\gamma  << dt$

# In[ ]:


integrator_mode=hoomd.md.integrate.mode_standard(dt=mydt);
all = hoomd.group.all();
integrator = hoomd.md.integrate.langevin(group=all,kT=0,seed=0,dscale=alpha);


# Define a callback, which we want run periodically

# In[ ]:


class WriteData:
    def __init__(self, system):
        self.system = system;
    def __call__(self, timestep):
        snap = self.system.take_snapshot();
        OutputMesh.points = snap.particles.position
        OutputMesh.point_data={"g0": np.repeat(g0,len(InputMesh.points))}              
        OutputMesh.write(DataFolder+RunFolder+RunName+str(timestep)+".vtk",binary=True)
        
hoomd.analyze.callback(callback=WriteData(system), period=LogInterval);


# # Run the simulation

# In[ ]:


for step in range(0, NumInflations):
    g0 = g0+deltag
    RunSteps=RunStepsPerInflation(g0)
    print("Inflation step is "+str(step)+" g0 increasing to "+str(g0) + ", running for " +str(RunSteps))
    print
    
    # reset the bonds with the new g0
    for i in snapshot.bonds.typeid:
        p1,p2 = snapshot.bonds.group[i]
        if(1==bondclassification[i]):
            restlength=np.linalg.norm(InputMesh.points[p2] - InputMesh.points[p1])
            NeoHookeanbond.bond_coeff.set(str(i), func=NeoHookean, rmin=0.5*InitLength,rmax=15*InitLength, coeff=dict(khook=k,r0=g0*InitLength));

    
    hoomd.run(RunSteps)
    


# In[ ]:




