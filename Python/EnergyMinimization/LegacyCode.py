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

def vTotalArea3D(pts,tri):
    AB=pts[tri[:,0:2]]
    t1 = np.subtract(AB[:,0,:],AB[:,1,:])
    BC=pts[tri[:,1:3]]
    t2 = np.subtract(BC[:,0,:],BC[:,1,:])
    return np.linalg.norm(0.5*np.cross(t1,t2),axis=1).sum()


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

@jit(nopython=True)
def NumbaNeoHookean3D(r_ij,r0_ij,khook,MatNon):
    kneo_ij = (r0_ij**2)*khook/3  
    lam_ij=r_ij/r0_ij
    V_ij=kneo_ij*(  ((1-MatNon)/2)*((2/lam_ij) + lam_ij**2)+ (MatNon/2)*((1/lam_ij)**2 + 2*lam_ij)  )
    # shift so zero extension is 0 energy
    V_ij = V_ij -1.5*kneo_ij
    return V_ij

