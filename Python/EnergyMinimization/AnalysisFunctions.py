from numpy.linalg import eig, inv, svd
from math import atan2
import numpy as np
import scipy.optimize as opt

def BoundingEllipseOfBestFit(mesh):
    # get the edge bonds
    triangles=mesh.cells[0].data
    x = [[[triangle[0],triangle[1]],[triangle[0],triangle[2]],[triangle[1],triangle[2]] ]   for triangle in triangles]
    flattenedx = [val for sublist in x for val in sublist]
    bonds = [[x[0],x[1]] if x[0]<x[1] else [x[1],x[0]] for x in flattenedx]

    edgebonds=[]
    for elem in bonds:
        if 1==bonds.count(elem):
            edgebonds.append(elem)
   

    bondvertices=[val for bond in edgebonds for val in bond]
    uniqueids= list(set(bondvertices))
    edgepoints= mesh.points[uniqueids]
    
    x=edgepoints[:,0]
    y=edgepoints[:,1]
    return fit_ellipse(x,y)

def TotalArea(mesh):
    TotalArea=0
    
    triangles=mesh.cells[0].data
    for triangle in triangles:
        v1 = mesh.points[triangle[1]]-mesh.points[triangle[0]]
        v2 = mesh.points[triangle[2]]-mesh.points[triangle[0]]
        TriArea= 0.5*np.linalg.norm( (np.cross(v1,v2)))
        TotalArea = TotalArea+TriArea
    return TotalArea

# fit ellipse, taken from https://github.com/ndvanforeest/fit_ellipse/blob/master/run_fit_ellipse.py
def __fit_ellipse(x, y):
    x, y = x[:, np.newaxis], y[:, np.newaxis]
    D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
    S, C = np.dot(D.T, D), np.zeros([6, 6])
    C[0, 2], C[2, 0], C[1, 1] = 2, 2, -1
    U, s, V = svd(np.dot(inv(S), C))
    a = U[:, 0]
    return a

def ellipse_center(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    num = b * b - a * c
    x0 = (c * d - b * f) / num
    y0 = (a * f - b * d) / num
    return np.array([x0, y0])

def ellipse_axis_length(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
    down1 = (b * b - a * c) * (
        (c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a)
    )
    down2 = (b * b - a * c) * (
        (a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a)
    )
    res1 = np.sqrt(up / down1)
    res2 = np.sqrt(up / down2)
    return np.array([res1, res2])

def ellipse_angle_of_rotation(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    return atan2(2 * b, (a - c)) / 2

def fit_ellipse(x, y):
    """@brief fit an ellipse to supplied data points: the 5 params
        returned are:

        M - major axis length
        m - minor axis length
        cx - ellipse centre (x coord.)
        cy - ellipse centre (y coord.)
        phi - rotation angle of ellipse bounding box

    @param x first coordinate of points to fit (array)
    @param y second coord. of points to fit (array)
    """
    a = __fit_ellipse(x, y)
    centre = ellipse_center(a)
    phi = ellipse_angle_of_rotation(a)
    M, m = ellipse_axis_length(a)
    # assert that the major axix M > minor axis m
    if m > M:
        M, m = m, M
    # ensure the angle is betwen 0 and 2*pi
    phi -= 2 * np.pi * int(phi / (2 * np.pi))
    return [M, m, centre[0], centre[1], phi]


# Adapted from http://www.juddzone.com/ALGORITHMS/least_squares_3D_ellipsoid.html
#least squares fit to a 3D-ellipsoid
#  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz  = 1
#
# Note that sometimes it is expressed as a solution to
#  Ax^2 + By^2 + Cz^2 + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz  = 1
# where the last six terms have a factor of 2 in them
# This is in anticipation of forming a matrix with the polynomial coefficients.
# Those terms with factors of 2 are all off diagonal elements.  These contribute
# two terms when multiplied out (symmetric) so would need to be divided by two
def ls_ellipsoid(xx,yy,zz):

   # change xx from vector of length N to Nx1 matrix so we can use hstack
    x = xx[:,np.newaxis]
    y = yy[:,np.newaxis]
    z = zz[:,np.newaxis]

    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz = 1
    J = np.hstack((x*x,y*y,z*z,x*y,x*z,y*z, x, y, z))
    K = np.ones_like(x) #column of ones

    #np.hstack performs a loop over all samples and creates
    #a row in J for each x,y,z sample:
    # J[ix,0] = x[ix]*x[ix]
    # J[ix,1] = y[ix]*y[ix]
    # etc.

    JT=J.transpose()
    JTJ = np.dot(JT,J)
    InvJTJ=np.linalg.inv(JTJ);
    ABC= np.dot(InvJTJ, np.dot(JT,K))

    # Rearrange, move the 1 to the other side
    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz - 1 = 0
    #    or
    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz + J = 0
    #  where J = -1
    vec=np.append(ABC,-1)
    
    # Now convert this polynomial into a center, axes, etc.
    
    Amat=np.array(
    [
    [ vec[0],     vec[3]/2.0, vec[4]/2.0, vec[6]/2.0 ],
    [ vec[3]/2.0, vec[1],     vec[5]/2.0, vec[7]/2.0 ],
    [ vec[4]/2.0, vec[5]/2.0, vec[2],     vec[8]/2.0 ],
    [ vec[6]/2.0, vec[7]/2.0, vec[8]/2.0, vec[9]     ]
    ])

   #See B.Bartoni, Preprint SMU-HEP-10-14 Multi-dimensional Ellipsoidal Fitting
   # equation 20 for the following method for finding the center
    A3=Amat[0:3,0:3]
    A3inv=inv(A3)
    ofs=vec[6:9]/2.0
    center=-np.dot(A3inv,ofs)

    # Center the ellipsoid at the origin
    Tofs=np.eye(4)
    Tofs[3,0:3]=center
    R = np.dot(Tofs,np.dot(Amat,Tofs.T))

    R3=R[0:3,0:3]
    R3test=R3/R3[0,0]
    s1=-R[3, 3]
    R3S=R3/s1
    (el,ec)=eig(R3S)

    recip=1.0/np.abs(el)
    axes=np.sqrt(recip)

    inve=inv(ec) #inverse is actually the transpose here
    return (center,axes,ec,inve,vec)


### Functions for comparison to the analytic predictions ###


def FTot(lam,MatNon,gamma,kappa):
    e=np.lib.scimath.sqrt(1-(1/lam**3))
    Area = (2*np.pi/lam)*( 1+(lam**(3/2)/e)*np.arcsin(e) )  
    Fel=(4/3)*np.pi*(  ((1-MatNon)/2)*((2/lam) + lam**2)+ (MatNon/2)*((1/lam)**2 + 2*lam)  )
    Fbend=(2/3)*np.pi*kappa*(7+(2/lam**3)+3*lam**3*np.arctanh(np.lib.scimath.sqrt(1-lam**3))/np.lib.scimath.sqrt(1-lam**3))
  
    return np.real(Fel+gamma*Area+Fbend)

def FindGlobalMinimum(alpha,gamma,kappa,leftstart,rightstart):
    
    minima=np.zeros(4)
    values=np.zeros(4)
    
    args=(alpha,gamma,kappa)
    
    minima[0]= opt.minimize(FTot,leftstart,args=args).x[0]
    values[0]= opt.minimize(FTot,leftstart,args=args).fun
    
    
    minima[1]= opt.minimize(FTot,rightstart,args=args).x[0]
    values[1]= opt.minimize(FTot,rightstart,args=args).fun
    
    minima[2]= opt.minimize(FTot,1.05,args=args).x[0]
    values[2]= opt.minimize(FTot,1.05,args=args).fun
    
    
    minima[3]= opt.minimize(FTot,0.95,args=args).x[0]
    values[3]= opt.minimize(FTot,0.95,args=args).fun
    
    sort=np.argsort(values)
    
    values=values[sort]
    minima=minima[sort]
    
    return minima,values


def FLandau(epsilon,DeltaAlpha,DeltaGamma,kappa0):
    
    eps2=(8/5)*np.pi*DeltaGamma
    eps3=(1/105)*np.pi*(-140*DeltaAlpha - 208*DeltaGamma)
    eps4=(1/105)*np.pi*(45+210*DeltaAlpha+220*DeltaGamma+504*kappa0)
    
    return eps2*epsilon**2+eps3*epsilon**3+eps4*epsilon**4


def FindGlobalMinimumLandau(DeltaAlpha,DeltaGamma,kappa0,leftstart,rightstart):
    
    minima=np.zeros(4)
    values=np.zeros(4)
    
    args=(DeltaAlpha,DeltaGamma,kappa0)
    
    minima[0]= opt.minimize(FLandau,leftstart,args=args).x[0]
    values[0]= opt.minimize(FLandau,leftstart,args=args).fun
    
    
    minima[1]= opt.minimize(FLandau,rightstart,args=args).x[0]
    values[1]= opt.minimize(FLandau,rightstart,args=args).fun
    
    minima[2]= opt.minimize(FLandau,0.05,args=args).x[0]
    values[2]= opt.minimize(FLandau,0.05,args=args).fun
    
    
    minima[3]= opt.minimize(FLandau,-0.05,args=args).x[0]
    values[3]= opt.minimize(FLandau,-0.05,args=args).fun
    
    sort=np.argsort(values)
    
    values=values[sort]
    minima=minima[sort]
    
    return minima,values



