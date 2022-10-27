import numpy as np
from scipy import linalg

def Modeshape(nNode, nElem, nDOFperNode, eigvec):
    z_beamnode = np.zeros(nNode)
    rot_beamnode = np.zeros(nNode)

    x_beamnode = eigvec[0:(nElem + 1) * nDOFperNode:nDOFperNode]
    y_beamnode = eigvec[1:(nElem + 2) * nDOFperNode:nDOFperNode]
    z_beamnode = eigvec[2:(nElem + 3) * nDOFperNode:nDOFperNode]
    th_x_beamnode = eigvec[3:(nElem + 4) * nDOFperNode:nDOFperNode]
    th_y_beamnode = eigvec[4:(nElem + 6) * nDOFperNode:nDOFperNode]
    th_z_beamnode = eigvec[5:(nElem + 6) * nDOFperNode:nDOFperNode]

    return x_beamnode, y_beamnode, z_beamnode

def SplineLHS(x_beamnode):
    nNode = len(x_beamnode)

    h = np.zeros(nNode - 1)
    for i in range(nNode - 1):
        h[i] = x_beamnode[i + 1] - x_beamnode[i]

    spline_lhs = np.zeros((nNode, nNode))

    ## --- SparOpt 
    # Looks like not-a-knot
    # https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.296.7452&rep=rep1&type=pdf
    for i in range(1, nNode - 1):
        spline_lhs[i, i] = 2. * (h[i] + h[i - 1])
        spline_lhs[i, i - 1] = h[i]
        spline_lhs[i, i + 1] = h[i - 1]

    spline_lhs[0, 0] = h[1]
    spline_lhs[0, 1] = h[0] + h[1]
    spline_lhs[-1, -1] = h[-2]
    spline_lhs[-1, -2] = h[-1] + h[-2]

    return spline_lhs

def SplineRHS(x_beamnode, z_beamnode):
    nNode = len(x_beamnode)

    h = np.zeros(nNode - 1)
    delta = np.zeros(nNode - 1)
    for i in range(nNode - 1):
        h[i] = x_beamnode[i + 1] - x_beamnode[i]
        delta[i] = (z_beamnode[i + 1] - z_beamnode[i]) / (x_beamnode[i + 1] - x_beamnode[i])

    spline_rhs = np.zeros(nNode)

    ## --- SparOpt 
    # Looks like not-a-knot
    # https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.296.7452&rep=rep1&type=pdf
    for i in range(1, nNode-1):
        spline_rhs[i] = 3. * (h[i - 1] * delta[i] + h[i] * delta[i - 1])

    spline_rhs[0] = ((2. * h[1] + 3. * h[0]) * h[1] * delta[0] + h[0]**2. * delta[1]) / (h[0] + h[1])
    spline_rhs[-1] = ((2. * h[-2] + 3. * h[-1]) * h[-2] * delta[-1] + h[-1]**2. * delta[-2]) / (h[-1] + h[-2])

    return spline_rhs

def SolveSpline(lhs, rhs):

    soln = linalg.solve(lhs, rhs)

    return soln

def ModeshapeElem(x_beamnode, z_beamnode, z_d_beamnode):
    nElem = len(x_beamnode) - 1

    h = np.zeros(nElem)
    for i in range(nElem):
        h[i] = x_beamnode[i + 1] - x_beamnode[i]

    z_beamelem = np.zeros(nElem)

    for i in range(nElem):
        z_beamelem[i] = (z_beamnode[i+1] + z_beamnode[i])/2. - (1./8.)*h[i]*(z_d_beamnode[i+1] - z_d_beamnode[i])

    return z_beamelem