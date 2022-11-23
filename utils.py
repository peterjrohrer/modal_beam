'''
Utility functions, adapted from WELIB by Emmanuel Branlard

'''

import numpy as np
import scipy.linalg


def LinearDOFMapping(nElem, nNodesPerElem, nDOFperNode):
    """ 
    returns the mappings from nodes to DOF and element to nodes and DOF
    for a structure with the same type of elements, assuming nodes are one after the other
    """
    nNodes = (nNodesPerElem-1)*nElem+1 # total number of nodes in system
    Nodes2DOF=np.zeros((nNodes,nDOFperNode), dtype=int)
    for i in np.arange(nNodes):
        Nodes2DOF[i,:]=np.arange(i*6, (i+1)*6) 
    Elem2DOF=np.zeros((nElem,nDOFperNode*nNodesPerElem),dtype=int)
    for i in np.arange(nElem):
        Elem2DOF[i,:]=np.concatenate((Nodes2DOF[i,:], Nodes2DOF[i+1,:]))
    Elem2Nodes=np.zeros((nElem,nNodesPerElem), dtype=int)
    for i in np.arange(nElem):
        Elem2Nodes[i,:]=(i,i+1)
    return Elem2Nodes, Nodes2DOF, Elem2DOF

def elementDCMforHorizontal(nElem):
    """ Generate element Direction cosine matrices (DCM) 
    for pure horizontal elements

    INPUTS:
        nElem: scalar
    OUTPUTS:
        DCM:  (nElem) x 3 x 3
    """

    DCM = np.zeros((nElem,3,3))
    
    ## Match SIMA directions, horizontal beam
    DCM[:,0,0] = 1. # cosine between global-x and local-x
    DCM[:,1,1] = 1. # cosine between global-y and local-y
    DCM[:,2,2] = 1. # cosine between global-z and local-z

    return DCM
    
def elementDCMforVertical(nElem):
    """ Generate element Direction cosine matrices (DCM) 
    for pure vertical elements

    INPUTS:
        nElem: scalar
    OUTPUTS:
        DCM:  (nElem) x 3 x 3
    """

    DCM = np.zeros((nElem,3,3))
    
    ## Match SIMA directions 
    DCM[:,0,2] = 1. # cosine between global-z and local-x
    DCM[:,1,1] = -1. # cosine between global-y and local-y
    DCM[:,2,0] = 1. # cosine between global-x and local-z

    return DCM

def elementDCMforPontoons(nElem, nPont):
    """ Generate element Direction cosine matrices (DCM) 
    for pontoon elements

    INPUTS:
        nElem: scalar, number of elements per pontoon
        nPont: scalar, number of pontoons
    OUTPUTS:
        DCM:  (nElem*nPont) x 3 x 3
    """

    DCM_p1 = np.zeros((nElem,3,3))
    DCM_p2 = np.zeros((nElem,3,3))
    DCM_p3 = np.zeros((nElem,3,3))
    # Preserve global z direction in all local axes
    if nPont == 3:
        # Pontoon 1 has same direction as global axes
        DCM_p1[:,0,0] = 1.
        DCM_p1[:,1,1] = 1.
        DCM_p1[:,2,2] = 1.
        # Pontoon 2 in the global negative y-direction
        DCM_p2[:,0,0] = np.cos((4./3.)*np.pi)
        DCM_p2[:,0,1] = np.cos((5./6.)*np.pi)
        DCM_p2[:,1,0] = np.cos((11./6.)*np.pi)
        DCM_p2[:,1,1] = np.cos((4./3.)*np.pi)
        DCM_p2[:,2,2] = 1.
        # Pontoon 3 in the global positive y-direction
        DCM_p3[:,0,0] = np.cos((2./3.)*np.pi)
        DCM_p3[:,0,1] = np.cos((1./6.)*np.pi)
        DCM_p3[:,1,0] = np.cos((7./6.)*np.pi)
        DCM_p3[:,1,1] = np.cos((2./3.)*np.pi)
        DCM_p3[:,2,2] = 1.
    else:
        raise Exception('Not defined for %d pontoons' %nPont)

    DCM = np.concatenate((DCM_p1, DCM_p2, DCM_p3), axis=0)
    
    return DCM

def elementDCMfromBeamNodes(x_nodes, y_nodes, z_nodes, phi=None):
    """ Generate element Direction cosine matricse (DCM) 
    from a set of ordered node coordinates defining a beam mean line

    INPUTS:
        x_nodes: (nNodes)
        y_nodes: (nNodes)
        z_nodes: (nNodes)
        phi (optional): nNodes angles about mean line to rotate the section axes
    OUTPUTS:
        DCM:  3 x 3 x (nNodes-1)
    """
    def null(a, rtol=1e-5):
        u, s, v = np.linalg.svd(a)
        rank = (s > rtol*s[0]).sum()
        return v[rank:].T.copy()

    nodes = np.vstack((x_nodes,y_nodes,z_nodes))
    nElem= nodes.shape[1] - 1
    DCM = np.zeros((nElem,3,3))

    for i in np.arange(nElem):
        dx= (nodes[:,i+1]-nodes[:,i]).reshape(3,1)
        le = np.linalg.norm(dx) # element length
        e1 = dx/le # tangent vector
        if i==0:
            e1_last = e1
            e2_last = null(e1.T)[:,0].reshape(3,1) # x,z-> y , y-> -x 
        # normal vector
        de1 = e1 - e1_last
        if np.linalg.norm(de1)<1e-8:
            e2 = e2_last
        else:
            e2 = de1/np.linalg.norm(de1)
        # # Rotation about e1
        # if phi is not None:
        #     R  = np.cos(phi[i])*np.eye(3) + np.sin(phi[i])*skew(e1) + (1-np.cos(phi[i]))*e1.dot(e1.T);
        #     e2 = R.dot(e2)
        # Third vector
        e3=np.cross(e1.ravel(),e2.ravel()).reshape(3,1)
        DCM[i,:,:]= np.column_stack((e1,e2,e3)).T
        e1_last= e1
        e2_last= e2
    return DCM

def transformMatrixfromDCM(DCM):
    """ Generate transformation matrix from DCMs 

    INPUTS:
        DCM: (nElem)
    OUTPUTS:
        RR:  (nElem) x 12 x 12
    """
    nElem = DCM.shape[0]
    RR = np.zeros((nElem,12,12))

    for i in range(nElem):
        R = DCM[i,:,:]
        RR[i,:,:] = scipy.linalg.block_diag(R,R,R,R)

    return RR
