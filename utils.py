'''
Utility functions, adapted from WELIB by Emmanuel Branlard

'''

import numpy as np
import numpy.ma as ma

def LinearDOFMappingSingle(nElem, nNodesPerElem, nDOFperNode):
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

def LinearDOFMapping(nNodesPerElem, nDOFperNode, nElems, atNodes):
    """ 
    returns the mappings from nodes to DOF and element to nodes and DOF
    for a structure with the same type of elements, assuming nodes are one after the other

    INPUTS:
        nElems: (nBeams) vector of the number of elements in each beam
        atNode: (nBeams - 1) vector of the attachment nodes for extra beams

    """
    nBeams = len(nElems)
    nNodes = []
    for i in range(nBeams): nNodes.append(nElems[i]+1)
    
    layout = []
    layout.append([0 + i for i in range(nNodes[0])])
    
    for i in range(nBeams - 1):
        # attach = np.arange(np.max(layout),np.max(layout)+11)
        attach = [max(layout[-1]) + i for i in range(nNodes[i+1])]
        attach[0] = atNodes[i]
        layout.append(attach)

    allNodes = [item for sublist in layout for item in sublist]
    nNodes_tot = len(set(allNodes)) # total number of nodes in system

    Nodes2DOF=np.zeros((nNodes_tot,nDOFperNode), dtype=int)
    Elem2DOF=np.zeros((sum(nElems),nDOFperNode*nNodesPerElem),dtype=int)
    Elem2Nodes=np.zeros((sum(nElems),nNodesPerElem), dtype=int)

    for i in range(nNodes_tot):
        Nodes2DOF[i,:]=np.arange(i*nDOFperNode, (i+1)*nDOFperNode) 
    
    idx = 0
    for j in range(nBeams):
        if j != 0: idx += nElems[j-1]

        for i in range(nElems[j]):
            Elem2DOF[idx+i,:]=np.concatenate((Nodes2DOF[layout[j][i],:], Nodes2DOF[layout[j][i+1],:]))
        
        for i in range(nElems[j]):
            Elem2Nodes[idx+i,:]=(layout[j][i],layout[j][i+1])
    
    return Elem2Nodes, Nodes2DOF, Elem2DOF, layout, nNodes_tot

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

def BuildGlobalMatrix(KK, Ke, index):
    """Assembly of element matrices into the system matrix
    INPUTS
        KK - system matrix
        Ke  - element matrix
        index - d.o.f. vector associated with an element
    """
    for i,ii in enumerate(index):
        for j,jj in enumerate(index):
            KK[ii,jj] += Ke[i,j]
    #
    #KK[np.ix_(index,index)] += Ke
    return KK

def insertFixedBCinModes(Qr, Tr):
    """
    Qr : (nr x nr) reduced modes
    Tr : (n x nr) reduction matrix that removed "0" fixed DOF
          such that  Mr = Tr' MM Tr
    """
    return Tr.dot(Qr)