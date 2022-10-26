'''
Utility functions, adapted from WELIB by Emmanuel Branlard

'''

import numpy as np

def LinearDOFMapping(nElem, nNodesPerElem, nDOFperNode):
    """ 
    returns the mappings from nodes to DOF and element to nodes and DOF
    for a structure with the same type of elements, assuming nodes are one after the other
    """
    nNodes = (nNodesPerElem-1)*nElem+1 # total number of nodes in system
    Nodes2DOF=np.zeros((nNodes,nDOFperNode), dtype=int)
    for i in np.arange(nNodes):
        Nodes2DOF[i,:]=np.arange(i*nDOFperNode, (i+1)*nDOFperNode) 
    Elem2DOF=np.zeros((nElem,nDOFperNode*nNodesPerElem),dtype=int)
    for i in np.arange(nElem):
        Elem2DOF[i,:]=np.concatenate((Nodes2DOF[i,:], Nodes2DOF[i+1,:]))
    Elem2Nodes=np.zeros((nElem,nNodesPerElem), dtype=int)
    for i in np.arange(nElem):
        Elem2Nodes[i,:]=(i,i+1)
    return Elem2Nodes, Nodes2DOF, Elem2DOF

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