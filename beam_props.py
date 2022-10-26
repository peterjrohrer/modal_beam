import numpy as np
import myconstants as myconst


def BeamProps(nNode, nElem, D_beam, wt_beam, L_tot):
    """
    Calculates beam properties from inputs

    INPUTS:
        nNode: (int) number of nodes in model
        nElem: (int) number of elements in model
        D_beam: (nElem) diameter of beam elements
        wt_beam: (nElem) wall thickness of beam elements
        L_tot: (float) length of total beam

    OUTPUTS:
        x_beamnode: (nElem) x-locations of each node in the beam
        x_beamelem: (nElem) x-locations of each element center in the beam
        L_beam: (nElem) length of each element in model
        M_beam: (nElem) mass of each element in model
        tot_M_beam: (float) total mass of model

    """
    
    L_per_elem = L_tot/nElem 
    L_beam = np.ones(nElem)*L_per_elem

    x_beamnode = np.concatenate(([0.],np.cumsum(L_beam)))

    h = np.zeros(nElem)
    for i in range(len(h)):
        h[i] = x_beamnode[i + 1] - x_beamnode[i]
        x_beamelem = x_beamnode[i]+(h[i]/2.)

    M_beam = np.zeros_like(D_beam)
    for i in range(nElem):                
        M_beam[i] = L_beam[i] * myconst.RHO_STL * np.pi * 0.25 * ((D_beam[i]*D_beam[i]) - ((D_beam[i] - 2.*wt_beam[i])*(D_beam[i] - 2.*wt_beam[i])))

    tot_M_beam = np.sum(M_beam)

    return x_beamnode, x_beamelem, L_beam, M_beam, tot_M_beam
