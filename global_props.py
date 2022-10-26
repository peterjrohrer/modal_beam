import numpy as np
import myconstants as myconst


def GlobalMass(M_elem, L_elem):
    """
    Calculates local-axis mass matrix of element from inputs

    INPUTS:
        mel: (nElem, 4, 4) local element mass matrices for all elements 

    OUTPUTS:
        M_glob: (nDOF, nDOF) global finite element model mass matrix

    """
    
    mel = np.zeros((4, 4))

    mel[0, 0] = mel[2, 2] = 156.
    mel[1, 1] = mel[3, 3] = 4. * L_elem**2.
    mel[0, 1] = mel[1, 0] = 22. * L_elem
    mel[2, 3] = mel[3, 2] = -22. * L_elem
    mel[0, 2] = mel[2, 0] = 54.
    mel[1, 2] = mel[2, 1] = 13. * L_elem
    mel[0, 3] = mel[3, 0] = -13. * L_elem
    mel[1, 3] = mel[3, 1] = -3. * L_elem**2.
    
    mel = mel * M_elem * L_elem / 420.

    return mel

def GlobalStiff(kel_mat, kel_geom):
    """
    Calculates local-axis stiffness matrix of element from inputs

    INPUTS:
        kel_mat: (4x4) local element material stiffness matrix
        kel_geom: (4x4) local element geometrical stiffness matrix

    OUTPUTS:
        kel: (4x4) local element stiffness matrix

    """
    
    kel = kel_mat + kel_geom

    return kel