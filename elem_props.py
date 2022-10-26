import numpy as np
import myconstants as myconst


def ElemMass(M_elem, L_elem):
    """
    Calculates local-axis mass matrix of element from inputs

    INPUTS:
        L_elem: (float) length of element 
        M_elem: (float) mass of element 

    OUTPUTS:
        mel: (4x4) local element mass matrix

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

def ElemMatStiff(D_elem, wt_elem, L_elem):
    """
    Calculates local-axis material stiffness matrix of element from inputs

    INPUTS:
        D_elem: (float) outer diameter of element 
        wt_elem: (float) wall thickness of element 
        L_elem: (float) length of element 

    OUTPUTS:
        kel_mat: (4x4) local element material stiffness matrix

    """

    EI = (np.pi / 64.) * (D_elem**4. - (D_elem - 2. * wt_elem)**4.) * myconst.E_STL

    kel_mat = np.zeros((4, 4))

    kel_mat[0, 0] = kel_mat[2, 2] = 12. / L_elem**3.
    kel_mat[0, 2] = kel_mat[2, 0] = -12. / L_elem**3.
    kel_mat[0, 1] = kel_mat[1, 0] = kel_mat[0, 3] = kel_mat[3, 0] = 6. / L_elem**2.
    kel_mat[1, 2] = kel_mat[2, 1] = kel_mat[2, 3] = kel_mat[3, 2] = -6. / L_elem**2.
    kel_mat[1, 1] = kel_mat[3, 3] = 4. / L_elem
    kel_mat[1, 3] = kel_mat[3, 1] = 2. / L_elem
    kel_mat = kel_mat * EI
            
    return kel_mat

def ElemGeomStiff(P_elem, L_elem):
    """
    Calculates local-axis geometrical stiffness matrix of element from inputs

    INPUTS:
        P_elem: (float) axial load on element 
        L_elem: (float) length of element 

    OUTPUTS:
        kel_geom: (4x4) local element geometrical stiffness matrix

    """

    kel_geom = np.zeros((4, 4))

    # See Cook FEA book, page 643
    kel_geom[0, 0] = kel_geom[2, 2] = 6. / (5. * L_elem)
    kel_geom[0, 2] = kel_geom[2, 0] = -6. / (5. * L_elem)
    kel_geom[0, 1] = kel_geom[1, 0] = kel_geom[0, 3] = kel_geom[3, 0] = 1. / 10.
    kel_geom[1, 2] = kel_geom[2, 1] = kel_geom[2, 3] = kel_geom[3, 2] = -1. / 10.
    kel_geom[1, 1] = kel_geom[3, 3] = 2. * L_elem / 15.
    kel_geom[1, 3] = kel_geom[3, 1] = -L_elem / 30.

    kel_geom = kel_geom * P_elem

            
    return kel_geom

def ElemStiff(kel_mat, kel_geom):
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