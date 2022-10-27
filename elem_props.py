import numpy as np
import myconstants as myconst


def ElemMass(M_elem, D_elem, wt_elem, L_elem):
    """
    Calculates local-axis mass matrix of element from inputs

    INPUTS:
        M_elem: (float) mass of element 
        D_elem: (float) outer diameter of element 
        wt_elem: (float) wall thickness of element 
        L_elem: (float) length of element 

    OUTPUTS:
        mel: (12x12) local element mass matrix

    """
    
    area = (np.pi / 4.) * (D_elem**2. - (D_elem - 2. * wt_elem)**2.)
    Ix = (np.pi / 32.) * (D_elem**4. - (D_elem - 2. * wt_elem)**4.) 
    
    a = L_elem / 2.
    a2 = a ** 2.
    rx = Ix / area

    mel = np.zeros((12, 12))

    mel[0, 0] = mel[6, 6] = 70.
    mel[1, 1] = mel[2, 2] = mel[7, 7] = mel[8, 8] = 78.
    mel[3, 3] = mel[9, 9] = 70. * rx
    mel[4, 4] = mel[5, 5] = mel[10, 10] = mel[11, 11] = 8. * a2
    mel[2, 4] = mel[4, 2] = mel[7, 11] = mel[11, 7] = -22. * a
    mel[1, 5] = mel[5, 1] = mel[8, 10] = mel[10, 8] = 22. * a 
    mel[0, 6] = mel[6, 0] = 35. 
    mel[1, 7] = mel[7, 1] = mel[2, 8] = mel[8, 2] = 27. 
    mel[1, 11] = mel[11, 1] = mel[4, 8] = mel[8, 4] = -13. * a
    mel[2, 10] = mel[10, 2] = mel[5, 7] = mel[7, 5] = 13. * a
    mel[3, 9] = mel[9, 3] = 35. * rx
    mel[4, 10] = mel[10, 4] = mel[5, 11] = mel[11, 5] = -6. * a2
    
    mel = mel * M_elem * a / 105.

    return mel
              
def ElemMatStiff(D_elem, wt_elem, L_elem):
    """
    Calculates local-axis material stiffness matrix of element from inputs

    INPUTS:
        D_elem: (float) outer diameter of element 
        wt_elem: (float) wall thickness of element 
        L_elem: (float) length of element 

    OUTPUTS:
        kel_mat: (12x12) local element material stiffness matrix

    """

    EA = (np.pi / 4.) * (D_elem**2. - (D_elem - 2. * wt_elem)**2.) * myconst.E_STL
    EI = (np.pi / 64.) * (D_elem**4. - (D_elem - 2. * wt_elem)**4.) * myconst.E_STL
    EIy = EIz = EI
    Kv  = EIy/(myconst.E_STL*10) # check this!

    # --- Stiffness matrix
    a = EA / L_elem
    b = 12. * EIz / L_elem ** 3.
    c = 6. * EIz / L_elem ** 2.
    d = 12. * EIy / L_elem ** 3.
    e = 6. * EIy / L_elem ** 2.
    f = myconst.G_STL * Kv / L_elem
    g = 2. * EIy / L_elem
    h = 2. * EIz / L_elem

    kel_mat = np.zeros((12,12))
    kel_mat[0,0] = kel_mat[6,6] = a
    kel_mat[1,1] = kel_mat[7,7] = b
    kel_mat[2,2] = kel_mat[8,8] = d
    kel_mat[3,3] = kel_mat[9,9] = f
    kel_mat[4,4] = kel_mat[10,10] = 2.*g
    kel_mat[5,5] = kel_mat[11,11] = 2.*h
    kel_mat[1,5] = kel_mat[5,1] = kel_mat[1,11] = kel_mat[11,1] = c
    kel_mat[2,4] = kel_mat[4,2] = kel_mat[2,10] = kel_mat[10,2] = -1. * e
    kel_mat[7,11] = kel_mat[11,7] = kel_mat[7,5] = kel_mat[5,7] = -1. * c
    kel_mat[8,10] = kel_mat[10,8] = kel_mat[4,8] = kel_mat[8,4] = e

    kel_mat[6,0] = kel_mat[0,6] = -1. * a
    kel_mat[7,1] = kel_mat[1,7] = -1. * b
    kel_mat[8,2] = kel_mat[2,8] = -1. * d
    kel_mat[9,3] = kel_mat[3,9] = -1. * f
    kel_mat[10,4] = kel_mat[4,10] = g
    kel_mat[11,5] = kel_mat[5,11] = h
            
    return kel_mat

def ElemGeomStiff(P_elem, L_elem):
    """
    Calculates local-axis geometrical stiffness matrix of element from inputs

    INPUTS:
        P_elem: (float) axial load on element 
        L_elem: (float) length of element 

    OUTPUTS:
        kel_geom: (12x12) local element geometrical stiffness matrix

    """

    kel_geom = np.zeros((12, 12))

    # See Cook FEA book, page 643
    kel_geom[1, 1] = kel_geom[2, 2] = kel_geom[7, 7] = kel_geom[8, 8] = 6. / (5. * L_elem)
    kel_geom[1, 7] = kel_geom[7, 1] = kel_geom[2, 8] = kel_geom[8, 2] = -6. / (5. * L_elem)
    kel_geom[1, 5] = kel_geom[5, 1] = kel_geom[1, 11] = kel_geom[11, 1] = 1. / 10.
    kel_geom[4, 8] = kel_geom[8, 4] = kel_geom[8, 10] = kel_geom[10, 8] = 1. / 10.
    kel_geom[2, 5] = kel_geom[5, 2] = kel_geom[2, 10] = kel_geom[10, 2] = -1. / 10.
    kel_geom[5, 7] = kel_geom[7, 5] = kel_geom[7, 11] = kel_geom[11, 7] = -1. / 10.
    kel_geom[4, 4] = kel_geom[5, 5] = kel_geom[10, 10] = kel_geom[11, 11] = 2. * L_elem / 15.
    kel_geom[4, 10] = kel_geom[10, 4] = kel_geom[5, 11] = kel_geom[11, 5] = -1. * L_elem / 30.

    kel_geom = kel_geom * P_elem
            
    return kel_geom

def ElemStiff(kel_mat, kel_geom):
    """
    Calculates local-axis stiffness matrix of element from inputs

    INPUTS:
        kel_mat: (12x12) local element material stiffness matrix
        kel_geom: (12x12) local element geometrical stiffness matrix

    OUTPUTS:
        kel: (12x12) local element stiffness matrix

    """
    
    kel = kel_mat + kel_geom

    return kel