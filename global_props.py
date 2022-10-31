import numpy as np
import myconstants as myconst
import scipy.linalg

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

def PointMassMatrix(m=None, J_G=None, Ref2COG=None):
    """ 
    Rigid body mass matrix (6x6) at a given reference point: 
      the center of gravity (if Ref2COG is None) 


    INPUTS:
     - m/tip: (scalar) body mass 
                     default: None, no mass
     - J_G: (3-vector or 3x3 matrix), diagonal coefficients or full inertia matrix
                     with respect to COG of body! 
                     The inertia is transferred to the reference point if Ref2COG is not None
                     default: None 
     - Ref2COG: (3-vector) x,y,z position of center of gravity (COG) with respect to a reference point
                     default: None, at first/last node.
    OUTPUTS:
      - M_point (6x6) : rigid body mass matrix at COG or given point 
    """

    # Default values
    if m is None: m=0
    if Ref2COG is None: Ref2COG=(0,0,0)
    if J_G is None: J_G=np.zeros((3,3))
    if len(J_G.flatten())==3:
        J_G = J_G * np.eye(3)

    M_point = np.zeros((6,6))
    x,y,z = Ref2COG
    Jxx,Jxy,Jxz = J_G[0,:]
    _  ,Jyy,Jyz = J_G[1,:]
    _  ,_  ,Jzz = J_G[2,:]

    # M_point[0,0] = M_point[1,1] = M_point[2,2] = m
    # M_point[0,4] = M_point[4,0] = z*m
    # M_point[0,5] = M_point[5,0] = -1.*y*m
    # M_point[1,3] = M_point[3,1] = -1.*z*m
    # M_point[2,3] = M_point[3,2] = y*m
    # M_point[2,4] = M_point[4,2] = -1.*x*m
    # M_point[1,5] = M_point[5,1] = x*m
    
    M_point[0, :] =[   m     ,   0     ,   0     ,   0                 ,  z*m                , -y*m                 ]
    M_point[1, :] =[   0     ,   m     ,   0     , -z*m                ,   0                 ,  x*m                 ]
    M_point[2, :] =[   0     ,   0     ,   m     ,  y*m                , -x*m                ,   0                  ]
    M_point[3, :] =[   0     , -z*m    ,  y*m    , Jxx + m*(y**2+z**2) , Jxy - m*x*y         , Jxz  - m*x*z         ]
    M_point[4, :] =[  z*m    ,   0     , -x*m    , Jxy - m*x*y         , Jyy + m*(x**2+z**2) , Jyz  - m*y*z         ]
    M_point[5, :] =[ -y*m    , x*m     ,   0     , Jxz - m*x*z         , Jyz - m*y*z         , Jzz  + m*(x**2+y**2) ]
    
    return M_point