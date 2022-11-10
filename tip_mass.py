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

import numpy as np
import openmdao.api as om

class TipMass(om.ExplicitComponent):

    def setup(self):
        self.add_input('tip_mass', val=0., units='kg')
        self.add_input('ref_to_cog', val=np.zeros(3), units='m')
        self.add_input('tip_inertia', val=np.zeros((3,3)), units='kg*m*m')

        self.add_output('tip_mass_mat', val=np.zeros((6,6)))

    def setup_partials(self):
        self.declare_partials('tip_mass_mat', ['tip_mass', 'ref_to_cog', 'tip_inertia'])

    def compute(self, inputs, outputs):
        m = inputs['tip_mass']
        Ref2COG = inputs['ref_to_cog']
        J_G = inputs['tip_inertia']

        M_point = np.zeros((6,6))
        x,y,z = Ref2COG
        Jxx,Jxy,Jxz = J_G[0,:]
        _  ,Jyy,Jyz = J_G[1,:]
        _  ,_  ,Jzz = J_G[2,:]

        M_point[0,0] = M_point[1,1] = M_point[2,2] = m
        M_point[0,4] = M_point[4,0] = z*m
        M_point[0,5] = M_point[5,0] = -1.*y*m
        M_point[1,3] = M_point[3,1] = -1.*z*m
        M_point[2,3] = M_point[3,2] = y*m
        M_point[2,4] = M_point[4,2] = -1.*x*m
        M_point[1,5] = M_point[5,1] = x*m
        M_point[3,3] = Jxx + m*(y**2+z**2)
        M_point[4,4] = Jyy + m*(x**2+z**2)
        M_point[5,5] = Jzz + m*(x**2+y**2)
        M_point[3,4] = M_point[4,3] = Jxy - (m*x*y)
        M_point[3,5] = M_point[5,3] = Jxz - (m*x*z)
        M_point[4,5] = M_point[5,4] = Jyz - (m*y*z)
        
        outputs['tip_mass_mat'] = M_point

    def compute_partials(self, inputs, partials):
        m = inputs['tip_mass']
        Ref2COG = inputs['ref_to_cog']
        J_G = inputs['tip_inertia']

        M_point = np.zeros((6,6))
        x,y,z = Ref2COG
        Jxx,Jxy,Jxz = J_G[0,:]
        _  ,Jyy,Jyz = J_G[1,:]
        _  ,_  ,Jzz = J_G[2,:]

        M_point[0,0] = M_point[1,1] = M_point[2,2] = m
        M_point[0,4] = M_point[4,0] = z*m
        M_point[0,5] = M_point[5,0] = -1.*y*m
        M_point[1,3] = M_point[3,1] = -1.*z*m
        M_point[2,3] = M_point[3,2] = y*m
        M_point[2,4] = M_point[4,2] = -1.*x*m
        M_point[1,5] = M_point[5,1] = x*m
        M_point[3,3] = Jxx + m*(y**2+z**2)
        M_point[4,4] = Jyy + m*(x**2+z**2)
        M_point[5,5] = Jzz + m*(x**2+y**2)
        M_point[3,4] = M_point[4,3] = Jxy - (m*x*y)
        M_point[3,5] = M_point[5,3] = Jxz - (m*x*z)
        M_point[4,5] = M_point[5,4] = Jyz - (m*y*z)
        
        ##TODO define these partials
        dM_dm = np.zeros((6,6))
        dM_dcog = np.zeros((6,6,3))
        dM_diner = np.zeros((6,6,3,3))

        # -- Tip Mass
        dM_dm[0,0] = dM_dm[1,1] = dM_dm[2,2] = 1.
        dM_dm[0,4] = dM_dm[4,0] = z
        dM_dm[0,5] = dM_dm[5,0] = -1.*y
        dM_dm[1,3] = dM_dm[3,1] = -1.*z
        dM_dm[2,3] = dM_dm[3,2] = y
        dM_dm[2,4] = dM_dm[4,2] = -1.*x
        dM_dm[1,5] = dM_dm[5,1] = x
        dM_dm[3,3] = (y**2+z**2)
        dM_dm[4,4] = (x**2+z**2)
        dM_dm[5,5] = (x**2+y**2)
        dM_dm[3,4] = dM_dm[4,3] = -1.*x*y
        dM_dm[3,5] = dM_dm[5,3] = -1.*x*z
        dM_dm[4,5] = dM_dm[5,4] = -1.*y*z

        # -- CoG
        dM_dcog[0,4,2] = dM_dcog[4,0,2] = m
        dM_dcog[0,5,1] = dM_dcog[5,0,1] = -1.*m
        dM_dcog[1,3,2] = dM_dcog[3,1,2] = -1.*m
        dM_dcog[2,3,1] = dM_dcog[3,2,1] = m
        dM_dcog[2,4,0] = dM_dcog[4,2,0] = -1.*m
        dM_dcog[1,5,0] = dM_dcog[5,1,0] = m
        
        dM_dcog[3,3,1] = m*(2.*y)
        dM_dcog[3,3,2] = m*(2.*z)
        dM_dcog[4,4,0] = m*(2.*x)
        dM_dcog[4,4,2] = m*(2.*z)
        dM_dcog[5,5,0] = m*(2.*x)
        dM_dcog[5,5,1] = m*(2.*y)
        
        dM_dcog[3,4,0] = dM_dcog[4,3,0] = -1. * (m*y)
        dM_dcog[3,4,1] = dM_dcog[4,3,1] = -1. * (m*x)
        dM_dcog[3,5,0] = dM_dcog[5,3,0] = -1. * (m*z)
        dM_dcog[3,5,2] = dM_dcog[5,3,2] = -1. * (m*x)
        dM_dcog[4,5,1] = dM_dcog[5,4,1] = -1. * (m*z)
        dM_dcog[4,5,2] = dM_dcog[5,4,2] = -1. * (m*y)

        # -- Inertia
        dM_diner[3,3,0,0] = 1.
        dM_diner[4,4,1,1] = 1.
        dM_diner[5,5,2,2] = 1.
        dM_diner[3,4,0,1] = dM_diner[4,3,0,1] = 1.
        dM_diner[3,5,0,2] = dM_diner[5,3,0,2] = 1.
        dM_diner[4,5,1,2] = dM_diner[5,4,1,2] = 1.

        partials['tip_mass_mat', 'tip_mass'] = dM_dm.flatten()
        partials['tip_mass_mat', 'ref_to_cog'] = np.reshape(dM_dcog,(6*6,3))
        partials['tip_mass_mat', 'tip_inertia'] = np.reshape(dM_diner,(6*6,3*3))