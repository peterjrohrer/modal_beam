"""
    Calculates local-axis material stiffness matrix of element from inputs

    INPUTS:
        D_elem: (float) outer diameter of element 
        wt_elem: (float) wall thickness of element 
        L_beam[i]: (float) length of element 
        R: Transformation matrix (3x3) from global coord to element coord: x_e = R.x_g
                if provided, element matrix is provided in global coord

    OUTPUTS:
        kel_mat: (12x12) local element material stiffness matrix

"""
import numpy as np
import scipy.linalg
import myconstants as myconst

import openmdao.api as om

class ModeshapeElemMatStiff(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nElem = self.nodal_data['nElem']
        nNode = self.nodal_data['nNode']
    
        self.add_input('L_beam', val=np.zeros(nElem), units='m')
        self.add_input('A_beam', val=np.zeros(nElem), units='m**2')
        self.add_input('Iy_beam', val=np.zeros(nElem), units='m**4')
        self.add_input('dir_cosines', val=np.zeros((nElem,3,3)))     

        self.add_output('kel_mat', val=np.zeros((nElem,12,12)), units='N/m')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        nElem = self.nodal_data['nElem']
        nNode = self.nodal_data['nNode']

        L_beam = inputs['L_beam']
        EA = inputs['A_beam'] * myconst.E_STL
        EIy = inputs['Iy_beam'] * myconst.E_STL
        EIz = EIy
        Kv  = EIy/(myconst.E_STL*10) # check this!

        kel_mat = np.zeros((nElem, 12, 12))
        Kem = np.zeros((nElem, 12, 12))
        for i in range(nElem):
            a = EA / L_beam[i]
            b = 12. * EIz / L_beam[i] ** 3.
            c = 6. * EIz / L_beam[i] ** 2.
            d = 12. * EIy / L_beam[i] ** 3.
            e = 6. * EIy / L_beam[i] ** 2.
            f = myconst.G_STL * Kv / L_beam[i]
            g = 2. * EIy / L_beam[i]
            h = 2. * EIz / L_beam[i]

            kel_mat[i,0,0] = kel_mat[i,6,6] = a
            kel_mat[i,1,1] = kel_mat[i,7,7] = b
            kel_mat[i,2,2] = kel_mat[i,8,8] = d
            kel_mat[i,3,3] = kel_mat[i,9,9] = f
            kel_mat[i,4,4] = kel_mat[i,10,10] = 2.*g
            kel_mat[i,5,5] = kel_mat[i,11,11] = 2.*h
            kel_mat[i,1,5] = kel_mat[i,5,1] = kel_mat[i,1,11] = kel_mat[i,11,1] = c
            kel_mat[i,2,4] = kel_mat[i,4,2] = kel_mat[i,2,10] = kel_mat[i,10,2] = -1. * e
            kel_mat[i,7,11] = kel_mat[i,11,7] = kel_mat[i,7,5] = kel_mat[i,5,7] = -1. * c
            kel_mat[i,8,10] = kel_mat[i,10,8] = kel_mat[i,4,8] = kel_mat[i,8,4] = e

            kel_mat[i,6,0] = kel_mat[i,0,6] = -1. * a
            kel_mat[i,7,1] = kel_mat[i,1,7] = -1. * b
            kel_mat[i,8,2] = kel_mat[i,2,8] = -1. * d
            kel_mat[i,9,3] = kel_mat[i,3,9] = -1. * f
            kel_mat[i,10,4] = kel_mat[i,4,10] = g
            kel_mat[i,11,5] = kel_mat[i,5,11] = h
            
            ## Element in global coord
            R = inputs['dir_cosines'][i,:,:]
            RR = scipy.linalg.block_diag(R,R,R,R)
            Kem[i,:,:] = np.transpose(RR).dot(kel_mat[i,:,:]).dot(RR)
        
        outputs['kel_mat'] = Kem


    def compute_partials(self, inputs, partials):
        nElem = self.nodal_data['nElem']
        nNode = self.nodal_data['nNode']

        partials['kel_mat', 'L_beam'] = np.zeros(((nElem*12*12), nElem))
        partials['kel_mat', 'A_beam'] = np.zeros(((nElem*12*12), nElem))