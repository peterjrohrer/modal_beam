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
import myconstants as myconst
import openmdao.api as om

class ModeshapeElemMatStiff(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nElem = self.nodal_data['nElem']
    
        self.add_input('L_beam', val=np.zeros(nElem), units='m')
        self.add_input('A_beam', val=np.zeros(nElem), units='m**2')
        self.add_input('Iy_beam', val=np.zeros(nElem), units='m**4')
        self.add_input('Iz_beam', val=np.zeros(nElem), units='m**4')

        self.add_output('kel_mat', val=np.zeros((nElem,12,12)), units='N/m')

    def setup_partials(self):
        nElem = self.nodal_data['nElem']

        Hcols = np.repeat(np.arange(nElem),(12*12))

        self.declare_partials('kel_mat', ['L_beam', 'A_beam', 'Iy_beam', 'Iz_beam'], rows=np.arange(nElem*12*12), cols=Hcols)

    def compute(self, inputs, outputs):
        nElem = self.nodal_data['nElem']

        L_beam = inputs['L_beam']
        EA = inputs['A_beam'] * myconst.E_STL
        EIy = inputs['Iy_beam'] * myconst.E_STL
        EIz = inputs['Iz_beam'] * myconst.E_STL
        Kv = EIy/(myconst.E_STL*10) # check this!

        kel_mat = np.zeros((nElem, 12, 12))
        for i in range(nElem):
            a = EA[i] / L_beam[i]
            b = 12. * EIz[i] / (L_beam[i] ** 3.)
            c = 6. * EIz[i] / (L_beam[i] ** 2.)
            d = 12. * EIy[i] / (L_beam[i] ** 3.)
            e = 6. * EIy[i] / (L_beam[i] ** 2.)
            f = myconst.G_STL * Kv[i] / L_beam[i]
            g = 2. * EIy[i] / L_beam[i]
            h = 2. * EIz[i] / L_beam[i]

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
        
        outputs['kel_mat'] = kel_mat

    def compute_partials(self, inputs, partials):
        nElem = self.nodal_data['nElem']      
        nPart = 12 * 12 * nElem

        partials['kel_mat', 'L_beam'] = np.zeros(nPart)
        partials['kel_mat', 'A_beam'] = np.zeros(nPart)
        partials['kel_mat', 'Iy_beam'] = np.zeros(nPart)
        partials['kel_mat', 'Iz_beam'] = np.zeros(nPart)

        L_beam = inputs['L_beam']
        EA = inputs['A_beam'] * myconst.E_STL
        EIy = inputs['Iy_beam'] * myconst.E_STL
        EIz = inputs['Iz_beam'] * myconst.E_STL
        Kv  = EIy/(myconst.E_STL*10) # check this!

        dkel_dL = np.zeros((nElem, 12, 12))
        dkel_dA = np.zeros((nElem, 12, 12))
        dkel_dIy = np.zeros((nElem, 12, 12))
        dkel_dIz = np.zeros((nElem, 12, 12))

        for i in range(nElem):
            a = EA[i] / L_beam[i]
            b = 12. * EIz[i] / (L_beam[i] ** 3.)
            c = 6. * EIz[i] / (L_beam[i] ** 2.)
            d = 12. * EIy[i] / (L_beam[i] ** 3.)
            e = 6. * EIy[i] / (L_beam[i] ** 2.)
            f = myconst.G_STL * Kv[i] / L_beam[i]
            g = 2. * EIy[i] / L_beam[i]
            h = 2. * EIz[i] / L_beam[i]

            da_dL = -1. * EA[i] / (L_beam[i] ** 2.)
            db_dL = -36. * EIz[i] / (L_beam[i] ** 4.)
            dc_dL = -12. * EIz[i] / (L_beam[i] ** 3.)
            dd_dL = -36. * EIy[i] / (L_beam[i] ** 4.)
            de_dL = -12. * EIy[i] / (L_beam[i] ** 3.)
            df_dL = -1. * myconst.G_STL * Kv[i] / (L_beam[i] ** 2.)
            dg_dL = -2. * EIy[i] / (L_beam[i] ** 2.)
            dh_dL = -2. * EIz[i] / (L_beam[i] ** 2.)

            da_dA = myconst.E_STL / L_beam[i]

            dd_dIy = 12. * myconst.E_STL / (L_beam[i] ** 3.)
            de_dIy = 6. * myconst.E_STL / (L_beam[i] ** 2.)
            df_dIy = myconst.G_STL * (myconst.E_STL/(myconst.E_STL*10)) / L_beam[i]
            dg_dIy = 2. * myconst.E_STL / L_beam[i]

            db_dIz = 12. * myconst.E_STL / (L_beam[i] ** 3.)
            dc_dIz = 6. * myconst.E_STL / (L_beam[i] ** 2.)
            dh_dIz = 2. * myconst.E_STL / L_beam[i]

            # Derivatives wrt L
            dkel_dL[i,0,0] = dkel_dL[i,6,6] = da_dL
            dkel_dL[i,1,1] = dkel_dL[i,7,7] = db_dL
            dkel_dL[i,2,2] = dkel_dL[i,8,8] = dd_dL
            dkel_dL[i,3,3] = dkel_dL[i,9,9] = df_dL
            dkel_dL[i,4,4] = dkel_dL[i,10,10] = 2.*dg_dL
            dkel_dL[i,5,5] = dkel_dL[i,11,11] = 2.*dh_dL
            dkel_dL[i,1,5] = dkel_dL[i,5,1] = dkel_dL[i,1,11] = dkel_dL[i,11,1] = dc_dL
            dkel_dL[i,2,4] = dkel_dL[i,4,2] = dkel_dL[i,2,10] = dkel_dL[i,10,2] = -1. * de_dL
            dkel_dL[i,7,11] = dkel_dL[i,11,7] = dkel_dL[i,7,5] = dkel_dL[i,5,7] = -1. * dc_dL
            dkel_dL[i,8,10] = dkel_dL[i,10,8] = dkel_dL[i,4,8] = dkel_dL[i,8,4] = de_dL

            dkel_dL[i,6,0] = dkel_dL[i,0,6] = -1. * da_dL
            dkel_dL[i,7,1] = dkel_dL[i,1,7] = -1. * db_dL
            dkel_dL[i,8,2] = dkel_dL[i,2,8] = -1. * dd_dL
            dkel_dL[i,9,3] = dkel_dL[i,3,9] = -1. * df_dL
            dkel_dL[i,10,4] = dkel_dL[i,4,10] = dg_dL
            dkel_dL[i,11,5] = dkel_dL[i,5,11] = dh_dL

            # Derivatives wrt A
            dkel_dA[i,0,0] = dkel_dA[i,6,6] = da_dA
            dkel_dA[i,6,0] = dkel_dA[i,0,6] = -1. * da_dA

            # Derivatives wrt Iy
            dkel_dIy[i,2,2] = dkel_dIy[i,8,8] = dd_dIy
            dkel_dIy[i,3,3] = dkel_dIy[i,9,9] = df_dIy
            dkel_dIy[i,4,4] = dkel_dIy[i,10,10] = 2. * dg_dIy
            dkel_dIy[i,2,4] = dkel_dIy[i,4,2] = dkel_dIy[i,2,10] = dkel_dIy[i,10,2] = -1. * de_dIy
            dkel_dIy[i,8,10] = dkel_dIy[i,10,8] = dkel_dIy[i,4,8] = dkel_dIy[i,8,4] = de_dIy

            dkel_dIy[i,8,2] = dkel_dIy[i,2,8] = -1. * dd_dIy
            dkel_dIy[i,9,3] = dkel_dIy[i,3,9] = -1. * df_dIy
            dkel_dIy[i,10,4] = dkel_dIy[i,4,10] = dg_dIy

            # Derivatives wrt Iz
            dkel_dIz[i,1,1] = dkel_dIz[i,7,7] = db_dIz
            dkel_dIz[i,5,5] = dkel_dIz[i,11,11] = 2. * dh_dIz
            dkel_dIz[i,1,5] = dkel_dIz[i,5,1] = dkel_dIz[i,1,11] = dkel_dIz[i,11,1] = dc_dIz
            dkel_dIz[i,7,11] = dkel_dIz[i,11,7] = dkel_dIz[i,7,5] = dkel_dIz[i,5,7] = -1. * dc_dIz

            dkel_dIz[i,7,1] = dkel_dIz[i,1,7] = -1. * db_dIz
            dkel_dIz[i,11,5] = dkel_dIz[i,5,11] = dh_dIz
            
            pt0 = i * 12 * 12
            pt1 = (i+1) * 12 * 12
            partials['kel_mat', 'L_beam'][pt0:pt1] += dkel_dL[i,:,:].flatten()
            partials['kel_mat', 'A_beam'][pt0:pt1] += dkel_dA[i,:,:].flatten()
            partials['kel_mat', 'Iy_beam'][pt0:pt1] += dkel_dIy[i,:,:].flatten()
            partials['kel_mat', 'Iz_beam'][pt0:pt1] += dkel_dIz[i,:,:].flatten()