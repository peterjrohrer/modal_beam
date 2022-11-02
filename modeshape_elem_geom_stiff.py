import numpy as np
import scipy.linalg
import myconstants as myconst

import openmdao.api as om

class ModeshapeElemGeomStiff(om.ExplicitComponent):
    
    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nElem = self.nodal_data['nElem']
        nNode = self.nodal_data['nNode']
        
        self.add_input('L_beam', val=np.zeros(nElem), units='m')
        self.add_input('P_beam', val=np.zeros(nElem), units='N')
        self.add_input('dir_cosines', val=np.zeros((nElem,3,3)))     

        self.add_output('kel_geom', val=np.zeros((nElem,12,12)), units='N/m')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        nElem = self.nodal_data['nElem']
        nNode = self.nodal_data['nNode']

        L_beam = inputs['L_beam']
        P_beam = inputs['P_beam']

        kel_geom = np.zeros((nElem, 12, 12))
        Keg = np.zeros((nElem, 12, 12))
        for i in range(nElem):
            # See Cook FEA book, page 643
            kel_geom[i, 1, 1] = kel_geom[i, 2, 2] = kel_geom[i, 7, 7] = kel_geom[i, 8, 8] = 6. / (5. * L_beam[i])
            kel_geom[i, 1, 7] = kel_geom[i, 7, 1] = kel_geom[i, 2, 8] = kel_geom[i, 8, 2] = -6. / (5. * L_beam[i])
            kel_geom[i, 1, 5] = kel_geom[i, 5, 1] = kel_geom[i, 1, 11] = kel_geom[i, 11, 1] = 1. / 10.
            kel_geom[i, 4, 8] = kel_geom[i, 8, 4] = kel_geom[i, 8, 10] = kel_geom[i, 10, 8] = 1. / 10.
            kel_geom[i, 2, 5] = kel_geom[i, 5, 2] = kel_geom[i, 2, 10] = kel_geom[i, 10, 2] = -1. / 10.
            kel_geom[i, 5, 7] = kel_geom[i, 7, 5] = kel_geom[i, 7, 11] = kel_geom[i, 11, 7] = -1. / 10.
            kel_geom[i, 4, 4] = kel_geom[i, 5, 5] = kel_geom[i, 10, 10] = kel_geom[i, 11, 11] = 2. * L_beam[i] / 15.
            kel_geom[i, 4, 10] = kel_geom[i, 10, 4] = kel_geom[i, 5, 11] = kel_geom[i, 11, 5] = -1. * L_beam[i] / 30.
            
            kel_geom[i,:,:] = P_beam[i] * kel_geom[i,:,:]

            ## Element in global coord
            R = inputs['dir_cosines'][i,:,:]
            RR = scipy.linalg.block_diag(R,R,R,R)
            Keg[i,:,:] = np.transpose(RR).dot(kel_geom[i,:,:]).dot(RR)
        
        outputs['kel_geom'] = Keg


    def compute_partials(self, inputs, partials):
        nElem = self.nodal_data['nElem']
        nNode = self.nodal_data['nNode']

        partials['kel_geom', 'L_beam'] = np.zeros(((nElem*12*12), nElem))
        partials['kel_geom', 'a_beam'] = np.zeros(((nElem*12*12), nElem))
