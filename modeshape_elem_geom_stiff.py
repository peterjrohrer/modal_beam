import numpy as np
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

        self.add_output('kel_geom', val=np.zeros((nElem,12,12)), units='N/m')

    def setup_partials(self):
        nElem = self.nodal_data['nElem']
        Hcols = np.repeat(np.arange(nElem),(12*12))

        self.declare_partials('kel_geom', ['L_beam', 'P_beam'], rows=np.arange(nElem*12*12), cols=Hcols)

    def compute(self, inputs, outputs):
        nElem = self.nodal_data['nElem']

        L_beam = inputs['L_beam']
        P_beam = inputs['P_beam']

        kel_geom = np.zeros((nElem, 12, 12))
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

        outputs['kel_geom'] = kel_geom

    def compute_partials(self, inputs, partials):
        nElem = self.nodal_data['nElem']      
        nPart = 12 * 12 * nElem

        partials['kel_geom', 'L_beam'] = np.zeros(nPart)
        partials['kel_geom', 'P_beam'] = np.zeros(nPart)

        L_beam = inputs['L_beam']
        P_beam = inputs['P_beam']

        kel_geom = np.zeros((nElem, 12, 12))
        dkel_dL = np.zeros((nElem, 12, 12))
        for i in range(nElem):
            # Original
            kel_geom[i, 1, 1] = kel_geom[i, 2, 2] = kel_geom[i, 7, 7] = kel_geom[i, 8, 8] = 6. / (5. * L_beam[i])
            kel_geom[i, 1, 7] = kel_geom[i, 7, 1] = kel_geom[i, 2, 8] = kel_geom[i, 8, 2] = -6. / (5. * L_beam[i])
            kel_geom[i, 1, 5] = kel_geom[i, 5, 1] = kel_geom[i, 1, 11] = kel_geom[i, 11, 1] = 1. / 10.
            kel_geom[i, 4, 8] = kel_geom[i, 8, 4] = kel_geom[i, 8, 10] = kel_geom[i, 10, 8] = 1. / 10.
            kel_geom[i, 2, 5] = kel_geom[i, 5, 2] = kel_geom[i, 2, 10] = kel_geom[i, 10, 2] = -1. / 10.
            kel_geom[i, 5, 7] = kel_geom[i, 7, 5] = kel_geom[i, 7, 11] = kel_geom[i, 11, 7] = -1. / 10.
            kel_geom[i, 4, 4] = kel_geom[i, 5, 5] = kel_geom[i, 10, 10] = kel_geom[i, 11, 11] = 2. * L_beam[i] / 15.
            kel_geom[i, 4, 10] = kel_geom[i, 10, 4] = kel_geom[i, 5, 11] = kel_geom[i, 11, 5] = -1. * L_beam[i] / 30.

            # Derivative wrt L
            dkel_dL[i, 1, 1] = dkel_dL[i, 2, 2] = dkel_dL[i, 7, 7] = dkel_dL[i, 8, 8] = -6. / (5. * L_beam[i] * L_beam[i])
            dkel_dL[i, 1, 7] = dkel_dL[i, 7, 1] = dkel_dL[i, 2, 8] = dkel_dL[i, 8, 2] = 6. / (5. * L_beam[i] * L_beam[i])
            dkel_dL[i, 1, 5] = dkel_dL[i, 5, 1] = dkel_dL[i, 1, 11] = dkel_dL[i, 11, 1] = 1. / 10.
            dkel_dL[i, 4, 8] = dkel_dL[i, 8, 4] = dkel_dL[i, 8, 10] = dkel_dL[i, 10, 8] = 1. / 10.
            dkel_dL[i, 2, 5] = dkel_dL[i, 5, 2] = dkel_dL[i, 2, 10] = dkel_dL[i, 10, 2] = -1. / 10.
            dkel_dL[i, 5, 7] = dkel_dL[i, 7, 5] = dkel_dL[i, 7, 11] = dkel_dL[i, 11, 7] = -1. / 10.
            dkel_dL[i, 4, 4] = dkel_dL[i, 5, 5] = dkel_dL[i, 10, 10] = dkel_dL[i, 11, 11] = 2. / 15.
            dkel_dL[i, 4, 10] = dkel_dL[i, 10, 4] = dkel_dL[i, 5, 11] = dkel_dL[i, 11, 5] = -1. / 30.

            dkel_dL[i,:,:] = P_beam[i] * dkel_dL[i,:,:]
            
            pt0 = i * 12 * 12
            pt1 = (i+1) * 12 * 12
            partials['kel_geom', 'L_beam'][pt0:pt1] += dkel_dL[i,:,:].flatten()
            partials['kel_geom', 'P_beam'][pt0:pt1] += kel_geom[i,:,:].flatten()