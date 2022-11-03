import numpy as np
import myconstants as myconst
import openmdao.api as om

class ModeshapeElemMass(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nElem = self.nodal_data['nElem']

        self.add_input('L_beam', val=np.zeros(nElem), units='m')
        self.add_input('A_beam', val=np.zeros(nElem), units='m**2')
        self.add_input('Ix_beam', val=np.zeros(nElem), units='m**4')
        self.add_input('M_beam', val=np.zeros(nElem), units='kg')

        self.add_output('mel_loc', val=np.zeros((nElem, 12, 12)), units='kg')

    def setup_partials(self):
        nElem = self.nodal_data['nElem']

        Hcols = np.repeat(np.arange(nElem),(12*12))

        self.declare_partials('mel_loc', ['L_beam', 'A_beam', 'Ix_beam', 'M_beam'], rows=np.arange(nElem*12*12), cols=Hcols)

    def compute(self, inputs, outputs):
        nElem = self.nodal_data['nElem']

        L_beam = inputs['L_beam']
        A_beam = inputs['A_beam']
        Ix_beam = inputs['Ix_beam']
        M_beam = inputs['M_beam']

        mel = np.zeros((nElem, 12, 12))
        for i in range(nElem):
            a = L_beam[i] / 2.
            a2 = a ** 2.
            rx = Ix_beam[i] / A_beam[i]
              
            # Construct individual mass matrices for each element
            mel[i, 0, 0] = mel[i, 6, 6] = 70.
            mel[i, 1, 1] = mel[i, 2, 2] = mel[i, 7, 7] = mel[i, 8, 8] = 78.
            mel[i, 3, 3] = mel[i, 9, 9] = 70. * rx
            mel[i, 4, 4] = mel[i, 5, 5] = mel[i, 10, 10] = mel[i, 11, 11] = 8. * a2
            mel[i, 2, 4] = mel[i, 4, 2] = mel[i, 7, 11] = mel[i, 11, 7] = -22. * a
            mel[i, 1, 5] = mel[i, 5, 1] = mel[i, 8, 10] = mel[i, 10, 8] = 22. * a 
            mel[i, 0, 6] = mel[i, 6, 0] = 35. 
            mel[i, 1, 7] = mel[i, 7, 1] = mel[i, 2, 8] = mel[i, 8, 2] = 27. 
            mel[i, 1, 11] = mel[i, 11, 1] = mel[i, 4, 8] = mel[i, 8, 4] = -13. * a
            mel[i, 2, 10] = mel[i, 10, 2] = mel[i, 5, 7] = mel[i, 7, 5] = 13. * a
            mel[i, 3, 9] = mel[i, 9, 3] = 35. * rx
            mel[i, 4, 10] = mel[i, 10, 4] = mel[i, 5, 11] = mel[i, 11, 5] = -6. * a2
            # Multiply each mass matrix by scalar        
            mel[i,:,:] = mel[i,:,:] * M_beam[i] * a / 105.

        outputs['mel_loc'] = mel
        
    def compute_partials(self, inputs, partials):
        nElem = self.nodal_data['nElem']      
        nPart = 12 * 12 * nElem

        partials['mel_loc', 'L_beam'] = np.zeros(nPart)
        partials['mel_loc', 'A_beam'] = np.zeros(nPart)
        partials['mel_loc', 'Ix_beam'] = np.zeros(nPart)
        partials['mel_loc', 'M_beam'] = np.zeros(nPart)

        L_beam = inputs['L_beam']
        A_beam = inputs['A_beam']
        Ix_beam = inputs['Ix_beam']
        M_beam = inputs['M_beam']

        mel = np.zeros((nElem, 12, 12))
        dmel_dL = np.zeros((nElem, 12, 12))
        dmel_dA = np.zeros((nElem, 12, 12))
        dmel_dIx = np.zeros((nElem, 12, 12))
        dmel_dM = np.zeros((nElem, 12, 12))

        for i in range(nElem):
            a = L_beam[i] / 2.
            a2 = a ** 2.
            rx = Ix_beam[i] / A_beam[i]

            da_dL = L_beam[i] / 2.
            da2_dL = (3. * L_beam[i] * L_beam[i]) / 8.
            
            drx_dA = -1. * Ix_beam[i] / (A_beam[i] * A_beam[i])
            drx_dIx = 1./ A_beam[i]
              
            # Construct individual mass matrices for each element
            mel[i, 0, 0] = mel[i, 6, 6] = 70.
            mel[i, 1, 1] = mel[i, 2, 2] = mel[i, 7, 7] = mel[i, 8, 8] = 78.
            mel[i, 3, 3] = mel[i, 9, 9] = 70. * rx
            mel[i, 4, 4] = mel[i, 5, 5] = mel[i, 10, 10] = mel[i, 11, 11] = 8. * a2
            mel[i, 2, 4] = mel[i, 4, 2] = mel[i, 7, 11] = mel[i, 11, 7] = -22. * a
            mel[i, 1, 5] = mel[i, 5, 1] = mel[i, 8, 10] = mel[i, 10, 8] = 22. * a 
            mel[i, 0, 6] = mel[i, 6, 0] = 35. 
            mel[i, 1, 7] = mel[i, 7, 1] = mel[i, 2, 8] = mel[i, 8, 2] = 27. 
            mel[i, 1, 11] = mel[i, 11, 1] = mel[i, 4, 8] = mel[i, 8, 4] = -13. * a
            mel[i, 2, 10] = mel[i, 10, 2] = mel[i, 5, 7] = mel[i, 7, 5] = 13. * a
            mel[i, 3, 9] = mel[i, 9, 3] = 35. * rx
            mel[i, 4, 10] = mel[i, 10, 4] = mel[i, 5, 11] = mel[i, 11, 5] = -6. * a2

            # Derivatives wrt L, maintain scalars because they get multiplied by a again
            dmel_dL[i, 0, 0] = dmel_dL[i, 6, 6] = 70. * 0.5
            dmel_dL[i, 1, 1] = dmel_dL[i, 2, 2] = dmel_dL[i, 7, 7] = dmel_dL[i, 8, 8] = 78. * 0.5
            dmel_dL[i, 3, 3] = dmel_dL[i, 9, 9] = 70. * rx * 0.5
            dmel_dL[i, 4, 4] = dmel_dL[i, 5, 5] = dmel_dL[i, 10, 10] = dmel_dL[i, 11, 11] = 8. * da2_dL
            dmel_dL[i, 2, 4] = dmel_dL[i, 4, 2] = dmel_dL[i, 7, 11] = dmel_dL[i, 11, 7] = -22. * da_dL
            dmel_dL[i, 1, 5] = dmel_dL[i, 5, 1] = dmel_dL[i, 8, 10] = dmel_dL[i, 10, 8] = 22. * da_dL
            dmel_dL[i, 0, 6] = dmel_dL[i, 6, 0] = 35. * 0.5
            dmel_dL[i, 1, 7] = dmel_dL[i, 7, 1] = dmel_dL[i, 2, 8] = dmel_dL[i, 8, 2] = 27. * 0.5
            dmel_dL[i, 1, 11] = dmel_dL[i, 11, 1] = dmel_dL[i, 4, 8] = dmel_dL[i, 8, 4] = -13. * da_dL
            dmel_dL[i, 2, 10] = dmel_dL[i, 10, 2] = dmel_dL[i, 5, 7] = dmel_dL[i, 7, 5] = 13. * da_dL
            dmel_dL[i, 3, 9] = dmel_dL[i, 9, 3] = 35. * rx * 0.5
            dmel_dL[i, 4, 10] = dmel_dL[i, 10, 4] = dmel_dL[i, 5, 11] = dmel_dL[i, 11, 5] = -6. * da2_dL

            # Derivatives wrt A and I
            dmel_dA[i, 3, 3] = dmel_dA[i, 9, 9] = 70. * drx_dA
            dmel_dA[i, 3, 9] = dmel_dA[i, 9, 3] = 35. * drx_dA

            dmel_dIx[i, 3, 3] = dmel_dIx[i, 9, 9] = 70. * drx_dIx
            dmel_dIx[i, 3, 9] = dmel_dIx[i, 9, 3] = 35. * drx_dIx

            # Multiply each mass matrix by scalar        
            dmel_dL[i,:,:] = dmel_dL[i,:,:] * M_beam[i] / 105.
            dmel_dA[i,:,:] = dmel_dA[i,:,:] * M_beam[i] * a / 105.
            dmel_dIx[i,:,:] = dmel_dIx[i,:,:] * M_beam[i] * a / 105.
            dmel_dM[i,:,:] = mel[i,:,:] * a / 105.

            pt0 = i * 12 * 12
            pt1 = (i+1) * 12 * 12
            partials['mel_loc', 'L_beam'][pt0:pt1] += dmel_dL[i,:,:].flatten()
            partials['mel_loc', 'A_beam'][pt0:pt1] += dmel_dA[i,:,:].flatten()
            partials['mel_loc', 'Ix_beam'][pt0:pt1] += dmel_dIx[i,:,:].flatten()
            partials['mel_loc', 'M_beam'][pt0:pt1] += dmel_dM[i,:,:].flatten()
