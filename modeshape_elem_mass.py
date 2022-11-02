import numpy as np
import scipy.linalg
import myconstants as myconst
import openmdao.api as om

class ModeshapeElemMass(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nElem = self.nodal_data['nElem']
        nNode = self.nodal_data['nNode']

        self.add_input('L_beam', val=np.zeros(nElem), units='m')
        self.add_input('A_beam', val=np.zeros(nElem), units='m**2')
        self.add_input('Ix_beam', val=np.zeros(nElem), units='m**4')
        self.add_input('M_beam', val=np.zeros(nElem), units='kg')
        self.add_input('dir_cosines', val=np.zeros((nElem,3,3)))     

        self.add_output('mel', val=np.zeros((nElem, 12, 12)), units='kg')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        nElem = self.nodal_data['nElem']
        nNode = self.nodal_data['nNode']

        L_beam = inputs['L_beam']
        A_beam = inputs['A_beam']
        Ix_beam = inputs['Ix_beam']
        M_beam = inputs['M_beam']

        mel = np.zeros((nElem, 12, 12))
        Me = np.zeros((nElem, 12, 12))
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

            ## Element in global coord
            R = inputs['dir_cosines'][i,:,:]
            RR = scipy.linalg.block_diag(R,R,R,R)
            Me[i,:,:] = np.transpose(RR).dot(mel[i,:,:]).dot(RR)

        outputs['mel'] = Me
        
    ##TODO Check these partials!
    def compute_partials(self, inputs, partials):
        nElem = self.nodal_data['nElem']
        nNode = self.nodal_data['nNode']

        L_beam = inputs['L_beam']
        M_beam = inputs['M_beam']
        L = inputs['L_mode_elem']
        
        N_mel_part = 12 * 12 * nElem

        partials['mel', 'L_beam'] = np.zeros((N_mel_part, nElem))
        partials['mel', 'M_beam'] = np.zeros((N_mel_part, nElem))
        partials['mel', 'L_mode_elem'] = np.zeros((N_mel_part, nElem))

        # dm_dLt = np.zeros((nElem, len(L_beam)))
        # dm_dMt = np.zeros((nElem, len(M_beam)))

        # m = np.zeros(nElem)

        # for i in range(nElem):
        #     m[i] = M_beam[i] / L_beam[i]
        #     dm_dLt[i, i] += -M_beam[i] / L_beam[i]**2.
        #     dm_dMt[i, i] += 1. / L_beam[i]

        # dmel_dm = np.zeros((nElem, 4, 4))
        # dmel_dLe = np.zeros((nElem, 4, 4))
        # for i in range(nElem):
        #     dmel_dm[i, 0, 0] = dmel_dm[i, 2, 2] = 156.
        #     dmel_dm[i, 1, 1] = dmel_dm[i, 3, 3] = 4. * L[i]**2.
        #     dmel_dm[i, 0, 1] = dmel_dm[i, 1, 0] = 22. * L[i]
        #     dmel_dm[i, 2, 3] = dmel_dm[i, 3, 2] = -22. * L[i]
        #     dmel_dm[i, 0, 2] = dmel_dm[i, 2, 0] = 54.
        #     dmel_dm[i, 1, 2] = dmel_dm[i, 2, 1] = 13. * L[i]
        #     dmel_dm[i, 0, 3] = dmel_dm[i, 3, 0] = -13. * L[i]
        #     dmel_dm[i, 1, 3] = dmel_dm[i, 3, 1] = -3. * L[i]**2.
        #     dmel_dLe[i, 0, 0] = dmel_dLe[i, 2, 2] = 156. * m[i] / 420.
        #     dmel_dLe[i, 1, 1] = dmel_dLe[i, 3, 3] = 12. * L[i]**2. * m[i] / 420.
        #     dmel_dLe[i, 0, 1] = dmel_dLe[i, 1, 0] = 44. * m[i] * L[i] / 420.
        #     dmel_dLe[i, 2, 3] = dmel_dLe[i, 3, 2] = -44. * L[i] * m[i] / 420.
        #     dmel_dLe[i, 0, 2] = dmel_dLe[i, 2, 0] = 54. * m[i] / 420.
        #     dmel_dLe[i, 1, 2] = dmel_dLe[i, 2, 1] = 26. * m[i] * L[i] / 420.
        #     dmel_dLe[i, 0, 3] = dmel_dLe[i, 3, 0] = -26. * m[i] * L[i] / 420.
        #     dmel_dLe[i, 1, 3] = dmel_dLe[i, 3, 1] = -9. * L[i]**2. * m[i] / 420.
        #     dmel_dm[i] = dmel_dm[i] * L[i] / 420.

        # for i in range(len(L_beam)):
        #     dmel_dLt = []
        #     dmel_dMt = []
        #     for j in range(nElem):
        #         for k in range(4):
        #             for l in range(4):
        #                 dmel_dLt.append(dmel_dm[j, k, l] * dm_dLt[j, i])
        #                 dmel_dMt.append(dmel_dm[j, k, l] * dm_dMt[j, i])

        #     partials['mel', 'L_beam'][:, i] = np.array(dmel_dLt)
        #     partials['mel', 'M_beam'][:, i] = np.array(dmel_dMt)

        # for i in range(nElem):
        #     partials['mel', 'L_mode_elem'][16 * i:16 * i + 16, i] = dmel_dLe[i].flatten()
