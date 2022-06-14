import numpy as np
import myconstants as myconst
from openmdao.api import ExplicitComponent


class ModeshapeElemMass(ExplicitComponent):

    def initialize(self):
        self.options.declare('nNode', types=int)
        self.options.declare('nElem', types=int)

    def setup(self):
        nNode = self.options['nNode']        
        nElem = self.options['nElem']

        self.add_input('L_beam', val=np.zeros(nElem), units='m')
        self.add_input('M_beam', val=np.zeros(nElem), units='kg')
        self.add_input('L_mode_elem', val=np.zeros(nElem), units='m')

        self.add_output('mel', val=np.zeros((nElem, 4, 4)), units='kg')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        L_beam = inputs['L_beam']
        M_beam = inputs['M_beam']
        L = inputs['L_mode_elem']

        N_beamelem = len(M_beam)
        N_elem = N_beamelem

        m = np.zeros(N_elem)  # kg/m

        for i in range(N_beamelem):
            m[i] = M_beam[i] / L_beam[i]

        outputs['mel'] = np.zeros((N_elem, 4, 4))
        
        for i in range(N_elem):
            # Construct individiual mass matrices for each element
            outputs['mel'][i, 0, 0] = outputs['mel'][i, 2, 2] = 156.
            outputs['mel'][i, 1, 1] = outputs['mel'][i, 3, 3] = 4. * L[i]**2.
            outputs['mel'][i, 0, 1] = outputs['mel'][i, 1, 0] = 22. * L[i]
            outputs['mel'][i, 2, 3] = outputs['mel'][i, 3, 2] = -22. * L[i]
            outputs['mel'][i, 0, 2] = outputs['mel'][i, 2, 0] = 54.
            outputs['mel'][i, 1, 2] = outputs['mel'][i, 2, 1] = 13. * L[i]
            outputs['mel'][i, 0, 3] = outputs['mel'][i, 3, 0] = -13. * L[i]
            outputs['mel'][i, 1, 3] = outputs['mel'][i, 3, 1] = -3. * L[i]**2.
            # Multiply each mass matrix by scalar
            outputs['mel'][i] = outputs['mel'][i] * m[i] * L[i] / 420.

    ##TODO Check these partials!
    def compute_partials(self, inputs, partials):
        L_beam = inputs['L_beam']
        M_beam = inputs['M_beam']
        L = inputs['L_mode_elem']

        partials['mel', 'D_spar'] = np.zeros((352, 10))
        partials['mel', 'L_spar'] = np.zeros((352, 10))
        partials['mel', 'M_spar'] = np.zeros((352, 10))
        partials['mel', 'L_beam'] = np.zeros((352, 10))
        partials['mel', 'M_beam'] = np.zeros((352, 10))
        partials['mel', 'spar_draft'] = np.zeros(352)
        partials['mel', 'M_ball_elem'] = np.zeros((352, 10))
        partials['mel', 'L_ball_elem'] = np.zeros((352, 10))
        partials['mel', 'L_ball'] = np.zeros(352)
        partials['mel', 'z_sparnode'] = np.zeros((352, 13))
        partials['mel', 'L_mode_elem'] = np.zeros((352, 22))

        N_beamelem = len(M_beam)
        N_elem = N_beamelem

        dm_dLt = np.zeros((N_elem, len(L_beam)))
        dm_dMt = np.zeros((N_elem, len(M_beam)))
        
        m = np.zeros(N_elem)

        for i in range(N_beamelem):
            m[i] = M_beam[i] / L_beam[i]
            dm_dLt[i, i] += -M_beam[i] / L_beam[i]**2.
            dm_dMt[i, i] += 1. / L_beam[i]

        dmel_dm = np.zeros((N_elem, 4, 4))
        dmel_dLe = np.zeros((N_elem, 4, 4))
        for i in range(N_elem):
            dmel_dm[i, 0, 0] = dmel_dm[i, 2, 2] = 156.
            dmel_dm[i, 1, 1] = dmel_dm[i, 3, 3] = 4. * L[i]**2.
            dmel_dm[i, 0, 1] = dmel_dm[i, 1, 0] = 22. * L[i]
            dmel_dm[i, 2, 3] = dmel_dm[i, 3, 2] = -22. * L[i]
            dmel_dm[i, 0, 2] = dmel_dm[i, 2, 0] = 54.
            dmel_dm[i, 1, 2] = dmel_dm[i, 2, 1] = 13. * L[i]
            dmel_dm[i, 0, 3] = dmel_dm[i, 3, 0] = -13. * L[i]
            dmel_dm[i, 1, 3] = dmel_dm[i, 3, 1] = -3. * L[i]**2.
            dmel_dLe[i, 0, 0] = dmel_dLe[i, 2, 2] = 156. * m[i] / 420.
            dmel_dLe[i, 1, 1] = dmel_dLe[i, 3, 3] = 12. * L[i]**2. * m[i] / 420.
            dmel_dLe[i, 0, 1] = dmel_dLe[i, 1, 0] = 44. * m[i] * L[i] / 420.
            dmel_dLe[i, 2, 3] = dmel_dLe[i, 3, 2] = -44. * L[i] * m[i] / 420.
            dmel_dLe[i, 0, 2] = dmel_dLe[i, 2, 0] = 54. * m[i] / 420.
            dmel_dLe[i, 1, 2] = dmel_dLe[i, 2, 1] = 26. * m[i] * L[i] / 420.
            dmel_dLe[i, 0, 3] = dmel_dLe[i, 3, 0] = -26. * m[i] * L[i] / 420.
            dmel_dLe[i, 1, 3] = dmel_dLe[i, 3, 1] = -9. * L[i]**2. * m[i] / 420.
            dmel_dm[i] = dmel_dm[i] * L[i] / 420.

        for i in range(len(L_beam)):
            dmel_dLt = []
            dmel_dMt = []
            for j in range(N_elem):
                for k in range(4):
                    for l in range(4):
                        dmel_dLt.append(dmel_dm[j, k, l] * dm_dLt[j, i])
                        dmel_dMt.append(dmel_dm[j, k, l] * dm_dMt[j, i])

            partials['mel', 'L_beam'][:, i] = np.array(dmel_dLt)
            partials['mel', 'M_beam'][:, i] = np.array(dmel_dMt)

        for i in range(N_elem):
            partials['mel', 'L_mode_elem'][16 * i:16 * i + 16, i] = dmel_dLe[i].flatten()
