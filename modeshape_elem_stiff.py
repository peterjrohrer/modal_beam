import numpy as np

from openmdao.api import ExplicitComponent


class ModeshapeElemStiff(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('nNode', types=int)
        self.options.declare('nElem', types=int)

    def setup(self):
        nNode = self.options['nNode']
        nElem = self.options['nElem']

        self.add_input('EI_mode_elem', val=np.zeros(nElem), units='N*m**2')
        self.add_input('L_mode_elem', val=np.zeros(nElem), units='m')
        self.add_input('normforce_mode_elem', val=np.zeros(nElem), units='N')
        
        self.add_output('kel', val=np.zeros((nElem, 4, 4)), units='N/m')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        EI = inputs['EI_mode_elem']
        L = inputs['L_mode_elem']
        norm_force = inputs['normforce_mode_elem']

        N_elem = len(L)

        outputs['kel'] = np.zeros((N_elem, 4, 4))
        for i in range(N_elem):
            ke = np.zeros((4, 4))
            kg = np.zeros((4, 4))

            ke[0, 0] = ke[2, 2] = 12. / L[i]**3.
            ke[0, 2] = ke[2, 0] = -12. / L[i]**3.
            ke[0, 1] = ke[1, 0] = ke[0, 3] = ke[3, 0] = 6. / L[i]**2.
            ke[1, 2] = ke[2, 1] = ke[2, 3] = ke[3, 2] = -6. / L[i]**2.
            ke[1, 1] = ke[3, 3] = 4. / L[i]
            ke[1, 3] = ke[3, 1] = 2. / L[i]
            ke = ke * EI[i]

            kg[0, 0] = kg[2, 2] = 6. / (5. * L[i])
            kg[0, 2] = kg[2, 0] = -6. / (5. * L[i])
            kg[0, 1] = kg[1, 0] = kg[0, 3] = kg[3, 0] = 1. / 10.
            kg[1, 2] = kg[2, 1] = kg[2, 3] = kg[3, 2] = -1. / 10.
            kg[1, 1] = kg[3, 3] = 2. * L[i] / 15.
            kg[1, 3] = kg[3, 1] = -L[i] / 30.
            kg = kg * norm_force[i]

            outputs['kel'][i] += ke #+ kg

        a = 1.

    ##TODO Check these partials
    def compute_partials(self, inputs, partials):
        EI = inputs['EI_mode_elem']
        L = inputs['L_mode_elem']
        norm_force = inputs['normforce_mode_elem']

        partials['kel', 'EI_mode_elem'] = np.zeros((352, 22))
        partials['kel', 'L_mode_elem'] = np.zeros((352, 22))
        partials['kel', 'normforce_mode_elem'] = np.zeros((352, 22))
    
        N_elem = len(L)
        N_sparelem = N_elem - len(norm_force)

        for i in range(N_elem):
            dkel_dLe = np.zeros((4, 4))
            ke = np.zeros((4, 4))
            kg = np.zeros((4, 4))

            ke[0, 0] = ke[2, 2] = 12. / L[i]**3.
            ke[0, 2] = ke[2, 0] = -12. / L[i]**3.
            ke[0, 1] = ke[1, 0] = ke[0, 3] = ke[3, 0] = 6. / L[i]**2.
            ke[1, 2] = ke[2, 1] = ke[2, 3] = ke[3, 2] = -6. / L[i]**2.
            ke[1, 1] = ke[3, 3] = 4. / L[i]
            ke[1, 3] = ke[3, 1] = 2. / L[i]
            dkel_dLe[0, 0] = dkel_dLe[2, 2] = -36. / L[i]**4. * EI[i]
            dkel_dLe[0, 2] = dkel_dLe[2, 0] = 36. / L[i]**4. * EI[i]
            dkel_dLe[0, 1] = dkel_dLe[1, 0] = dkel_dLe[0, 3] = dkel_dLe[3, 0] = -12. / L[i]**3. * EI[i]
            dkel_dLe[1, 2] = dkel_dLe[2, 1] = dkel_dLe[2, 3] = dkel_dLe[3, 2] = 12. / L[i]**3. * EI[i]
            dkel_dLe[1, 1] = dkel_dLe[3, 3] = -4. / L[i]**2. * EI[i]
            dkel_dLe[1, 3] = dkel_dLe[3, 1] = -2. / L[i]**2. * EI[i]
            partials['kel', 'EI_mode_elem'][16 * i:16 * i + 16, i] += ke.flatten()

            if L[i] < 0.5:
                partials['kel', 'EI_mode_elem'][16 * i:16 * i + 16, i] = partials['kel', 'EI_mode_elem'][16 * i:16 * i + 16, i] / 1000.

            if i >= N_sparelem:
                kg[0, 0] = kg[2, 2] = 6. / (5. * L[i])
                kg[0, 2] = kg[2, 0] = -6. / (5. * L[i])
                kg[0, 1] = kg[1, 0] = kg[0, 3] = kg[3, 0] = 1. / 10.
                kg[1, 2] = kg[2, 1] = kg[2, 3] = kg[3, 2] = -1. / 10.
                kg[1, 1] = kg[3, 3] = 2. * L[i] / 15.
                kg[1, 3] = kg[3, 1] = -L[i] / 30.
                dkel_dLe[0, 0] += -6. / (5. * L[i]**2.) * norm_force[i - N_sparelem]
                dkel_dLe[0, 2] += 6. / (5. * L[i]**2.) * norm_force[i - N_sparelem]
                dkel_dLe[1, 1] += 2. / 15. * norm_force[i - N_sparelem]
                dkel_dLe[1, 3] += -1. / 30. * norm_force[i - N_sparelem]
                dkel_dLe[2, 2] += -6. / (5. * L[i]**2.) * norm_force[i - N_sparelem]
                dkel_dLe[2, 0] += 6. / (5. * L[i]**2.) * norm_force[i - N_sparelem]
                dkel_dLe[3, 3] += 2. / 15. * norm_force[i - N_sparelem]
                dkel_dLe[3, 1] += -1. / 30. * norm_force[i - N_sparelem]
                partials['kel', 'normforce_mode_elem'][16 * i:16 * i + 16, i - N_sparelem] += kg.flatten()

            partials['kel', 'L_mode_elem'][16 * i:16 * i + 16, i] += dkel_dLe.flatten()

            if L[i] < 0.5:
                partials['kel', 'L_mode_elem'][16 * i:16 * i + 16, i] = partials['kel', 'L_mode_elem'][16 * i:16 * i + 16, i] / 1000.

