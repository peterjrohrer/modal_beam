import numpy as np
from openmdao.api import ExplicitComponent

class ModeshapeEigmatrix(ExplicitComponent):
    # Assemble modeshape eigenmatrix

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nDOF = self.nodal_data['nDOF_r']

        self.add_input('Mr_glob_inv', val=np.zeros((nDOF, nDOF)), units='1/kg')
        self.add_input('Kr_glob', val=np.zeros((nDOF, nDOF)), units='N/m')

        self.add_output('Ar_eig', val=np.zeros((nDOF, nDOF)))

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        K = inputs['Kr_glob']
        M_inv = inputs['Mr_glob_inv']

        outputs['Ar_eig'] = np.matmul(M_inv, K)

    def compute_partials(self, inputs, partials):
        K = inputs['Kr_glob']
        M_inv = inputs['Mr_glob_inv']

        N_elem = len(K)

        partials['Ar_eig', 'Kr_glob'] = np.kron(M_inv, np.identity(N_elem))
        partials['Ar_eig', 'Mr_glob_inv'] = np.kron(np.identity(N_elem), K.T)
