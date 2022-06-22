import numpy as np

from openmdao.api import ExplicitComponent


class ModeshapeEigmatrix(ExplicitComponent):
    # Assemble modeshape eigenmatrix

    def initialize(self):
        self.options.declare('nNode', types=int)
        self.options.declare('nElem', types=int)
        self.options.declare('nDOF', types=int)

    def setup(self):
        nNode = self.options['nNode']
        nElem = self.options['nElem']
        nDOF = self.options['nDOF']

        self.add_input('K_mode', val=np.zeros((nDOF, nDOF)), units='N/m')
        self.add_input('M_mode_inv', val=np.zeros((nDOF, nDOF)), units='1/kg')

        self.add_output('A_eig', val=np.zeros((nDOF, nDOF)))

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        K = inputs['K_mode']
        M_inv = inputs['M_mode_inv']

        outputs['A_eig'] = np.matmul(M_inv, K)

    def compute_partials(self, inputs, partials):
        K = inputs['K_mode']
        M_inv = inputs['M_mode_inv']

        N_elem = len(K)

        partials['A_eig', 'K_mode'] = np.kron(M_inv, np.identity(N_elem))
        partials['A_eig', 'M_mode_inv'] = np.kron(np.identity(N_elem), K.T)
