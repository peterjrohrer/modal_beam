import numpy as np
from openmdao.api import ExplicitComponent


class ModeshapeEigenLHS2(ExplicitComponent):
    # abc

    def initialize(self):
        self.options.declare('nDOF', types=int)

    def setup(self):
        nDOF = self.options['nDOF']

        self.add_input('M_mode', val=np.zeros((nDOF, nDOF)), units='kg')
        self.add_input('K_mode', val=np.zeros((nDOF, nDOF)), units='N/m')
        self.add_input('eig_vectors', val=np.zeros((nDOF, nDOF)))

        self.add_output('lhs', val=np.ones((nDOF, nDOF)))
    
    def setup_partials(self):
        self.declare_partials('lhs', ['M_mode','eig_vectors'], method='fd')

    def compute(self, inputs, outputs):
        nDOF = self.options['nDOF']
        M = inputs['M_mode']
        PHI = inputs['eig_vectors']

        outputs['lhs'] = (PHI.T@(M@PHI)) - np.identity(nDOF)
    
    # def compute_partials(self, inputs, partials):
    #     nDOF = self.options['nDOF']
    #     M = inputs['M_mode']
    #     PHI = inputs['eig_vectors']

    #     # partials['lhs', 'eig_vectors'] = np.matmul(M, PHI) + np.matmul(PHI.T, M)
    #     # partials['lhs', 'M_mode'] = np.matmul(PHI.T, PHI)

    #     partials['lhs', 'eig_vectors'] = np.identity(nDOF*nDOF)
    #     partials['lhs', 'M_mode'] = np.identity(nDOF*nDOF)