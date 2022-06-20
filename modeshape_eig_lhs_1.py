import numpy as np
from openmdao.api import ExplicitComponent


class ModeshapeEigenLHS1(ExplicitComponent):
    # abc

    def initialize(self):
        self.options.declare('nDOF', types=int)

    def setup(self):
        nDOF = self.options['nDOF']

        self.add_input('M_mode', val=np.zeros((nDOF, nDOF)), units='kg')
        self.add_input('K_mode', val=np.zeros((nDOF, nDOF)), units='N/m')
        self.add_input('eig_vectors', val=np.zeros((nDOF, nDOF)))
        self.add_input('eig_vals', val=np.zeros((nDOF, nDOF)))

        self.add_output('lhs', val=np.ones((nDOF, nDOF)))
    
    def setup_partials(self):
        self.declare_partials('lhs', ['M_mode','K_mode','eig_vectors','eig_vals'], method='fd')

    def compute(self, inputs, outputs):
        K = inputs['K_mode']
        M = inputs['M_mode']
        PHI = inputs['eig_vectors']
        LAM = inputs['eig_vals']

        outputs['lhs'] = (M@PHI) - (K@(PHI@LAM))