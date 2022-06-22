import numpy as np

from openmdao.api import ExplicitComponent


class ModeshapeMInv(ExplicitComponent):
    # Invert mass matrix for modeshape eigenmatrix construction

    def initialize(self):
        self.options.declare('nNode', types=int)
        self.options.declare('nElem', types=int)
        self.options.declare('nDOF', types=int)

    def setup(self):
        nNode = self.options['nNode']
        nElem = self.options['nElem']
        nDOF = self.options['nDOF']

        self.add_input('M_mode', val=np.zeros((nDOF, nDOF)), units='kg')

        self.add_output('M_mode_inv', val=np.zeros((nDOF, nDOF)), units='1/kg')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        outputs['M_mode_inv'] = np.linalg.inv(inputs['M_mode'])

    def compute_partials(self, inputs, partials):
        M = inputs['M_mode']

        partials['M_mode_inv', 'M_mode'] = np.kron(np.linalg.inv(M), -1. * np.linalg.inv(M).T)
