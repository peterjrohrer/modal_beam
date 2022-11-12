import numpy as np
from openmdao.api import ExplicitComponent

class ModeshapeMInv(ExplicitComponent):
    # Invert mass matrix for modeshape eigenmatrix construction

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nDOF = self.nodal_data['nDOF_r']

        self.add_input('Mr_glob', val=np.zeros((nDOF, nDOF)), units='kg')

        self.add_output('Mr_glob_inv', val=np.zeros((nDOF, nDOF)), units='1/kg')

    def setup_partials(self):
        self.declare_partials('Mr_glob_inv', 'Mr_glob')

    def compute(self, inputs, outputs):
        outputs['Mr_glob_inv'] = np.linalg.inv(inputs['Mr_glob'])

    def compute_partials(self, inputs, partials):
        M = inputs['Mr_glob']

        partials['Mr_glob_inv', 'Mr_glob'] = np.kron(np.linalg.inv(M), -1. * np.linalg.inv(M).T)
