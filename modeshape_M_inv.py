import numpy as np
import scipy.linalg
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
        M = inputs['Mr_glob']
        M_inv = scipy.linalg.inv(M)
        ##TODO Check whether this is justified!
        # # Remove noise from inversion
        # M_inv[M==0.0] = 0.0

        outputs['Mr_glob_inv'] = M_inv        

    def compute_partials(self, inputs, partials):
        M = inputs['Mr_glob']
        M_inv = scipy.linalg.inv(M)
        # # Remove noise from inversion
        # M_inv[M==0.0] = 0.0

        partials['Mr_glob_inv', 'Mr_glob'] = np.kron(M_inv, -1. * M_inv.T)