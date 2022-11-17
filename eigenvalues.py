import numpy as np
import scipy.linalg
import openmdao.api as om

class Eigenvals(om.ExplicitComponent):
    # Eigenvalues from mass-normed eigenvectors for robustness

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nDOF_r = self.nodal_data['nDOF_r']
        nDOF_tot = self.nodal_data['nDOF_tot']

        self.add_input('Kr_glob', val=np.ones((nDOF_r, nDOF_r)))
        self.add_input('Q_mass_norm', val=np.ones((nDOF_r, nDOF_r)))

        self.add_output('eigenvals_raw', val=np.zeros((nDOF_r, nDOF_r)), units='1/s')

    def compute(self, inputs, outputs):
        nDOF = self.nodal_data['nDOF_r']
        K = inputs['Kr_glob']
        Q = inputs['Q_mass_norm']
                
        Lambda=np.dot(Q.T,K).dot(Q)

        outputs['eigenvals_raw'] = Lambda