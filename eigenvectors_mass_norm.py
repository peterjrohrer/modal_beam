import numpy as np
import openmdao.api as om

class EigenvecsMassNorm(om.ExplicitComponent):
    # Mass normalize eigenvectors

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nDOF_r = self.nodal_data['nDOF_r']

        self.add_input('Mr_glob', val=np.ones((nDOF_r, nDOF_r)))
        self.add_input('Q_raw', val=np.ones((nDOF_r, nDOF_r)))

        self.add_output('Q_mass_norm', val=np.zeros((nDOF_r, nDOF_r)), units='1/s')

    def compute(self, inputs, outputs):
        nDOF = self.nodal_data['nDOF_r']
        Q = inputs['Q_raw']
        M = inputs['Mr_glob']
                
        for j in range(M.shape[1]):
            q_j = Q[:,j]
            modalmass_j = np.dot(q_j.T,M).dot(q_j)
            Q[:,j]= Q[:,j]/np.sqrt(modalmass_j)

        outputs['Q_mass_norm'] = Q