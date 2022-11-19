import numpy as np
import openmdao.api as om

class EigenvecsMassNorm(om.ExplicitComponent):
    # Mass normalize eigenvectors

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nDOF_r = self.nodal_data['nDOF_r']

        self.add_input('Mr_glob', val=np.ones((nDOF_r, nDOF_r)), units='kg')
        self.add_input('Q_raw', val=np.ones((nDOF_r, nDOF_r)))

        self.add_output('Q_mass_norm', val=np.zeros((nDOF_r, nDOF_r)))

    def setup_partials(self):
        self.declare_partials('Q_mass_norm','Mr_glob')
        self.declare_partials('Q_mass_norm','Q_raw')
        
    def compute(self, inputs, outputs):
        nDOF = self.nodal_data['nDOF_r']
        Q = inputs['Q_raw']
        M = inputs['Mr_glob']
        Q_norm = np.zeros_like(Q)
                
        for j in range(nDOF):
            q_j = Q[:,j]
            modalmass_j = np.dot(q_j.T,M).dot(q_j)
            Q_norm[:,j]= Q[:,j]/np.sqrt(modalmass_j)
        
        Q = Q_norm
        Q_norm = np.zeros_like(Q)

        for j in range(nDOF):
            q_j = Q[:,j]
            iMax = np.argmax(np.abs(q_j))
            scale = q_j[iMax] # not using abs to normalize to "1" and not "+/-1"
            Q_norm[:,j]= Q[:,j]/scale

        outputs['Q_mass_norm'] = Q_norm

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        nDOF = self.nodal_data['nDOF_r']
        Q = inputs['Q_raw']
        M = inputs['Mr_glob']

        partials['Q_mass_norm', 'Mr_glob'] = np.zeros(((nDOF*nDOF),(nDOF*nDOF)))
        partials['Q_mass_norm', 'Q_raw'] = np.zeros(((nDOF*nDOF),(nDOF*nDOF)))