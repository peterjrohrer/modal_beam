import numpy as np
import openmdao.api as om

class EigenvecsMassNorm(om.ExplicitComponent):
    # Mass normalize eigenvectors

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nDOF_r = self.nodal_data['nDOF_r']

        self.add_input('M_mode_eig', val=np.zeros((nDOF_r, nDOF_r)), units='kg')
        self.add_input('Q_raw', val=np.zeros((nDOF_r, nDOF_r)))

        self.add_output('Q_mass_norm', val=np.zeros((nDOF_r, nDOF_r)))

    def setup_partials(self):
        self.declare_partials('Q_mass_norm','M_mode_eig')
        self.declare_partials('Q_mass_norm','Q_raw')
        
    def compute(self, inputs, outputs):
        nDOF = self.nodal_data['nDOF_r']
        Q = inputs['Q_raw']
        M_modal = inputs['M_mode_eig']
        Q_norm = np.zeros_like(Q)
                
        for j in range(nDOF):
            q_j = Q[:,j]
            modalmass_j = M_modal[j,j]

            # Quick fix to avoid problems with NaN
            if modalmass_j < 0. :
                print('[WARN] Found negative modal mass at position %d' %j)
                modalmass_j = 1.0e-6

            Q_norm[:,j] = Q[:,j]/np.sqrt(modalmass_j)

        outputs['Q_mass_norm'] = Q_norm

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        nDOF = self.nodal_data['nDOF_r']
        Q = inputs['Q_raw']
        M = inputs['M_mode_eig']

        partials['Q_mass_norm', 'M_mode_eig'] = np.zeros(((nDOF*nDOF),(nDOF*nDOF)))
        partials['Q_mass_norm', 'Q_raw'] = np.zeros(((nDOF*nDOF),(nDOF*nDOF)))