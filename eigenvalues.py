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

        self.add_input('Kr_glob', val=np.zeros((nDOF_r, nDOF_r)), units='N/m')
        self.add_input('Q_mass_norm', val=np.zeros((nDOF_r, nDOF_r)))

        self.add_output('eigenvals_raw', val=np.zeros((nDOF_r, nDOF_r)))

    def compute(self, inputs, outputs):
        nDOF = self.nodal_data['nDOF_r']
        K = inputs['Kr_glob']
        Q = inputs['Q_mass_norm']
                
        Lambda= Q.T @ K @ Q

        outputs['eigenvals_raw'] = Lambda

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        nDOF = self.nodal_data['nDOF_r']
        K = inputs['Kr_glob']
        Q = inputs['Q_mass_norm']

        ## Based on Giles (2008), https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf (first quadratic form)
        if mode == 'rev':    
            if 'eigenvals_raw' in d_outputs:
                if 'Q_mass_norm' in d_inputs:
                    d_inputs['Q_mass_norm'] += (K @ Q @ d_outputs['eigenvals_raw'].T) + (K.T @ Q @ d_outputs['eigenvals_raw']) 
                if 'Kr_glob' in d_inputs:
                    d_inputs['Kr_glob'] += Q @ d_outputs['eigenvals_raw'] @ Q.T

        elif mode == 'fwd':
            if 'eigenvals_raw' in d_outputs:
                if 'Q_mass_norm' in d_inputs:
                    d_outputs['eigenvals_raw'] += (d_inputs['Q_mass_norm'].T @ K @ Q) + (Q.T @ K @ d_inputs['Q_mass_norm']) 
                if 'Kr_glob' in d_inputs:
                    d_outputs['eigenvals_raw'] += (Q.T @ d_inputs['Kr_glob'] @ Q) 