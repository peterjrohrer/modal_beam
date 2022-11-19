import numpy as np
import scipy.linalg
import openmdao.api as om

class EigenvecsModalMass(om.ExplicitComponent):
    # Modal masses for eigenvector normalization

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nDOF_r = self.nodal_data['nDOF_r']
        nDOF_tot = self.nodal_data['nDOF_tot']

        self.add_input('Mr_glob', val=np.zeros((nDOF_r, nDOF_r)), units='kg')
        self.add_input('Q_raw', val=np.zeros((nDOF_r, nDOF_r)))

        self.add_output('M_mode_eig', val=np.zeros((nDOF_r, nDOF_r)), units='kg')

    def compute(self, inputs, outputs):
        M = inputs['Mr_glob']
        Q = inputs['Q_raw']

        outputs['M_mode_eig'] = Q.T @ M @ Q

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        M = inputs['Mr_glob']
        Q = inputs['Q_raw']

        ## Based on Giles (2008), https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf (first quadratic form)
        if mode == 'rev':    
            if 'M_mode_eig' in d_outputs:
                if 'Q_raw' in d_inputs:
                    d_inputs['Q_raw'] += (M @ Q @ d_outputs['M_mode_eig'].T) + (M.T @ Q @ d_outputs['M_mode_eig']) 
                if 'Mr_glob' in d_inputs:
                    d_inputs['Mr_glob'] += Q @ d_outputs['M_mode_eig'] @ Q.T

        elif mode == 'fwd':
            if 'M_mode_eig' in d_outputs:
                if 'Q_raw' in d_inputs:
                    d_outputs['M_mode_eig'] += (d_inputs['Q_raw'].T @ M @ Q) + (Q.T @ M @ d_inputs['Q_raw']) 
                if 'Mr_glob' in d_inputs:
                    d_outputs['M_mode_eig'] += (Q.T @ d_inputs['Mr_glob'] @ Q) 