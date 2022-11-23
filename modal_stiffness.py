
import numpy as np
import scipy.interpolate as si

from openmdao.api import ExplicitComponent


class ModalStiffness(ExplicitComponent):
    # Calculate modal stiffness for column with tendon contribution
    
    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nElem = self.nodal_data['nElem']
        nDOF_r = self.nodal_data['nDOF_r']
        nDOF_tot = self.nodal_data['nDOF_tot']
        nMode = self.nodal_data['nMode']

        self.add_input('K_glob', val=np.zeros((nDOF_tot, nDOF_tot)), units='N/m')
        self.add_input('Q', val=np.ones((nDOF_tot, nMode)))

        self.add_output('K_modal', val=np.zeros((nMode,nMode)), units='N/m')

    def compute(self, inputs, outputs):
        nMode = self.nodal_data['nMode']

        Q = inputs['Q']
        K_glob = inputs['K_glob']
        K_modal = Q.T @ K_glob @ Q
        
        outputs['K_modal'] = K_modal

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        M = inputs['K_glob']
        Q = inputs['Q']

        ## Based on Giles (2008), https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf (first quadratic form)
        if mode == 'rev':    
            if 'K_modal' in d_outputs:
                if 'Q' in d_inputs:
                    d_inputs['Q'] += (M @ Q @ d_outputs['K_modal'].T) + (M.T @ Q @ d_outputs['K_modal']) 
                if 'K_glob' in d_inputs:
                    d_inputs['K_glob'] += Q @ d_outputs['K_modal'] @ Q.T

        elif mode == 'fwd':
            if 'K_modal' in d_outputs:
                if 'Q' in d_inputs:
                    d_outputs['K_modal'] += (d_inputs['Q'].T @ M @ Q) + (Q.T @ M @ d_inputs['Q']) 
                if 'K_glob' in d_inputs:
                    d_outputs['K_modal'] += (Q.T @ d_inputs['K_glob'] @ Q)