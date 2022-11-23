import numpy as np
from openmdao.api import ExplicitComponent

class ModalMass(ExplicitComponent):
    # Calculate modal mass for TLPWT 
    
    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nDOF_tot = self.nodal_data['nDOF_tot']
        nMode = self.nodal_data['nMode']

        self.add_input('M_glob', val=np.zeros((nDOF_tot, nDOF_tot)), units='kg')
        self.add_input('Q', val=np.ones((nDOF_tot, nMode)))

        self.add_output('M_modal', val=np.zeros((nMode,nMode)), units='kg')

    def compute(self, inputs, outputs):
        nMode = self.nodal_data['nMode']

        Q = inputs['Q']
        M_glob = inputs['M_glob']
        M_modal = Q.T @ M_glob @ Q
        
        outputs['M_modal'] = M_modal

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        M = inputs['M_glob']
        Q = inputs['Q']

        ## Based on Giles (2008), https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf (first quadratic form)
        if mode == 'rev':    
            if 'M_modal' in d_outputs:
                if 'Q' in d_inputs:
                    d_inputs['Q'] += (M @ Q @ d_outputs['M_modal'].T) + (M.T @ Q @ d_outputs['M_modal']) 
                if 'M_glob' in d_inputs:
                    d_inputs['M_glob'] += Q @ d_outputs['M_modal'] @ Q.T

        elif mode == 'fwd':
            if 'M_modal' in d_outputs:
                if 'Q' in d_inputs:
                    d_outputs['M_modal'] += (d_inputs['Q'].T @ M @ Q) + (Q.T @ M @ d_inputs['Q']) 
                if 'M_glob' in d_inputs:
                    d_outputs['M_modal'] += (Q.T @ d_inputs['M_glob'] @ Q) 