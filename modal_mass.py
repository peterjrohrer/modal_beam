import numpy as np
from openmdao.api import ExplicitComponent

class ModalMass(ExplicitComponent):
    # Calculate modal mass for TLPWT 
    
    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nElem = self.nodal_data['nElem']
        nDOF_r = self.nodal_data['nDOF_r']
        nDOF_tot = self.nodal_data['nDOF_tot']
        nMode = self.nodal_data['nMode']

        self.add_input('M_glob', val=np.zeros((nDOF_tot, nDOF_tot)), units='kg')
        self.add_input('Q', val=np.ones((nDOF_tot, nDOF_r)))

        self.add_output('M_modal', val=np.zeros((nMode,nMode)), units='kg')

    def	setup_partials(self):
        self.declare_partials('M_modal', ['M_glob', 'Q'])

    def compute(self, inputs, outputs):
        nMode = self.nodal_data['nMode']

        Q = inputs['Q']
        M_glob = inputs['M_glob']
        M_modal = Q[:,:nMode].T @ M_glob @ Q[:,:nMode]
        
        outputs['M_modal'] = M_modal

    def compute_partials(self, inputs, partials):
        nMode = self.nodal_data['nMode']

        Q = inputs['Q']
        M_glob = inputs['M_glob']