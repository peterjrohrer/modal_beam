
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

    def	setup_partials(self):
        self.declare_partials('K_modal', ['K_glob', 'Q'])

    def compute(self, inputs, outputs):
        nMode = self.nodal_data['nMode']

        Q = inputs['Q']
        K_glob = inputs['K_glob']
        K_modal = Q.T @ K_glob @ Q
        
        outputs['K_modal'] = K_modal

    def compute_partials(self, inputs, partials):
        nMode = self.nodal_data['nMode']

        Q = inputs['Q']
        K_glob = inputs['K_glob']