import numpy as np
from openmdao.api import ExplicitComponent

class ModalReduction(ExplicitComponent):

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nDOF_tot = self.nodal_data['nDOF_tot']
        nDOF_r = self.nodal_data['nDOF_r']
        nMode = self.nodal_data['nMode']

        self.add_input('Q_full', val=np.zeros((nDOF_tot, nDOF_r)))
        self.add_input('eig_freqs_full', val=np.zeros(nDOF_r), units='1/s')
    
        self.add_output('Q', val=np.zeros((nDOF_tot, nMode)))
        self.add_output('eig_freqs', val=np.zeros(nMode), units='1/s')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        nMode = self.nodal_data['nMode']
       
        outputs['Q'] = inputs['Q_full'][:,:nMode]
        outputs['eig_freqs'] = inputs['eig_freqs_full'][:nMode]

    def compute_partials(self, inputs, partials):
        nElem = self.nodal_data['nElem']
        nDOF_tot = self.nodal_data['nDOF_tot']
        nDOF_r = self.nodal_data['nDOF_r']
        Tr = self.nodal_data['Tr']
