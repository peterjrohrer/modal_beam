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
        nDOF_tot = self.nodal_data['nDOF_tot']
        nDOF_r = self.nodal_data['nDOF_r']
        nMode = self.nodal_data['nMode']
        
        Hrows = np.arange(nDOF_tot * nMode)
        Hcols = np.arange(nDOF_tot * nDOF_r)
        cols_idx = np.setdiff1d(np.arange(nDOF_r),np.arange(nMode))

        for i in range(nDOF_tot):
            removed_cols = (i*nDOF_r) + cols_idx
            Hcols = np.setdiff1d(Hcols,removed_cols)

        self.declare_partials('Q', 'Q_full', rows=Hrows, cols=Hcols)
        self.declare_partials('eig_freqs', 'eig_freqs_full', rows=np.arange(nMode), cols=np.arange(nMode))

    def compute(self, inputs, outputs):
        nMode = self.nodal_data['nMode']
       
        outputs['Q'] = inputs['Q_full'][:,:nMode]
        outputs['eig_freqs'] = inputs['eig_freqs_full'][:nMode]

    def compute_partials(self, inputs, partials):
        nDOF_tot = self.nodal_data['nDOF_tot']
        nMode = self.nodal_data['nMode']

        partials['Q', 'Q_full'] = np.ones(nDOF_tot * nMode)        
        partials['eig_freqs', 'eig_freqs_full'] = np.ones(nMode)        
