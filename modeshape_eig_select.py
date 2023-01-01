import numpy as np
from openmdao.api import ExplicitComponent

class EigenvecsSelect(ExplicitComponent):

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nDOF = self.nodal_data['nDOF_r']
        nMode = self.nodal_data['nMode']

        self.add_input('Q_sort', val=np.zeros((nDOF, nDOF)))
    
        self.add_output('Q_basis', val=np.zeros((nDOF, nMode)))

    def setup_partials(self):
        nDOF = self.nodal_data['nDOF_r']
        nMode = self.nodal_data['nMode']
        N_part = nDOF * nMode
        
        Hcols = np.array([])
        Hcols0 = np.arange(nMode)
        for i in range(nDOF):
            Hcols_add = Hcols0 + (i*nDOF)
            Hcols = np.concatenate((Hcols,Hcols_add))

        self.declare_partials('Q_basis', 'Q_sort', rows=np.arange(N_part), cols=Hcols, val=np.ones(N_part))

    def compute(self, inputs, outputs):
        nMode = self.nodal_data['nMode']
        outputs['Q_basis'] = inputs['Q_sort'][:,:nMode]

    def compute_partials(self, inputs, partials):
        pass