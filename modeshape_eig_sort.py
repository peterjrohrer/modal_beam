import numpy as np
from openmdao.api import ExplicitComponent

class EigenvecsSort(ExplicitComponent):

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nDOF = self.nodal_data['nDOF_r']

        self.add_input('Q_raw', val=np.zeros((nDOF, nDOF)))
        self.add_input('sort_idx', val=np.zeros(nDOF))
    
        self.add_output('Q_sort', val=np.zeros((nDOF, nDOF)))

    def setup_partials(self):
        nDOF = self.nodal_data['nDOF_r']
        
        # Hcols = np.array([])
        # Hcols0 = np.arange(N_mode)
        # for i in range(nDOF):
        #     Hcols_add = Hcols0 + (i*nDOF)
        #     Hcols = np.concatenate((Hcols,Hcols_add))

        # self.declare_partials('Q_sort', 'Q_raw', rows=np.arange(N_part), cols=Hcols, val=np.ones(N_part))

    def compute(self, inputs, outputs):

        I = inputs['sort_idx'].astype(int)
        outputs['Q_sort'] = inputs['Q_raw'][:,I]

    def compute_partials(self, inputs, partials):
        pass