import numpy as np
import scipy.sparse
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
        nPart = nDOF**2

        self.declare_partials('Q_sort', 'Q_raw', val=scipy.sparse.coo_array(np.zeros((nPart,nPart))))
        self.declare_partials('Q_sort', 'sort_idx', dependent=False)

    def compute(self, inputs, outputs):
        I = inputs['sort_idx'].astype(int)
        outputs['Q_sort'] = inputs['Q_raw'][:,I]

    def compute_partials(self, inputs, partials):
        nDOF = self.nodal_data['nDOF_r']
        nPart = nDOF**2
        I = inputs['sort_idx'].astype(int)

        Hcols = np.tile(I,nDOF) + np.repeat((np.arange(nDOF)*nDOF),nDOF)
        partials['Q_sort', 'Q_raw'] = scipy.sparse.coo_array((np.ones(nPart), (np.arange(nPart), Hcols)), shape=(nPart, nPart))