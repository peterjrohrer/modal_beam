import numpy as np
from openmdao.api import ExplicitComponent

class ModeshapePointStiff(ExplicitComponent):

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nDOF = self.nodal_data['nDOF_tot']
       
        self.add_input('K_glob_pre', val=np.zeros((nDOF, nDOF)), units='N/m')
        
        self.add_output('K_glob', val=np.zeros((nDOF, nDOF)), units='N/m')

    def setup_partials(self):
        nDOF = self.nodal_data['nDOF_tot']
        nPart = nDOF * nDOF

        self.declare_partials('K_glob', 'K_glob_pre', rows=np.arange(nPart), cols=np.arange(nPart))

    def compute(self, inputs, outputs):
        IDOF_tip = self.nodal_data['IDOF_tip']

        K_tip = np.zeros((6,6))

        K_glob = inputs['K_glob_pre']
        K_glob[np.ix_(IDOF_tip, IDOF_tip)] += K_tip
       
        outputs['K_glob'] = K_glob

    def compute_partials(self, inputs, partials):
        nDOF = self.nodal_data['nDOF_tot']

        partials['K_glob', 'K_glob_pre'] = np.ones((nDOF*nDOF))