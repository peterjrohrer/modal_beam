import numpy as np
from openmdao.api import ExplicitComponent

class ModeshapePointMass(ExplicitComponent):

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nDOF = self.nodal_data['nDOF_tot']
       
        self.add_input('M_glob_pre', val=np.zeros((nDOF, nDOF)), units='kg')
        
        self.add_output('M_glob', val=np.zeros((nDOF, nDOF)), units='kg')

    def setup_partials(self):
        nDOF = self.nodal_data['nDOF_tot']
        nPart = nDOF * nDOF

        self.declare_partials('M_glob', 'M_glob_pre', rows=np.arange(nPart), cols=np.arange(nPart))

    def compute(self, inputs, outputs):
        IDOF_tip = self.nodal_data['IDOF_tip']

        M_tip = np.zeros((6,6))

        M_glob = inputs['M_glob_pre']
        M_glob[np.ix_(IDOF_tip, IDOF_tip)] += M_tip
       
        outputs['M_glob'] = M_glob

    def compute_partials(self, inputs, partials):
        nDOF = self.nodal_data['nDOF_tot']

        partials['M_glob', 'M_glob_pre'] = np.ones((nDOF*nDOF))