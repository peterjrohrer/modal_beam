import numpy as np
from openmdao.api import ExplicitComponent

class ModeshapePointMass(ExplicitComponent):

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nDOF = self.nodal_data['nDOF_tot']
       
        self.add_input('M_glob_pre', val=np.zeros((nDOF, nDOF)), units='kg')
        self.add_input('tip_mass_mat', val=np.zeros((6, 6)), units='kg')
        
        self.add_output('M_glob', val=np.zeros((nDOF, nDOF)), units='kg')

    def setup_partials(self):
        nDOF = self.nodal_data['nDOF_tot']
        nPart = nDOF * nDOF
        IDOF_tip = self.nodal_data['IDOF_tip']

        Hrows = np.zeros(36)
        for i in range(6):
            ix1 = i * 6
            ix2 = (i+1) * 6
            Hrows[ix1:ix2] = (IDOF_tip[i]*nDOF)+IDOF_tip

        self.declare_partials('M_glob', 'M_glob_pre', rows=np.arange(nPart), cols=np.arange(nPart), val=np.ones(nPart)) # declare a constant, sparse partial here
        self.declare_partials('M_glob', 'tip_mass_mat', rows=Hrows, cols=np.arange(36), val=np.ones(36))

    def compute(self, inputs, outputs):
        IDOF_tip = self.nodal_data['IDOF_tip']

        M_tip = inputs['tip_mass_mat']
        M_glob = inputs['M_glob_pre']
        M_glob[np.ix_(IDOF_tip, IDOF_tip)] += M_tip
       
        outputs['M_glob'] = M_glob

    def compute_partials(self, inputs, partials):
        pass