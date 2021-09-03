import numpy as np
import myconstants as myconst

from openmdao.api import ExplicitComponent


class GlobalStiffness(ExplicitComponent):
    # A 3DOF modal stiffness matrix

    def setup(self):
        self.nNode = 3

        self.add_input('K11', val=0., units='N/m')
        self.add_input('K12', val=0., units='N/m')
        self.add_input('K13', val=0., units='N/m')
        self.add_input('K22', val=0., units='N/m')
        self.add_input('K23', val=0., units='N/m')
        self.add_input('K33', val=0., units='N/m')

        self.add_output('K_global', val=np.zeros((3, 3)), units='N/m')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        
        K11 = inputs['K11']
        K12 = inputs['K12']
        K13 = inputs['K13']
        K22 = inputs['K22']
        K23 = inputs['K23']
        K33 = inputs['K33']

        outputs['K_global'] = np.zeros((3, 3))

        outputs['K_global'][0, 0] += K11
        outputs['K_global'][0, 1] += K12
        outputs['K_global'][1, 0] += K12
        outputs['K_global'][1, 1] += K22 
        outputs['K_global'][2, 0] += K13
        outputs['K_global'][0, 2] += K13
        outputs['K_global'][2, 1] += K23
        outputs['K_global'][1, 2] += K23
        outputs['K_global'][2, 2] += K33 

    def compute_partials(self, inputs, partials):
        
        partials['K_global', 'K11'] = np.array([1., 0., 0., 0., 0., 0., 0., 0., 0.])
        partials['K_global', 'K12'] = np.array([0., 1., 0., 1., 0., 0., 0., 0., 0.])
        partials['K_global', 'K13'] = np.array([0., 0., 1., 0., 0., 0., 1., 0., 0.])
        partials['K_global', 'K22'] = np.array([0., 0., 0., 0., 1., 0., 0., 0., 0.])
        partials['K_global', 'K23'] = np.array([0., 0., 0., 0., 0., 1., 0., 1., 0.])
        partials['K_global', 'K33'] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 1.])
