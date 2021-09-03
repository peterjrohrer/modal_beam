import numpy as np

from openmdao.api import ExplicitComponent


class TowerTotalMass(ExplicitComponent):

    def setup(self):
        self.add_input('M_tower', val=np.zeros(10), units='kg')

        self.add_output('tot_M_tower', val=1., units='kg')

        self.declare_partials('tot_M_tower', 'M_tower', val=np.ones((1, 10)))

    def compute(self, inputs, outputs):
        outputs['tot_M_tower'] = np.sum(inputs['M_tower'])
