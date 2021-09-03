import numpy as np

from openmdao.api import ExplicitComponent

class TowerThickness(ExplicitComponent):

    def setup(self):
        self.add_input('wt_tower_p', val=np.zeros(11), units='m')

        self.add_output('wt_tower', val=np.zeros(10), units='m')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        wt_tower_p = inputs['wt_tower_p']

        outputs['wt_tower'] = np.zeros(10)

        for i in range(len(wt_tower_p) - 1):
            outputs['wt_tower'][i] = (wt_tower_p[i] + wt_tower_p[i + 1]) / 2.

    def compute_partials(self, inputs, partials):
        wt_tower_p = inputs['wt_tower_p']

        partials['wt_tower', 'wt_tower_p'] = np.zeros((len(wt_tower_p) - 1, len(wt_tower_p)))

        for i in range(len(wt_tower_p) - 1):
            partials['wt_tower', 'wt_tower_p'][i, i] += 0.5
            partials['wt_tower', 'wt_tower_p'][i, i + 1] += 0.5
