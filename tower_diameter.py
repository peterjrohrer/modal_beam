import numpy as np

from openmdao.api import ExplicitComponent


class TowerDiameter(ExplicitComponent):

    def setup(self):
        self.add_input('D_tower_p', val=np.zeros(11), units='m')

        self.add_output('D_tower', val=np.zeros(10), units='m')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        D_tower_p = inputs['D_tower_p']

        outputs['D_tower'] = np.zeros(10)

        for i in range(len(D_tower_p) - 1):
            outputs['D_tower'][i] = (D_tower_p[i] + D_tower_p[i + 1]) / 2.

    def compute_partials(self, inputs, partials):
        D_tower_p = inputs['D_tower_p']

        partials['D_tower', 'D_tower_p'] = np.zeros((len(D_tower_p) - 1, len(D_tower_p)))

        for i in range(len(D_tower_p) - 1):
            partials['D_tower', 'D_tower_p'][i, i] += 0.5
            partials['D_tower', 'D_tower_p'][i, i + 1] += 0.5
