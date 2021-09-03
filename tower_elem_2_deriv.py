import numpy as np

from openmdao.api import ExplicitComponent


class TowerElem2Deriv(ExplicitComponent):
    # Second derivative of tower element x positions (curvature)

    def initialize(self):
        self.options.declare('nNode', types=int)
        self.options.declare('nElem', types=int)
        self.options.declare('nDOF', types=int)

    def setup(self):
        nNode = self.options['nNode']
        nElem = self.options['nElem']

        self.add_input('z_towernode', val=np.zeros(nNode), units='m/m')
        self.add_input('x_d_towernode', val=np.zeros(nNode), units='1/m')

        self.add_output('x_dd_towerelem', val=np.zeros(nElem), units='1/(m**2)')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        z = inputs['z_towernode']
        x_d = inputs['x_d_towernode']

        N_tower = len(z)

        h = np.zeros(N_tower - 1)
        for i in range(N_tower - 1):
            h[i] = z[i + 1] - z[i]

        outputs['x_dd_towerelem'] = np.zeros(N_tower - 1)

        for i in range(N_tower - 1):
            outputs['x_dd_towerelem'][i] = 1. / h[i] * (x_d[i + 1] - x_d[i])

    ##TODO Check these partials, add Z_tower partial
    def compute_partials(self, inputs, partials):
        z = inputs['z_towernode']
        x_d = inputs['x_d_towernode']

        N_tower = len(z)

        h = np.zeros(N_tower - 1)
        for i in range(N_tower - 1):
            h[i] = z[i + 1] - z[i]

        partials['x_dd_towerelem', 'z_towernode'] = np.zeros((N_tower - 1, N_tower))
        partials['x_dd_towerelem', 'x_d_towernode'] = np.zeros((N_tower - 1, N_tower))

        for i in range(N_tower - 1):
            partials['x_dd_towerelem', 'z_towernode'][i, i] = 1. / h[i]**2. * (x_d[i + 1] - x_d[i])
            partials['x_dd_towerelem', 'z_towernode'][i, i + 1] = -1. / h[i]**2. * (x_d[i + 1] - x_d[i])

            partials['x_dd_towerelem', 'x_d_towernode'][i, i] = -1. / h[i]
            partials['x_dd_towerelem', 'x_d_towernode'][i, i + 1] = 1. / h[i]
