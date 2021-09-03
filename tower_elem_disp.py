import numpy as np

from openmdao.api import ExplicitComponent


class TowerElemDisp(ExplicitComponent):

    def initialize(self):
        self.options.declare('nNode', types=int)
        self.options.declare('nElem', types=int)
        self.options.declare('nDOF', types=int)

    def setup(self):
        nNode = self.options['nNode']
        nElem = self.options['nElem']

        self.add_input('x_towernode', val=np.zeros(nNode), units='m/m')
        self.add_input('z_towernode', val=np.zeros(nNode), units='m/m')
        self.add_input('z_towerelem', val=np.zeros(nElem), units='m/m')
        self.add_input('x_d_towernode', val=np.zeros(nNode), units='1/m')

        self.add_output('x_towerelem', val=np.zeros(nElem), units='m/m')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        z = inputs['z_towernode']
        z_elem = inputs['z_towerelem']
        x = inputs['x_towernode']
        x_d = inputs['x_d_towernode']

        N_tower = len(z)

        h = np.zeros(N_tower - 1)
        for i in range(N_tower - 1):
            h[i] = z[i + 1] - z[i]

        outputs['x_towerelem'] = np.zeros(N_tower - 1)

        for i in range(N_tower - 1):
            # # Original
            # outputs['x_towerelem'][i] = (x[i + 1] + x[i]) / 2. - 1. / 8. * h[i] * (x_d[i + 1] - x_d[i])
            # Applying spline
            outputs['x_towerelem'][i] = ((x_d[i]/6.) * (((z[i+1] - z_elem[i])**3)/h[i] - (h[i]*(z[i+1] - z_elem[i])))) + ((x_d[i+1]/6.) * (((z_elem[i] - z[i])**3)/h[i] - (h[i]*(z_elem[i] - z[i])))) + (x[i] * (z[i+1] - z_elem[i])/h[i]) + (x[i + 1] * (z_elem[i] - z[i])/h[i])

    ##TODO Check these partials, add Z_tower partial
    def compute_partials(self, inputs, partials):
        z = inputs['z_towernode']
        x = inputs['x_towernode']
        x_d = inputs['x_d_towernode']

        N_tower = len(z)

        h = np.zeros(N_tower - 1)
        for i in range(N_tower - 1):
            h[i] = z[i + 1] - z[i]

        partials['x_towerelem', 'z_towernode'] = np.zeros((N_tower - 1, N_tower))
        partials['x_towerelem', 'x_towernode'] = np.zeros((N_tower - 1, N_tower))
        partials['x_towerelem', 'x_d_towernode'] = np.zeros((N_tower - 1, N_tower))

        for i in range(N_tower - 1):
            partials['x_towerelem', 'z_towernode'][i, i] = 1. / 8. * (x_d[i + 1] - x_d[i])
            partials['x_towerelem', 'z_towernode'][i, i + 1] = -1. / 8. * (x_d[i + 1] - x_d[i])

            partials['x_towerelem', 'x_towernode'][i, i] = 1. / 2.
            partials['x_towerelem', 'x_towernode'][i, i + 1] = 1. / 2.

            partials['x_towerelem', 'x_d_towernode'][i, i] = 1. / 8. * h[i]
            partials['x_towerelem', 'x_d_towernode'][i, i + 1] = -1. / 8. * h[i]
