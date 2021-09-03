import numpy as np

from openmdao.api import ExplicitComponent


class TowerNode1RHS(ExplicitComponent):
    # Righthand side of tower modeshape linear system

    def initialize(self):
        self.options.declare('nNode', types=int)
        self.options.declare('nElem', types=int)
        self.options.declare('nDOF', types=int)

    def setup(self):
        nNode = self.options['nNode']

        self.add_input('z_towernode', val=np.zeros(nNode), units='m/m')
        self.add_input('x_towernode', val=np.zeros(nNode), units='m/m')

        self.add_output('tower_spline_rhs', val=np.zeros(nNode), units='m/m')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        z = inputs['z_towernode']
        x = inputs['x_towernode']

        N_tower = len(z)

        h = np.zeros(N_tower - 1)
        delta = np.zeros(N_tower - 1)
        for i in range(N_tower - 1):
            h[i] = z[i + 1] - z[i]
            delta[i] = (x[i + 1] - x[i]) / (z[i + 1] - z[i])

        outputs['tower_spline_rhs'] = np.zeros(N_tower)

        ## --- SparOpt 
        # Looks like not-a-knot
        # https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.296.7452&rep=rep1&type=pdf
        for i in range(1, N_tower - 1):
            outputs['tower_spline_rhs'][i] = 3. * (h[i - 1] * delta[i] + h[i] * delta[i - 1])

        outputs['tower_spline_rhs'][0] = ((2. * h[1] + 3. * h[0]) * h[1] * delta[0] + h[0]**2. * delta[1]) / (h[0] + h[1])
        outputs['tower_spline_rhs'][-1] = ((2. * h[-2] + 3. * h[-1]) * h[-2] * delta[-1] + h[-1]**2. * delta[-2]) / (h[-1] + h[-2])

        # ## --- TLPOpt
        # # Attempting 'natural' from https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.296.7452&rep=rep1&type=pdf
        # for i in range(1, N_tower - 1):
        #     outputs['tower_spline_rhs'][i] = (h[i - 1] * delta[i]) + (h[i] * delta[i - 1])

        # outputs['tower_spline_rhs'][0] = delta[0]
        # outputs['tower_spline_rhs'][-1] = delta[-1]

        # print('rhs done')

    def compute_partials(self, inputs, partials):
        z = inputs['z_towernode']
        x = inputs['x_towernode']

        N_tower = len(z)

        h = np.zeros(N_tower - 1)
        delta = np.zeros(N_tower - 1)
        for i in range(N_tower - 1):
            h[i] = z[i + 1] - z[i]
            delta[i] = (x[i + 1] - x[i]) / (z[i + 1] - z[i])

        partials['tower_spline_rhs', 'z_towernode'] = np.zeros((N_tower, N_tower))
        partials['tower_spline_rhs', 'x_towernode'] = np.zeros((N_tower, N_tower))

        for i in range(1, N_tower - 1):
            partials['tower_spline_rhs', 'z_towernode'][i, i - 1] = -3. * delta[i] + 3. * h[i] * delta[i - 1] / h[i - 1]
            partials['tower_spline_rhs', 'z_towernode'][i, i] = 3. * (delta[i] + h[i - 1] * delta[i] / h[i] - delta[i - 1] - h[i] * delta[i - 1] / h[i - 1])
            partials['tower_spline_rhs', 'z_towernode'][i, i + 1] = -3. * h[i - 1] * delta[i] / h[i] + 3. * delta[i - 1]

            partials['tower_spline_rhs', 'x_towernode'][i, i - 1] = -3. * h[i] / h[i - 1]
            partials['tower_spline_rhs', 'x_towernode'][i, i] = -3. * h[i - 1] / h[i] + 3. * h[i] / h[i - 1]
            partials['tower_spline_rhs', 'x_towernode'][i, i + 1] = 3. * h[i - 1] / h[i]

        partials['tower_spline_rhs', 'z_towernode'][0,0] = -(3. * h[1] * delta[0] + 2. * h[0] * delta[1]) / (h[0] + h[1]) + ((2. * h[1] + 3. * h[0]) * h[1] * delta[0] + h[0]**2. * delta[1]) / (h[0] + h[1])**2. + (2. * h[1] + 3. * h[0]) * h[1] / (h[0] + h[1]) * delta[0] / h[0]
        partials['tower_spline_rhs', 'z_towernode'][0,1] = (3. * h[1] * delta[0] + 2. * h[0] * delta[1]) / (h[0] + h[1]) - (4. * h[1] + 3. * h[0]) * delta[0] / (h[0] + h[1]) - (2. * h[1] + 3 * h[0]) * h[1] / (h[0] + h[1]) * delta[0] / h[0] + h[0]**2. / (h[0] + h[1]) * delta[1] / h[1]
        partials['tower_spline_rhs', 'z_towernode'][0,2] = (4. * h[1] + 3. * h[0]) * delta[0] / (h[0] + h[1]) - ((2. * h[1] + 3. * h[0]) * h[1] * delta[0] + h[0]**2. * delta[1]) / (h[0] + h[1])**2. - h[0]**2. / (h[0] + h[1]) * delta[1] / h[1]

        partials['tower_spline_rhs', 'x_towernode'][0,0] = -(2. * h[1] + 3. * h[0]) * h[1] / (h[0] + h[1]) * 1. / h[0]
        partials['tower_spline_rhs', 'x_towernode'][0,1] = (2. * h[1] + 3 * h[0]) * h[1] / (h[0] + h[1]) * 1. / h[0] - h[0]**2. / (h[0] + h[1]) * 1. / h[1]
        partials['tower_spline_rhs', 'x_towernode'][0,2] = h[0]**2. / (h[0] + h[1]) * 1. / h[1]

        partials['tower_spline_rhs', 'z_towernode'][-1,-1] = (3. * h[-2] * delta[-1] + 2. * h[-1] * delta[-2]) / (h[-1] + h[-2]) - ((2. * h[-2] + 3. * h[-1]) * h[-2] * delta[-1] + h[-1]**2. * delta[-2]) / (h[-1] + h[-2])**2. - (2. * h[-2] + 3. * h[-1]) * h[-2] / (h[-1] + h[-2]) * delta[-1] / h[-1]
        partials['tower_spline_rhs', 'z_towernode'][-1,-2] = -(3. * h[-2] * delta[-1] + 2. * h[-1] * delta[-2]) / (h[-1] + h[-2]) + (4. * h[-2] + 3. * h[-1]) * delta[-1] / (h[-1] + h[-2]) + (2. * h[-2] + 3 * h[-1]) * h[-2] / (h[-1] + h[-2]) * delta[-1] / h[-1] - h[-1]**2. / (h[-1] + h[-2]) * delta[-2] / h[-2]
        partials['tower_spline_rhs', 'z_towernode'][-1,-3] = -(4. * h[-2] + 3. * h[-1]) * delta[-1] / (h[-1] + h[-2]) + ((2. * h[-2] + 3. * h[-1]) * h[-2] * delta[-1] + h[-1]**2. * delta[-2]) / (h[-1] + h[-2])**2. + h[-1]**2. / (h[-1] + h[-2]) * delta[-2] / h[-2]

        partials['tower_spline_rhs', 'x_towernode'][-1,-1] = (2. * h[-2] + 3. * h[-1]) * h[-2] / (h[-1] + h[-2]) * 1. / h[-1]
        partials['tower_spline_rhs', 'x_towernode'][-1,-2] = -(2. * h[-2] + 3 * h[-1]) * h[-2] / (h[-1] + h[-2]) * 1. / h[-1] + h[-1]**2. / (h[-1] + h[-2]) * 1. / h[-2]
        partials['tower_spline_rhs', 'x_towernode'][-1,-3] = -h[-1]**2. / (h[-1] + h[-2]) * 1. / h[-2]
