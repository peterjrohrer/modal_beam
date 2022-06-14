import numpy as np

from openmdao.api import ExplicitComponent


class BeamElem2Deriv(ExplicitComponent):
    # Second derivative of beam element x positions (curvature)

    def initialize(self):
        self.options.declare('nNode', types=int)
        self.options.declare('nElem', types=int)
        self.options.declare('nDOF', types=int)

    def setup(self):
        nNode = self.options['nNode']
        nElem = self.options['nElem']

        self.add_input('z_beamnode', val=np.zeros(nNode), units='m/m')
        self.add_input('x_d_beamnode', val=np.zeros(nNode), units='1/m')

        self.add_output('x_dd_beamelem', val=np.zeros(nElem), units='1/(m**2)')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        z = inputs['z_beamnode']
        x_d = inputs['x_d_beamnode']

        N_beam = len(z)

        h = np.zeros(N_beam - 1)
        for i in range(N_beam - 1):
            h[i] = z[i + 1] - z[i]

        outputs['x_dd_beamelem'] = np.zeros(N_beam - 1)

        for i in range(N_beam - 1):
            outputs['x_dd_beamelem'][i] = 1. / h[i] * (x_d[i + 1] - x_d[i])

    ##TODO Check these partials, add Z_beam partial
    def compute_partials(self, inputs, partials):
        z = inputs['z_beamnode']
        x_d = inputs['x_d_beamnode']

        N_beam = len(z)

        h = np.zeros(N_beam - 1)
        for i in range(N_beam - 1):
            h[i] = z[i + 1] - z[i]

        partials['x_dd_beamelem', 'z_beamnode'] = np.zeros((N_beam - 1, N_beam))
        partials['x_dd_beamelem', 'x_d_beamnode'] = np.zeros((N_beam - 1, N_beam))

        for i in range(N_beam - 1):
            partials['x_dd_beamelem', 'z_beamnode'][i, i] = 1. / h[i]**2. * (x_d[i + 1] - x_d[i])
            partials['x_dd_beamelem', 'z_beamnode'][i, i + 1] = -1. / h[i]**2. * (x_d[i + 1] - x_d[i])

            partials['x_dd_beamelem', 'x_d_beamnode'][i, i] = -1. / h[i]
            partials['x_dd_beamelem', 'x_d_beamnode'][i, i + 1] = 1. / h[i]
