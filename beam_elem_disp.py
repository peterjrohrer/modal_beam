import numpy as np

from openmdao.api import ExplicitComponent


class BeamElemDisp(ExplicitComponent):

    def initialize(self):
        self.options.declare('nNode', types=int)
        self.options.declare('nElem', types=int)
        self.options.declare('nDOF', types=int)

    def setup(self):
        nNode = self.options['nNode']
        nElem = self.options['nElem']

        self.add_input('x_beamnode', val=np.zeros(nNode), units='m/m')
        self.add_input('z_beamnode', val=np.zeros(nNode), units='m/m')
        self.add_input('z_beamelem', val=np.zeros(nElem), units='m/m')
        self.add_input('x_d_beamnode', val=np.zeros(nNode), units='1/m')

        self.add_output('x_beamelem', val=np.zeros(nElem), units='m/m')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        z = inputs['z_beamnode']
        z_elem = inputs['z_beamelem']
        x = inputs['x_beamnode']
        x_d = inputs['x_d_beamnode']

        N_beam = len(z)

        h = np.zeros(N_beam - 1)
        for i in range(N_beam - 1):
            h[i] = z[i + 1] - z[i]

        outputs['x_beamelem'] = np.zeros(N_beam - 1)

        for i in range(N_beam - 1):
            # # Original
            # outputs['x_beamelem'][i] = (x[i + 1] + x[i]) / 2. - 1. / 8. * h[i] * (x_d[i + 1] - x_d[i])
            # Applying spline
            outputs['x_beamelem'][i] = ((x_d[i]/6.) * (((z[i+1] - z_elem[i])**3)/h[i] - (h[i]*(z[i+1] - z_elem[i])))) + ((x_d[i+1]/6.) * (((z_elem[i] - z[i])**3)/h[i] - (h[i]*(z_elem[i] - z[i])))) + (x[i] * (z[i+1] - z_elem[i])/h[i]) + (x[i + 1] * (z_elem[i] - z[i])/h[i])

    ##TODO Check these partials, add Z_beam partial
    def compute_partials(self, inputs, partials):
        z = inputs['z_beamnode']
        x = inputs['x_beamnode']
        x_d = inputs['x_d_beamnode']

        N_beam = len(z)

        h = np.zeros(N_beam - 1)
        for i in range(N_beam - 1):
            h[i] = z[i + 1] - z[i]

        partials['x_beamelem', 'z_beamnode'] = np.zeros((N_beam - 1, N_beam))
        partials['x_beamelem', 'x_beamnode'] = np.zeros((N_beam - 1, N_beam))
        partials['x_beamelem', 'x_d_beamnode'] = np.zeros((N_beam - 1, N_beam))

        for i in range(N_beam - 1):
            partials['x_beamelem', 'z_beamnode'][i, i] = 1. / 8. * (x_d[i + 1] - x_d[i])
            partials['x_beamelem', 'z_beamnode'][i, i + 1] = -1. / 8. * (x_d[i + 1] - x_d[i])

            partials['x_beamelem', 'x_beamnode'][i, i] = 1. / 2.
            partials['x_beamelem', 'x_beamnode'][i, i + 1] = 1. / 2.

            partials['x_beamelem', 'x_d_beamnode'][i, i] = 1. / 8. * h[i]
            partials['x_beamelem', 'x_d_beamnode'][i, i + 1] = -1. / 8. * h[i]
