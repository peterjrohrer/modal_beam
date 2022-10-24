import numpy as np

from openmdao.api import ExplicitComponent


class BeamNode1RHS(ExplicitComponent):
    # Righthand side of beam modeshape linear system

    def initialize(self):
        self.options.declare('nNode', types=int)
        self.options.declare('nElem', types=int)
        self.options.declare('nDOF', types=int)

    def setup(self):
        nNode = self.options['nNode']

        self.add_input('z_beamnode', val=np.zeros(nNode), units='m/m')
        self.add_input('x_beamnode', val=np.zeros(nNode), units='m/m')

        self.add_output('beam_spline_rhs', val=np.zeros(nNode), units='m/m')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        z = inputs['z_beamnode']
        x = inputs['x_beamnode']

        N_beam = len(z)

        h = np.zeros(N_beam - 1)
        delta = np.zeros(N_beam - 1)
        for i in range(N_beam - 1):
            h[i] = z[i + 1] - z[i]
            delta[i] = (x[i + 1] - x[i]) / (z[i + 1] - z[i])

        outputs['beam_spline_rhs'] = np.zeros(N_beam)

        ## --- SparOpt 
        # Looks like not-a-knot
        # https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.296.7452&rep=rep1&type=pdf
        for i in range(1, N_beam - 1):
            outputs['beam_spline_rhs'][i] = 3. * (h[i - 1] * delta[i] + h[i] * delta[i - 1])

        outputs['beam_spline_rhs'][0] = ((2. * h[1] + 3. * h[0]) * h[1] * delta[0] + h[0]**2. * delta[1]) / (h[0] + h[1])
        outputs['beam_spline_rhs'][-1] = ((2. * h[-2] + 3. * h[-1]) * h[-2] * delta[-1] + h[-1]**2. * delta[-2]) / (h[-1] + h[-2])

        # ## --- TLPOpt
        # # Attempting 'natural' from https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.296.7452&rep=rep1&type=pdf
        # for i in range(1, N_beam - 1):
        #     outputs['beam_spline_rhs'][i] = (h[i - 1] * delta[i]) + (h[i] * delta[i - 1])

        # outputs['beam_spline_rhs'][0] = delta[0]
        # outputs['beam_spline_rhs'][-1] = delta[-1]

        # print('rhs done')

    def compute_partials(self, inputs, partials):
        z = inputs['z_beamnode']
        x = inputs['x_beamnode']

        N_beam = len(z)

        h = np.zeros(N_beam - 1)
        delta = np.zeros(N_beam - 1)
        for i in range(N_beam - 1):
            h[i] = z[i + 1] - z[i]
            delta[i] = (x[i + 1] - x[i]) / (z[i + 1] - z[i])

        partials['beam_spline_rhs', 'z_beamnode'] = np.zeros((N_beam, N_beam))
        partials['beam_spline_rhs', 'x_beamnode'] = np.zeros((N_beam, N_beam))

        for i in range(1, N_beam - 1):
            partials['beam_spline_rhs', 'z_beamnode'][i, i - 1] = -3. * delta[i] + 3. * h[i] * delta[i - 1] / h[i - 1]
            partials['beam_spline_rhs', 'z_beamnode'][i, i] = 3. * (delta[i] + h[i - 1] * delta[i] / h[i] - delta[i - 1] - h[i] * delta[i - 1] / h[i - 1])
            partials['beam_spline_rhs', 'z_beamnode'][i, i + 1] = -3. * h[i - 1] * delta[i] / h[i] + 3. * delta[i - 1]

            partials['beam_spline_rhs', 'x_beamnode'][i, i - 1] = -3. * h[i] / h[i - 1]
            partials['beam_spline_rhs', 'x_beamnode'][i, i] = -3. * h[i - 1] / h[i] + 3. * h[i] / h[i - 1]
            partials['beam_spline_rhs', 'x_beamnode'][i, i + 1] = 3. * h[i - 1] / h[i]

        partials['beam_spline_rhs', 'z_beamnode'][0,0] = -(3. * h[1] * delta[0] + 2. * h[0] * delta[1]) / (h[0] + h[1]) + ((2. * h[1] + 3. * h[0]) * h[1] * delta[0] + h[0]**2. * delta[1]) / (h[0] + h[1])**2. + (2. * h[1] + 3. * h[0]) * h[1] / (h[0] + h[1]) * delta[0] / h[0]
        partials['beam_spline_rhs', 'z_beamnode'][0,1] = (3. * h[1] * delta[0] + 2. * h[0] * delta[1]) / (h[0] + h[1]) - (4. * h[1] + 3. * h[0]) * delta[0] / (h[0] + h[1]) - (2. * h[1] + 3 * h[0]) * h[1] / (h[0] + h[1]) * delta[0] / h[0] + h[0]**2. / (h[0] + h[1]) * delta[1] / h[1]
        partials['beam_spline_rhs', 'z_beamnode'][0,2] = (4. * h[1] + 3. * h[0]) * delta[0] / (h[0] + h[1]) - ((2. * h[1] + 3. * h[0]) * h[1] * delta[0] + h[0]**2. * delta[1]) / (h[0] + h[1])**2. - h[0]**2. / (h[0] + h[1]) * delta[1] / h[1]

        partials['beam_spline_rhs', 'x_beamnode'][0,0] = -(2. * h[1] + 3. * h[0]) * h[1] / (h[0] + h[1]) * 1. / h[0]
        partials['beam_spline_rhs', 'x_beamnode'][0,1] = (2. * h[1] + 3 * h[0]) * h[1] / (h[0] + h[1]) * 1. / h[0] - h[0]**2. / (h[0] + h[1]) * 1. / h[1]
        partials['beam_spline_rhs', 'x_beamnode'][0,2] = h[0]**2. / (h[0] + h[1]) * 1. / h[1]

        partials['beam_spline_rhs', 'z_beamnode'][-1,-1] = (3. * h[-2] * delta[-1] + 2. * h[-1] * delta[-2]) / (h[-1] + h[-2]) - ((2. * h[-2] + 3. * h[-1]) * h[-2] * delta[-1] + h[-1]**2. * delta[-2]) / (h[-1] + h[-2])**2. - (2. * h[-2] + 3. * h[-1]) * h[-2] / (h[-1] + h[-2]) * delta[-1] / h[-1]
        partials['beam_spline_rhs', 'z_beamnode'][-1,-2] = -(3. * h[-2] * delta[-1] + 2. * h[-1] * delta[-2]) / (h[-1] + h[-2]) + (4. * h[-2] + 3. * h[-1]) * delta[-1] / (h[-1] + h[-2]) + (2. * h[-2] + 3 * h[-1]) * h[-2] / (h[-1] + h[-2]) * delta[-1] / h[-1] - h[-1]**2. / (h[-1] + h[-2]) * delta[-2] / h[-2]
        partials['beam_spline_rhs', 'z_beamnode'][-1,-3] = -(4. * h[-2] + 3. * h[-1]) * delta[-1] / (h[-1] + h[-2]) + ((2. * h[-2] + 3. * h[-1]) * h[-2] * delta[-1] + h[-1]**2. * delta[-2]) / (h[-1] + h[-2])**2. + h[-1]**2. / (h[-1] + h[-2]) * delta[-2] / h[-2]

        partials['beam_spline_rhs', 'x_beamnode'][-1,-1] = (2. * h[-2] + 3. * h[-1]) * h[-2] / (h[-1] + h[-2]) * 1. / h[-1]
        partials['beam_spline_rhs', 'x_beamnode'][-1,-2] = -(2. * h[-2] + 3 * h[-1]) * h[-2] / (h[-1] + h[-2]) * 1. / h[-1] + h[-1]**2. / (h[-1] + h[-2]) * 1. / h[-2]
        partials['beam_spline_rhs', 'x_beamnode'][-1,-3] = -h[-1]**2. / (h[-1] + h[-2]) * 1. / h[-2]
