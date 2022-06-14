import numpy as np

from openmdao.api import ExplicitComponent


class BeamNode1LHS(ExplicitComponent):
    # Lefthand side of beam modeshape linear system

    def initialize(self):
        self.options.declare('nNode', types=int)
        self.options.declare('nElem', types=int)
        self.options.declare('nDOF', types=int)

    def setup(self):
        nNode = self.options['nNode']

        self.add_input('z_beamnode', val=np.zeros(nNode), units='m/m')

        self.add_output('beam_spline_lhs', val=np.zeros((nNode, nNode)), units='m/m')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        z = inputs['z_beamnode']

        N_beam = len(z)

        h = np.zeros(N_beam - 1)
        for i in range(N_beam - 1):
            h[i] = z[i + 1] - z[i]

        outputs['beam_spline_lhs'] = np.zeros((N_beam, N_beam))

        ## --- SparOpt 
        # Looks like not-a-knot
        # https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.296.7452&rep=rep1&type=pdf
        for i in range(1, N_beam - 1):
            outputs['beam_spline_lhs'][i, i] = 2. * (h[i] + h[i - 1])
            outputs['beam_spline_lhs'][i, i - 1] = h[i]
            outputs['beam_spline_lhs'][i, i + 1] = h[i - 1]

        outputs['beam_spline_lhs'][0, 0] = h[1]
        outputs['beam_spline_lhs'][0, 1] = h[0] + h[1]
        outputs['beam_spline_lhs'][-1, -1] = h[-2]
        outputs['beam_spline_lhs'][-1, -2] = h[-1] + h[-2]

        # ## --- TLPOpt
        # # Attempting 'natural' from https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.296.7452&rep=rep1&type=pdf
        # for i in range(1, N_beam - 1):
        #     outputs['beam_spline_lhs'][i, i] = 2. * (h[i] + h[i - 1])
        #     outputs['beam_spline_lhs'][i, i - 1] = h[i]
        #     outputs['beam_spline_lhs'][i, i + 1] = h[i - 1]

        # outputs['beam_spline_lhs'][0, 0] = 2.
        # outputs['beam_spline_lhs'][0, 1] = 1.
        # outputs['beam_spline_lhs'][-1, -1] = 2.
        # outputs['beam_spline_lhs'][-1, -2] = 1.

        # print('lhs done')

    def compute_partials(self, inputs, partials):
        z = inputs['z_beamnode']

        N_beam = len(z)

        partials['beam_spline_lhs', 'z_beamnode'] = np.zeros(
            (N_beam * N_beam, N_beam))

        for i in range(1, N_beam - 1):
            partials['beam_spline_lhs', 'z_beamnode'][i*N_beam-1+i,i] = -1.
            partials['beam_spline_lhs', 'z_beamnode'][i*N_beam-1+i,i+1] = 1.

            partials['beam_spline_lhs', 'z_beamnode'][i*N_beam+i,i-1] = -2.
            partials['beam_spline_lhs', 'z_beamnode'][i*N_beam+i,i] = 0.
            partials['beam_spline_lhs', 'z_beamnode'][i*N_beam+i,i+1] = 2.

            partials['beam_spline_lhs', 'z_beamnode'][i*N_beam+1+i,i-1] = -1.
            partials['beam_spline_lhs', 'z_beamnode'][i*N_beam+1+i,i] = 1.

        partials['beam_spline_lhs', 'z_beamnode'][0, 1] = -1.
        partials['beam_spline_lhs', 'z_beamnode'][0, 2] = 1.
        partials['beam_spline_lhs', 'z_beamnode'][1, 0] = -1.
        partials['beam_spline_lhs', 'z_beamnode'][1, 2] = 1.
        partials['beam_spline_lhs', 'z_beamnode'][-1, -3] = -1.
        partials['beam_spline_lhs', 'z_beamnode'][-1, -2] = 1.
        partials['beam_spline_lhs', 'z_beamnode'][-2, -3] = -1.
        partials['beam_spline_lhs', 'z_beamnode'][-2, -1] = 1.
