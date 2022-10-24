import numpy as np
from scipy import linalg

from openmdao.api import ImplicitComponent


class BeamNode1Deriv(ImplicitComponent):
    # Solve beam modeshape linear system

    def initialize(self):
        self.options.declare('nNode', types=int)
        self.options.declare('nElem', types=int)
        self.options.declare('nDOF', types=int)

    def setup(self):
        nNode = self.options['nNode']

        self.add_input('beam_spline_lhs', val=np.zeros((nNode, nNode)), units='m/m')
        self.add_input('beam_spline_rhs', val=np.zeros(nNode), units='m/m')

        self.add_output('x_d_beamnode', val=np.zeros(nNode), units='1/m') # This is a bit of a made-up unit

        self.declare_partials('*', '*')

    def apply_nonlinear(self, inputs, outputs, residuals):
        A = inputs['beam_spline_lhs']
        b = inputs['beam_spline_rhs']

        residuals['x_d_beamnode'] = A.dot(outputs['x_d_beamnode']) - b

    def solve_nonlinear(self, inputs, outputs):
        A = inputs['beam_spline_lhs']
        b = inputs['beam_spline_rhs']

        outputs['x_d_beamnode'] = linalg.solve(A, b)

    def linearize(self, inputs, outputs, partials):
        nNode = self.options['nNode']

        partials['x_d_beamnode', 'beam_spline_lhs'] = np.kron(np.identity(nNode), np.transpose(outputs['x_d_beamnode']))
        partials['x_d_beamnode', 'beam_spline_rhs'] = -1. * np.identity(nNode)
        partials['x_d_beamnode', 'x_d_beamnode'] = inputs['beam_spline_lhs']
