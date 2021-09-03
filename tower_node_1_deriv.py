import numpy as np
from scipy import linalg

from openmdao.api import ImplicitComponent


class TowerNode1Deriv(ImplicitComponent):
    # Solve tower modeshape linear system

    def initialize(self):
        self.options.declare('nNode', types=int)
        self.options.declare('nElem', types=int)
        self.options.declare('nDOF', types=int)

    def setup(self):
        nNode = self.options['nNode']

        self.add_input('tower_spline_lhs', val=np.zeros((nNode, nNode)), units='m/m')
        self.add_input('tower_spline_rhs', val=np.zeros(nNode), units='m/m')

        self.add_output('x_d_towernode', val=np.zeros(nNode), units='1/m') # This is a bit of a made-up unit

        self.declare_partials('*', '*')

    def apply_nonlinear(self, inputs, outputs, residuals):
        A = inputs['tower_spline_lhs']
        b = inputs['tower_spline_rhs']

        residuals['x_d_towernode'] = A.dot(outputs['x_d_towernode']) - b

    def solve_nonlinear(self, inputs, outputs):
        A = inputs['tower_spline_lhs']
        b = inputs['tower_spline_rhs']

        outputs['x_d_towernode'] = linalg.solve(A, b)

    def linearize(self, inputs, outputs, partials):
        partials['x_d_towernode', 'tower_spline_lhs'] = np.kron(np.identity(11), np.transpose(outputs['x_d_towernode']))
        partials['x_d_towernode', 'tower_spline_rhs'] = -1. * np.identity(11)
        partials['x_d_towernode', 'x_d_towernode'] = inputs['tower_spline_lhs']
