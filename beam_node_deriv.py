import numpy as np
from scipy import linalg
from openmdao.api import ImplicitComponent

class BeamNodeDeriv(ImplicitComponent):
    # Solve beam modeshape linear system

    def initialize(self):
        self.options.declare('nodal_data', types=dict)
        self.options.declare('absca', types=str) # abscissa or horizontal coordinate
        self.options.declare('ordin', types=str) # ordinate or vertical coordinate
        self.options.declare('level', types=int) # derivative level

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nNode = self.nodal_data['nNode']
        nMode = self.nodal_data['nMode']
        absca = self.absca = self.options['absca']
        ordin = self.ordin = self.options['ordin']
        
        self.lhs = 'beam_spline_%s_lhs' %absca

        if self.options['level'] == 1:
            self.rhs = 'beam_spline_%s_1_rhs' %ordin
            self.otp = '%s_d_nodes' %ordin
        elif self.options['level'] == 2:
            self.rhs = 'beam_spline_%s_2_rhs' %ordin
            self.otp = '%s_dd_nodes' %ordin
        elif self.options['level'] == 3:
            self.rhs = 'beam_spline_%s_3_rhs' %ordin
            self.otp = '%s_ddd_nodes' %ordin
        else:
            raise Exception('Derivative level not defined!')

        self.add_input(self.lhs, val=np.zeros((nNode, nNode, nMode)))
        self.add_input(self.rhs, val=np.zeros((nNode, nMode)))

        self.add_output(self.otp, val=np.zeros((nNode, nMode)))

    def setup_partials(self):
        self.declare_partials(self.otp, self.lhs)
        self.declare_partials(self.otp, self.rhs)
        self.declare_partials(self.otp, self.otp)

    def apply_nonlinear(self, inputs, outputs, residuals):
        nMode = self.nodal_data['nMode']

        for m in range(nMode):
            A = inputs[self.lhs][:,:,m]
            b = inputs[self.rhs][:,m]

            residuals[self.otp][:,m] = A.dot(outputs[self.otp][:,m]) - b

    def solve_nonlinear(self, inputs, outputs):
        nMode = self.nodal_data['nMode']

        for m in range(nMode):
            A = inputs[self.lhs][:,:,m]
            b = inputs[self.rhs][:,m]
        
            outputs[self.otp][:,m] = linalg.solve(A, b)

    def linearize(self, inputs, outputs, partials):
        nNode = self.nodal_data['nNode']
        nMode = self.nodal_data['nMode']
        
        lhs_partial = np.zeros((nNode, nMode, nNode * nNode, nMode))
        rhs_partial = np.zeros((nNode, nMode, nNode, nMode))
        otp_partial = np.zeros((nNode, nMode, nNode, nMode))

        for m in range(nMode):
            lhs_partial[:, m, :, m] = np.kron(np.identity(nNode), np.transpose(outputs[self.otp][:, m]))
            rhs_partial[:, m, :, m] = -1. * np.identity(nNode)
            otp_partial[:, m, :, m] = inputs[self.lhs][:, :, m]

        partials[self.otp, self.lhs] = np.reshape(lhs_partial, (nNode * nMode, nNode * nNode * nMode))
        partials[self.otp, self.rhs] = np.reshape(rhs_partial, (nNode * nMode, nNode * nMode))
        partials[self.otp, self.otp] = np.reshape(otp_partial, (nNode * nMode, nNode * nMode))