import numpy as np
from scipy import linalg
from openmdao.api import ImplicitComponent

class BeamNode1Deriv(ImplicitComponent):
    # Solve beam modeshape linear system

    def initialize(self):
        self.options.declare('nodal_data', types=dict)
        self.options.declare('key1', types=str)
        self.options.declare('key2', types=str)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nNode = self.nodal_data['nNode']
        nMode = self.nodal_data['nMode']
        key1 = self.key1 = self.options['key1']
        key2 = self.key2 = self.options['key2']
        self.lhs = 'beam_spline_%s_lhs' %key1
        self.rhs = 'beam_spline_%s_1_rhs' %key2
        self.otp = '%s_d_beamnode' %key2

        self.add_input(self.lhs, val=np.zeros((nNode, nNode, nMode)), units='m/m')
        self.add_input(self.rhs, val=np.zeros((nNode, nMode)), units='m/m')

        self.add_output(self.otp, val=np.zeros((nNode, nMode)), units='1/m') # This is a bit of a made-up unit

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