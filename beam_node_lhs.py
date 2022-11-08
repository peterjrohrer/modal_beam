import numpy as np
from openmdao.api import ExplicitComponent

class BeamNodeLHS(ExplicitComponent):
    # Lefthand side of beam modeshape linear system

    def initialize(self):
        self.options.declare('nodal_data', types=dict)
        self.options.declare('key', types=str)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nNode = self.nodal_data['nNode']
        nDOF_tot = self.nodal_data['nDOF_tot']
        nMode = self.nodal_data['nMode']
        key = self.key = self.options['key']
        self.inp = '%s_nodes' %key
        self.otp = 'beam_spline_%s_lhs' %key

        self.add_input(self.inp, val=np.zeros((nNode,nMode)), units='m/m')

        self.add_output(self.otp, val=np.zeros((nNode, nNode, nMode)), units='m/m')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        nNode = self.nodal_data['nNode']
        nElem = self.nodal_data['nElem']
        nMode = self.nodal_data['nMode']

        for m in range(nMode):
            z = inputs[self.inp][:,m]
            h = np.zeros(nElem)

            for i in range(nElem):
                h[i] = z[i + 1] - z[i]

            lhs = np.zeros((nNode, nNode))
            ## --- Not-a-knot boundary conditions
            # https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.296.7452&rep=rep1&type=pdf
            for i in range(1, nElem):
                lhs[i, i] = 2. * (h[i] + h[i - 1])
                lhs[i, i - 1] = h[i]
                lhs[i, i + 1] = h[i - 1]

            lhs[0, 0] = h[1]
            lhs[0, 1] = h[0] + h[1]
            lhs[-1, -1] = h[-2]
            lhs[-1, -2] = h[-1] + h[-2]

            outputs[self.otp][:,:,m] = lhs

    def compute_partials(self, inputs, partials):
        nNode = self.nodal_data['nNode']
        nElem = self.nodal_data['nElem']
        nMode = self.nodal_data['nMode']
        
        lhs_partial = np.zeros((nNode * nNode, nMode, nNode, nMode))

        for m in range(nMode):
            for i in range(1, nElem):
                lhs_partial[(i*nNode)-1+i, m, i, m] = -1.
                lhs_partial[(i*nNode)-1+i, m, i+1, m] = 1.
                lhs_partial[(i*nNode)+i, m, i-1, m] = -2.
                lhs_partial[(i*nNode)+i, m, i, m] = 0.
                lhs_partial[(i*nNode)+i, m, i+1, m] = 2.
                lhs_partial[(i*nNode)+1+i, m, i-1, m] = -1.
                lhs_partial[(i*nNode)+1+i, m, i, m] = 1.

            lhs_partial[0, m, 1, m] = -1.
            lhs_partial[0, m, 2, m] = 1.
            lhs_partial[1, m, 0, m] = -1.
            lhs_partial[1, m, 2, m] = 1.
            lhs_partial[-1, m, -3, m] = -1.
            lhs_partial[-1, m, -2, m] = 1.
            lhs_partial[-2, m, -3, m] = -1.
            lhs_partial[-2, m, -1, m] = 1.

        partials[self.otp, self.inp] = np.reshape(lhs_partial, (nNode * nNode * nMode, nNode * nMode))
