import numpy as np

from openmdao.api import ExplicitComponent


class BeamNodeRHS(ExplicitComponent):
    # Righthand side of beam modeshape linear system

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

        self.inp_a = '%s_nodes' %absca
        self.inp_o = '%s_nodes' %ordin
        self.otp = 'beam_spline_%s_%s_rhs' %(ordin, self.options['level'])

        self.add_input(self.inp_a, val=np.zeros((nNode,nMode)))
        self.add_input(self.inp_o, val=np.zeros((nNode,nMode)))

        self.add_output(self.otp, val=np.zeros((nNode, nMode)))

    def setup_partials(self):
        self.declare_partials(self.otp, self.inp_a)
        self.declare_partials(self.otp, self.inp_o)

    def compute(self, inputs, outputs):
        nNode = self.nodal_data['nNode']
        nElem = self.nodal_data['nElem']
        nMode = self.nodal_data['nMode']

        for m in range(nMode):
            x = inputs[self.inp_a][:,m]
            z = inputs[self.inp_o][:,m]

            h = np.zeros(nElem)
            delta = np.zeros(nElem)
            for i in range(nElem):
                h[i] = z[i + 1] - z[i]
                if h[i] != 0.:
                    delta[i] = (x[i + 1] - x[i]) / (z[i + 1] - z[i])

            rhs = np.zeros(nNode)
            ## --- Not-a-knot boundary conditions
            # https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.296.7452&rep=rep1&type=pdf
            for i in range(1, nElem):
                rhs[i] = 3. * (h[i - 1] * delta[i] + h[i] * delta[i - 1])

            if (h[0] + h[1]) != 0.:
                rhs[0] = ((2. * h[1] + 3. * h[0]) * h[1] * delta[0] + h[0]**2. * delta[1]) / (h[0] + h[1])
            if (h[-1] + h[-2]) != 0.:
                rhs[-1] = ((2. * h[-2] + 3. * h[-1]) * h[-2] * delta[-1] + h[-1]**2. * delta[-2]) / (h[-1] + h[-2])
            
            outputs[self.otp][:,m] = rhs

    def compute_partials(self, inputs, partials):
        nNode = self.nodal_data['nNode']
        nElem = self.nodal_data['nElem']
        nMode = self.nodal_data['nMode']

        rhs_partial1 = np.zeros((nNode, nMode, nNode, nMode))
        rhs_partial2 = np.zeros((nNode, nMode, nNode, nMode))

        for m in range(nMode):
            x = inputs[self.inp_a][:,m]
            z = inputs[self.inp_o][:,m]

            h = np.zeros(nElem)
            delta = np.zeros(nElem)
            for i in range(nElem):
                h[i] = z[i + 1] - z[i]
                delta[i] = (x[i + 1] - x[i]) / (z[i + 1] - z[i])

            for i in range(1, nElem):
                rhs_partial2[i, m, i - 1, m] = -3. * delta[i] + 3. * h[i] * delta[i - 1] / h[i - 1]
                rhs_partial2[i, m, i, m] = 3. * (delta[i] + h[i - 1] * delta[i] / h[i] - delta[i - 1] - h[i] * delta[i - 1] / h[i - 1])
                rhs_partial2[i, m, i + 1, m] = -3. * h[i - 1] * delta[i] / h[i] + 3. * delta[i - 1]

                rhs_partial1[i, m, i - 1, m] = -3. * h[i] / h[i - 1]
                rhs_partial1[i, m, i, m] = -3. * h[i - 1] / h[i] + 3. * h[i] / h[i - 1]
                rhs_partial1[i, m, i + 1, m] = 3. * h[i - 1] / h[i]

            rhs_partial2[0, m, 0, m] = -(3. * h[1] * delta[0] + 2. * h[0] * delta[1]) / (h[0] + h[1]) + ((2. * h[1] + 3. * h[0]) * h[1] * delta[0] + h[0]**2. * delta[1]) / (h[0] + h[1])**2. + (2. * h[1] + 3. * h[0]) * h[1] / (h[0] + h[1]) * delta[0] / h[0]
            rhs_partial2[0, m, 1, m] = (3. * h[1] * delta[0] + 2. * h[0] * delta[1]) / (h[0] + h[1]) - (4. * h[1] + 3. * h[0]) * delta[0] / (h[0] + h[1]) - (2. * h[1] + 3 * h[0]) * h[1] / (h[0] + h[1]) * delta[0] / h[0] + h[0]**2. / (h[0] + h[1]) * delta[1] / h[1]
            rhs_partial2[0, m, 2, m] = (4. * h[1] + 3. * h[0]) * delta[0] / (h[0] + h[1]) - ((2. * h[1] + 3. * h[0]) * h[1] * delta[0] + h[0]**2. * delta[1]) / (h[0] + h[1])**2. - h[0]**2. / (h[0] + h[1]) * delta[1] / h[1]

            rhs_partial1[0, m, 0, m] = -(2. * h[1] + 3. * h[0]) * h[1] / (h[0] + h[1]) * 1. / h[0]
            rhs_partial1[0, m, 1, m] = (2. * h[1] + 3 * h[0]) * h[1] / (h[0] + h[1]) * 1. / h[0] - h[0]**2. / (h[0] + h[1]) * 1. / h[1]
            rhs_partial1[0, m, 2, m] = h[0]**2. / (h[0] + h[1]) * 1. / h[1]

            rhs_partial2[-1, m, -1, m] = (3. * h[-2] * delta[-1] + 2. * h[-1] * delta[-2]) / (h[-1] + h[-2]) - ((2. * h[-2] + 3. * h[-1]) * h[-2] * delta[-1] + h[-1]**2. * delta[-2]) / (h[-1] + h[-2])**2. - (2. * h[-2] + 3. * h[-1]) * h[-2] / (h[-1] + h[-2]) * delta[-1] / h[-1]
            rhs_partial2[-1, m, -2, m] = -(3. * h[-2] * delta[-1] + 2. * h[-1] * delta[-2]) / (h[-1] + h[-2]) + (4. * h[-2] + 3. * h[-1]) * delta[-1] / (h[-1] + h[-2]) + (2. * h[-2] + 3 * h[-1]) * h[-2] / (h[-1] + h[-2]) * delta[-1] / h[-1] - h[-1]**2. / (h[-1] + h[-2]) * delta[-2] / h[-2]
            rhs_partial2[-1, m, -3, m] = -(4. * h[-2] + 3. * h[-1]) * delta[-1] / (h[-1] + h[-2]) + ((2. * h[-2] + 3. * h[-1]) * h[-2] * delta[-1] + h[-1]**2. * delta[-2]) / (h[-1] + h[-2])**2. + h[-1]**2. / (h[-1] + h[-2]) * delta[-2] / h[-2]

            rhs_partial1[-1, m, -1, m] = (2. * h[-2] + 3. * h[-1]) * h[-2] / (h[-1] + h[-2]) * 1. / h[-1]
            rhs_partial1[-1, m, -2, m] = -(2. * h[-2] + 3 * h[-1]) * h[-2] / (h[-1] + h[-2]) * 1. / h[-1] + h[-1]**2. / (h[-1] + h[-2]) * 1. / h[-2]
            rhs_partial1[-1, m, -3, m] = -h[-1]**2. / (h[-1] + h[-2]) * 1. / h[-2]

        partials[self.otp, self.inp_a] = np.reshape(rhs_partial1, (nNode * nMode, nNode * nMode))
        partials[self.otp, self.inp_o] = np.reshape(rhs_partial2, (nNode * nMode, nNode * nMode))