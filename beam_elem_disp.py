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

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        nNode = self.options['nNode']
        nElem = self.options['nElem']

        z = inputs['z_beamnode']
        # z_elem = inputs['z_beamelem']
        x = inputs['x_beamnode']
        x_d = inputs['x_d_beamnode']

        h = np.zeros(nElem)
        # s = np.zeros(nElem)
        for i in range(nElem):
            h[i] = z[i + 1] - z[i]
            # s[i] = z_elem[i] - z[i]

        outputs['x_beamelem'] = np.zeros(nElem)

        for i in range(nElem):
            # # Spline Interpolation, pchip from page 29 of https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.296.7452&rep=rep1&type=pdf
            # # Partials for this are really tough
            # outputs['x_beamelem'][i] = (((3.*h[i]*s[i]*s[i] - 2.*s[i]**3)/(h[i]**3))*x[i+1]) + (((h[i]**3 - 3.*h[i]*s[i]*s[i] + 2.*s[i]**3)/(h[i]**3))*x[i]) + (((s[i]*s[i]*(s[i]-h[i]))/(h[i]**2))*x_d[i+1]) + (((s[i]*(s[i]-h[i])*(s[i]-h[i]))/(h[i]**2))*x_d[i])
            
            # Only using one of the derivatives
            # outputs['x_beamelem'][i] = x[i] + (x_d[i]*s[i]) + (((s[i]*s[i])/(h[i]*h[i]))*((x[i+1]-x[i]) - (x_d[i]*h[i])))

            # Using no element locations
            outputs['x_beamelem'][i] = (x[i+1] + x[i])/2. - (1./8.)*h[i]*(x_d[i+1] - x_d[i])
            
    def compute_partials(self, inputs, partials):
        nNode = self.options['nNode']
        nElem = self.options['nElem']
        
        z = inputs['z_beamnode']
        # z_elem = inputs['z_beamelem']
        x = inputs['x_beamnode']
        x_d = inputs['x_d_beamnode']

        h = np.zeros(nElem)
        # s = np.zeros(nElem)
        for i in range(nElem):
            h[i] = z[i + 1] - z[i]
            # s[i] = z_elem[i] - z[i]


        partials['x_beamelem', 'z_beamnode'] = np.zeros((nElem, nNode))
        # partials['x_beamelem', 'z_beamelem'] = np.zeros((nElem, nElem))
        partials['x_beamelem', 'x_beamnode'] = np.zeros((nElem, nNode))
        partials['x_beamelem', 'x_d_beamnode'] = np.zeros((nElem, nNode))
        
        x_beamelem = np.zeros(nElem)
        for i in range(nElem):
            # Using no derivatives
            x_beamelem[i] = (x[i+1] + x[i])/2. - (1./8.)*h[i]*(x_d[i+1] - x_d[i])

            partials['x_beamelem', 'x_beamnode'][i, i] = (1./2.)
            partials['x_beamelem', 'x_beamnode'][i, i + 1] = (1./2.)

            partials['x_beamelem', 'x_d_beamnode'][i, i] = (1./8.)*h[i]
            partials['x_beamelem', 'x_d_beamnode'][i, i+1] = (-1./8.)*h[i]

            partials['x_beamelem', 'z_beamnode'][i, i] = (1./8.) * (x_d[i+1] - x_d[i])
            partials['x_beamelem', 'z_beamnode'][i, i + 1] = (-1./8.) * (x_d[i+1] - x_d[i])
