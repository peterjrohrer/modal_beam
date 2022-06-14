import numpy as np

from openmdao.api import ExplicitComponent


class ModeshapeElemLength(ExplicitComponent):

    def initialize(self):
        self.options.declare('nNode', types=int)
        self.options.declare('nElem', types=int)

    def setup(self):
        nNode = self.options['nNode']        
        nElem = self.options['nElem']

        self.add_input('z_beamnode', val=np.zeros(nNode), units='m/m')

        self.add_output('L_mode_elem', val=np.zeros(nElem), units='m')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        z_beamnode = inputs['z_beamnode']

        N_beamelem = len(z_beamnode) - 1

        outputs['L_mode_elem'] = np.zeros(N_beamelem)

        for i in range(N_beamelem):
            outputs['L_mode_elem'][i] = (z_beamnode[i + 1] - z_beamnode[i])

    def compute_partials(self, inputs, partials):
        nNode = self.options['nNode']        
        nElem = self.options['nElem']
        z_beamnode = inputs['z_beamnode']

        N_beamelem = len(z_beamnode) - 1

        partials['L_mode_elem', 'z_beamnode'] = np.zeros((nElem, nNode))
        partials['L_mode_elem', 'Z_beam'] = np.zeros((nElem, nNode))

        for i in range(N_beamelem):
            partials['L_mode_elem', 'z_beamnode'][i, i] = -1.
            partials['L_mode_elem', 'z_beamnode'][i, i + 1] = 1.
