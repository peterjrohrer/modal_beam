import numpy as np

from openmdao.api import ExplicitComponent


class ModeshapeElemLength(ExplicitComponent):

    def initialize(self):
        self.options.declare('nNode', types=int)
        self.options.declare('nElem', types=int)

    def setup(self):
        nNode = self.options['nNode']        
        nElem = self.options['nElem']

        self.add_input('z_towernode', val=np.zeros(nNode), units='m/m')

        self.add_output('L_mode_elem', val=np.zeros(nElem), units='m')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        z_towernode = inputs['z_towernode']

        N_towerelem = len(z_towernode) - 1

        outputs['L_mode_elem'] = np.zeros(N_towerelem)

        for i in range(N_towerelem):
            outputs['L_mode_elem'][i] = (z_towernode[i + 1] - z_towernode[i])

    def compute_partials(self, inputs, partials):
        nNode = self.options['nNode']        
        nElem = self.options['nElem']
        z_towernode = inputs['z_towernode']

        N_towerelem = len(z_towernode) - 1

        partials['L_mode_elem', 'z_towernode'] = np.zeros((nElem, nNode))
        partials['L_mode_elem', 'Z_tower'] = np.zeros((nElem, nNode))

        for i in range(N_towerelem):
            partials['L_mode_elem', 'z_towernode'][i, i] = -1.
            partials['L_mode_elem', 'z_towernode'][i, i + 1] = 1.
