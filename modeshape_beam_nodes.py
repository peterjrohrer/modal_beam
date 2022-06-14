import abc
import numpy as np

from openmdao.api import ExplicitComponent


class ModeshapeBeamNodes(ExplicitComponent):

    def initialize(self):
        self.options.declare('nNode', types=int)
        self.options.declare('nElem', types=int)

    def setup(self):
        nNode = self.options['nNode']        
        nElem = self.options['nElem']

        self.add_input('Z_beam', val=np.zeros(nNode), units='m')

        self.add_output('z_beamnode', val=np.zeros(nNode), units='m/m')
        self.add_output('z_beamelem', val=np.zeros(nElem), units='m/m')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        nNode = self.options['nNode']        
        nElem = self.options['nElem']
        
        z_node = inputs['Z_beam']
        z_elem = np.zeros(nElem)

        h = np.zeros(nElem)
        for i in range(len(h)):
            h[i] = (z_node[i + 1] - z_node[i])/2.
            z_elem[i] = z_node[i]+h[i]

        outputs['z_beamnode'] = z_node
        outputs['z_beamelem'] = z_elem

    def compute_partials(self, inputs, partials):
        nNode = self.options['nNode']        
        nElem = self.options['nElem']
        Z_beam = inputs['Z_beam']
        
        partials['z_beamnode', 'Z_beam'] = np.zeros((nNode,nNode))
        
        for i in range(11):
            partials['z_beamnode', 'Z_beam'][i,i] += 1.
            partials['z_beamnode', 'Z_beam'][i,-1] += -1. * Z_beam[i]

        partials['z_beamnode', 'Z_beam'][-1,-1] = 0.

