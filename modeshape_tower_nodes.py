import abc
import numpy as np

from openmdao.api import ExplicitComponent


class ModeshapeTowerNodes(ExplicitComponent):

    def initialize(self):
        self.options.declare('nNode', types=int)
        self.options.declare('nElem', types=int)

    def setup(self):
        nNode = self.options['nNode']        
        nElem = self.options['nElem']

        self.add_input('Z_tower', val=np.zeros(nNode), units='m')

        self.add_output('z_towernode', val=np.zeros(nNode), units='m/m')
        self.add_output('z_towerelem', val=np.zeros(nElem), units='m/m')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        nNode = self.options['nNode']        
        nElem = self.options['nElem']
        hub_height = inputs['Z_tower'][-1]
        
        z_node = inputs['Z_tower']/hub_height
        z_elem = np.zeros(nElem)

        h = np.zeros(nElem)
        for i in range(len(h)):
            h[i] = (z_node[i + 1] - z_node[i])/2.
            z_elem[i] = z_node[i]+h[i]

        outputs['z_towernode'] = z_node
        outputs['z_towerelem'] = z_elem

    def compute_partials(self, inputs, partials):
        nNode = self.options['nNode']        
        nElem = self.options['nElem']
        Z_tower = inputs['Z_tower']
        hub_height = inputs['Z_tower'][-1]
        
        partials['z_towernode', 'Z_tower'] = np.zeros((nNode,nNode))
        
        for i in range(11):
            partials['z_towernode', 'Z_tower'][i,i] += 1./hub_height 
            partials['z_towernode', 'Z_tower'][i,-1] += -1. * Z_tower[i]/(hub_height*hub_height)

        partials['z_towernode', 'Z_tower'][-1,-1] = 0.

