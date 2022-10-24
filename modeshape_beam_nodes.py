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

    def setup_partials(self):
        self.declare_partials('z_beamnode', 'Z_beam')
        self.declare_partials('z_beamelem', 'Z_beam')

    def compute(self, inputs, outputs):
        nNode = self.options['nNode']        
        nElem = self.options['nElem']

        outputs['z_beamnode'] = inputs['Z_beam']
        outputs['z_beamelem'] = np.zeros(nElem)

        h = np.zeros(nElem)
        for i in range(len(h)):
            h[i] = inputs['Z_beam'][i + 1] - inputs['Z_beam'][i]
            outputs['z_beamelem'][i] = inputs['Z_beam'][i]+(h[i]/2.)

    def compute_partials(self, inputs, partials):
        nNode = self.options['nNode']        
        nElem = self.options['nElem']

        z_node = inputs['Z_beam']
        z_elem = np.zeros(nElem)

        partials['z_beamnode', 'Z_beam'] = np.eye(nNode)
        partials['z_beamelem', 'Z_beam'] = np.zeros((nElem,nNode))
        
        h = np.zeros(nElem)
        for i in range(len(h)):
            h[i] = z_node[i + 1] - z_node[i]
            z_elem[i] = z_node[i]+(h[i]/2.)
            partials['z_beamelem', 'Z_beam'][i,i] += 0.5
            if not i == 0 :
                partials['z_beamelem', 'Z_beam'][i-1,i] += 0.5
        
        partials['z_beamelem', 'Z_beam'][-1,-1] += 0.5

