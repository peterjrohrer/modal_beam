import numpy as np
from scipy.sparse import linalg
from scipy.linalg import det

from openmdao.api import ExplicitComponent


class ModeshapeDisp(ExplicitComponent):

    def initialize(self):
        self.options.declare('nNode', types=int)
        self.options.declare('nElem', types=int)
        self.options.declare('nDOF', types=int)

    def setup(self):
        nNode = self.options['nNode']
        nElem = self.options['nElem']
        nDOF = self.options['nDOF']

        self.add_input('eig_vector', val=np.zeros(nDOF), units='m')

        self.add_output('x_beamnode', val=np.zeros(nNode), units='m/m')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        nNode = self.options['nNode']
        nElem = self.options['nElem']
        nDOF = self.options['nDOF']

        x_beamnode = np.zeros(nNode)
        rot_beamnode = np.zeros(nNode)

        x_beamnode[1:] = inputs['eig_vector'][0:(nElem + 1) * 2:2]
        rot_beamnode[1:] = inputs['eig_vector'][1:(nElem + 2) * 2:2]

        x_node = np.hstack((x_beamnode[:-1], x_beamnode))
        max_x_node_idx = np.argmax(np.abs(x_node))
        max_x_node = x_node[max_x_node_idx]

        outputs['x_beamnode'] = x_beamnode / max_x_node

        # print('Spar rot [deg]:', *np.round(rot_sparnode*(180/np.pi),5), sep=', ')
        # print('Beam rot [deg]:', *np.round(rot_beamnode*(180/np.pi),5), sep=', ')

    def compute_partials(self, inputs, partials):
        nNode = self.options['nNode']
        nElem = self.options['nElem']
        nDOF = self.options['nDOF']

        x_beamnode = inputs['eig_vector'][0:(nElem + 1) * 2:2]
        rot_beamnode = inputs['eig_vector'][1:(nElem + 2) * 2:2]

        max_x_node = np.max(np.abs(x_beamnode))
        max_x_node_idx = np.argmax(np.abs(x_beamnode))

        partials['x_beamnode', 'eig_vector'] = np.zeros((nNode, nDOF))
    
        x_node_partial = np.zeros((nNode,nDOF))
        for i in range(nElem) :
            x_node_partial[i, 2 * i] += 1. / max_x_node

        partials['x_beamnode', 'eig_vector'] = x_node_partial[0:nNode,...]
        # Adjust partials for normalization
        for i in range(nElem) :
            partials['x_beamnode', 'eig_vector'][i, 2*max_x_node_idx] += -1 * x_beamnode[i] / (max_x_node**2)