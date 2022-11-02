import numpy as np
from scipy.sparse import linalg
from scipy.linalg import det

from openmdao.api import ExplicitComponent


class ModeshapeDisp(ExplicitComponent):

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nNode = self.nodal_data['nNode']
        nDOF_tot = self.nodal_data['nDOF_tot']
        nMode = self.nodal_data['nMode']

        self.add_input('Q', val=np.zeros((nDOF_tot,nMode)))
        self.add_input('x_beamnode', val=np.zeros(nNode), units='m')
        self.add_input('y_beamnode', val=np.zeros(nNode), units='m')
        self.add_input('z_beamnode', val=np.zeros(nNode), units='m')

        self.add_output('x_nodes', val=np.zeros((nNode,nMode)), units='m')
        self.add_output('y_nodes', val=np.zeros((nNode,nMode)), units='m')
        self.add_output('z_nodes', val=np.zeros((nNode,nMode)), units='m')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        nElem = self.nodal_data['nElem']
        nDOF_tot = self.nodal_data['nDOF_tot']
        nMode = self.nodal_data['nMode']
        nDOFperNode = self.nodal_data['nDOFperNode']

        Q = inputs['Q']

        x_nodes = np.tile(inputs['x_beamnode'],(nMode,1)).T
        y_nodes = np.tile(inputs['y_beamnode'],(nMode,1)).T
        z_nodes = np.tile(inputs['z_beamnode'],(nMode,1)).T
        th_x_nodes = np.zeros_like(x_nodes)
        th_y_nodes = np.zeros_like(x_nodes)
        th_z_nodes = np.zeros_like(x_nodes)
        
        for j in range(nMode):
            eigvec = Q[:,j]
            x_nodes[:,j] += eigvec[0:(nElem + 1) * nDOFperNode:nDOFperNode]
            y_nodes[:,j] += eigvec[1:(nElem + 2) * nDOFperNode:nDOFperNode]
            z_nodes[:,j] += eigvec[2:(nElem + 3) * nDOFperNode:nDOFperNode]
            th_x_nodes[:,j] += eigvec[3:(nElem + 4) * nDOFperNode:nDOFperNode]
            th_y_nodes[:,j] += eigvec[4:(nElem + 6) * nDOFperNode:nDOFperNode]
            th_z_nodes[:,j] += eigvec[5:(nElem + 6) * nDOFperNode:nDOFperNode]

        outputs['x_nodes'] = x_nodes
        outputs['y_nodes'] = y_nodes
        outputs['z_nodes'] = z_nodes

    def compute_partials(self, inputs, partials):
        nElem = self.nodal_data['nElem']
        nDOF_tot = self.nodal_data['nDOF_tot']
        nMode = self.nodal_data['nMode']
        nDOFperNode = self.nodal_data['nDOFperNode']