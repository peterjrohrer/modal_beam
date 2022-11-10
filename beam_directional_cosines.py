""" Generate element Direction cosine matricse (DCM) 
from a set of ordered node coordinates defining a beam mean line

INPUTS:
    x_nodes: (nNodes)
    y_nodes: (nNodes)
    z_nodes: (nNodes)
    phi (optional): nNodes angles about mean line to rotate the section axes
OUTPUTS:
    DCM:  3 x 3 x (nNodes-1)
"""

import numpy as np
import openmdao.api as om

class BeamDirectionalCosines(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nElem = self.nodal_data['nElem']
        nNode = self.nodal_data['nNode']

        self.add_input('x_beamnode', val=np.zeros(nNode), units='m')
        self.add_input('y_beamnode', val=np.zeros(nNode), units='m')
        self.add_input('z_beamnode', val=np.zeros(nNode), units='m')

        self.add_output('dir_cosines', val=np.zeros((nElem,3,3)))

    def setup_partials(self):
        self.declare_partials('dir_cosines', ['x_beamnode', 'y_beamnode', 'z_beamnode'])

    def compute(self, inputs, outputs):
        nElem = self.nodal_data['nElem']
        nNode = self.nodal_data['nNode']

        def null(a, rtol=1e-5):
            u, s, v = np.linalg.svd(a)
            rank = (s > rtol*s[0]).sum()
            return v[rank:].T.copy()
        
        x_nodes = inputs['x_beamnode']
        y_nodes = inputs['y_beamnode']
        z_nodes = inputs['z_beamnode']

        nodes = np.vstack((x_nodes,y_nodes,z_nodes))
        DCM = np.zeros((nElem,3,3))

        for i in np.arange(nElem):
            dx= (nodes[:,i+1]-nodes[:,i]).reshape(3,1)
            le = np.linalg.norm(dx) # element length
            # tangent vector
            e1 = dx/le 
            if i==0:
                e1_last = e1
                e2_last = null(e1.T)[:,0].reshape(3,1) # x,z-> y , y-> -x 
            # normal vector
            de1 = e1 - e1_last
            if np.linalg.norm(de1)<1e-8:
                e2 = e2_last
            else:
                e2 = de1/np.linalg.norm(de1) 
            # Third vector
            e3=np.cross(e1.ravel(),e2.ravel()).reshape(3,1)
            DCM[i,:,:]= np.column_stack((e1,e2,e3)).T
            e1_last= e1
            e2_last= e2
        
        outputs['dir_cosines'] = DCM

    def compute_partials(self, inputs, partials):
        nElem = self.nodal_data['nElem']
        nNode = self.nodal_data['nNode']

        partials['dir_cosines', 'x_beamnode'] = np.zeros(((nElem*3*3),(nNode)))
        partials['dir_cosines', 'y_beamnode'] = np.zeros(((nElem*3*3),(nNode)))
        partials['dir_cosines', 'z_beamnode'] = np.zeros(((nElem*3*3),(nNode)))

        ##TODO define these partials
