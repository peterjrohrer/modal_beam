import numpy as np
import openmdao.api as om

class DeltaNodes(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nElem = self.nodal_data['nElem']
        nNode = self.nodal_data['nNode']

        self.add_input('x_beamnode', val=np.zeros(nNode), units='m')
        self.add_input('y_beamnode', val=np.zeros(nNode), units='m')
        self.add_input('z_beamnode', val=np.zeros(nNode), units='m')

        self.add_output('d_node', val=np.zeros((nElem,3,1)))
        self.add_output('elem_norm', val=np.zeros(nElem))

    def setup_partials(self):
        self.declare_partials('d_node', ['x_beamnode', 'y_beamnode', 'z_beamnode'])
        self.declare_partials('elem_norm', ['x_beamnode', 'y_beamnode', 'z_beamnode'])

    def compute(self, inputs, outputs):
        nElem = self.nodal_data['nElem']
        nNode = self.nodal_data['nNode']
        
        x_nodes = inputs['x_beamnode']
        y_nodes = inputs['y_beamnode']
        z_nodes = inputs['z_beamnode']

        nodes = np.vstack((x_nodes,y_nodes,z_nodes))

        dx = np.zeros((nElem,3,1))
        le = np.zeros(nElem)
        le_test = np.zeros(nElem)

        for i in range(nElem):
            dx[i,:,:] = (nodes[:,i+1]-nodes[:,i]).reshape(3,1)
            le[i] = np.sqrt((dx[i,0,0])**2. + (dx[i,1,0])**2. + (dx[i,2,0])**2.) # element length
        
        outputs['d_node'] = dx
        outputs['elem_norm'] = le

    def compute_partials(self, inputs, partials):
        nElem = self.nodal_data['nElem']
        nNode = self.nodal_data['nNode']
        
        x_nodes = inputs['x_beamnode']
        y_nodes = inputs['y_beamnode']
        z_nodes = inputs['z_beamnode']

        nodes = np.vstack((x_nodes,y_nodes,z_nodes))

        dNode_xBeam = np.zeros((nElem, 3, nNode))
        dNode_yBeam = np.zeros((nElem, 3, nNode))
        dNode_zBeam = np.zeros((nElem, 3, nNode))

        partials['elem_norm', 'x_beamnode'] = np.zeros((nElem,nNode))
        partials['elem_norm', 'y_beamnode'] = np.zeros((nElem,nNode))
        partials['elem_norm', 'z_beamnode'] = np.zeros((nElem,nNode))

        for i in range(nElem):
            dNode_xBeam[i,0,i] = -1.
            dNode_xBeam[i,0,i+1] = 1.
            dNode_yBeam[i,1,i] = -1.
            dNode_yBeam[i,1,i+1] = 1.
            dNode_zBeam[i,2,i] = -1.
            dNode_zBeam[i,2,i+1] = 1.

            dx = (nodes[:,i+1]-nodes[:,i]).reshape(3,1)
            le = np.sqrt((dx[0])**2. + (dx[1])**2. + (dx[2])**2.)

            partials['elem_norm', 'x_beamnode'][i,i+1] += dx[0]/le
            partials['elem_norm', 'x_beamnode'][i,i] += -1.*dx[0]/le
            partials['elem_norm', 'y_beamnode'][i,i+1] += dx[1]/le
            partials['elem_norm', 'y_beamnode'][i,i] += -1.*dx[1]/le
            partials['elem_norm', 'z_beamnode'][i,i+1] += dx[2]/le
            partials['elem_norm', 'z_beamnode'][i,i] += -1.*dx[2]/le


        partials['d_node', 'x_beamnode'] = np.reshape(dNode_xBeam, ((nElem*3),(nNode)))
        partials['d_node', 'y_beamnode'] = np.reshape(dNode_yBeam, ((nElem*3),(nNode)))
        partials['d_node', 'z_beamnode'] = np.reshape(dNode_zBeam, ((nElem*3),(nNode)))
