import numpy as np
import scipy.linalg
from openmdao.api import ExplicitComponent


class ModeshapeBlockRotation(ExplicitComponent):
    '''
    Block DCM to make partials possible
    '''

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nElem = self.nodal_data['nElem']

        self.add_input('dir_cosines', val=np.zeros((nElem,3,3)))     
    
        self.add_output('block_rot_mat', val=np.zeros((nElem, 12, 12)))

    def setup_partials(self):
        nElem = self.nodal_data['nElem']
        nPart = (nElem * 12 * 12) * (nElem * 3 * 3)

        Hcols = np.repeat(np.arange(nElem),(3*3))

        self.declare_partials('block_rot_mat', 'dir_cosines')#, rows=np.arange(nPart), cols=np.arange(nPart))

    def compute(self, inputs, outputs):
        nElem = self.nodal_data['nElem']
        outputs['block_rot_mat'] = np.zeros((nElem, 12, 12)) 

        for i in range(nElem):
            R = inputs['dir_cosines'][i,:,:]
            RR = scipy.linalg.block_diag(R,R,R,R)
            outputs['block_rot_mat'][i,:,:] = RR

    def compute_partials(self, inputs, partials):
        nElem = self.nodal_data['nElem']

        # These are hardcoded, but that is fine for 3D model
        a = np.concatenate((np.eye(3),np.zeros((3,6))),axis=1)
        b = np.concatenate((np.zeros((3,3)),np.eye(3),np.zeros((3,3))),axis=1)
        c = np.concatenate((np.zeros((3,6)),np.eye(3)),axis=1)
        subblock = np.concatenate((a,np.zeros((9,9)),b,np.zeros((9,9)),c),axis=0)
        block = np.concatenate((subblock,np.zeros((12,9)),subblock,np.zeros((12,9)),subblock,np.zeros((12,9)),subblock),axis=0)

        blocks = []
        for i in range(nElem): blocks.append(block)

        partials['block_rot_mat', 'dir_cosines'] = scipy.linalg.block_diag(*blocks)