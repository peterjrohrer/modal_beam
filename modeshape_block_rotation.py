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
    
        self.add_output('block_rot_mat', val=np.zeros((nElem, 12, 12)))

    def compute(self, inputs, outputs):
        nElem = self.nodal_data['nElem']
        DCM = self.nodal_data['DCM']
        outputs['block_rot_mat'] = np.zeros((nElem, 12, 12)) 

        for i in range(nElem):
            R = DCM[i,:,:]
            RR = scipy.linalg.block_diag(R,R,R,R)
            outputs['block_rot_mat'][i,:,:] = RR