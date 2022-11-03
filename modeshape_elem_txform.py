import numpy as np
import scipy.linalg
from openmdao.api import ExplicitComponent


class ModeshapeElemTransform(ExplicitComponent):
    '''
    Apply transformation to each element's local matrices
    '''

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nElem = self.nodal_data['nElem']
        nDOF = self.nodal_data['nDOF_tot']

        self.add_input('mel_loc', val=np.zeros((nElem, 12, 12)), units='kg')        
        self.add_input('kel_loc', val=np.zeros((nElem, 12, 12)), units='N/m')        
        self.add_input('dir_cosines', val=np.zeros((nElem,3,3)))     
    
        self.add_output('mel', val=np.zeros((nElem, 12, 12)), units='kg')
        self.add_output('kel', val=np.zeros((nElem, 12, 12)), units='N/m')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        nElem = self.nodal_data['nElem']

        mel = inputs['mel_loc']
        kel = inputs['kel_loc']
        Me = np.zeros((nElem, 12, 12))
        Ke = np.zeros((nElem, 12, 12))
        
        for i in range(nElem):
            ## Element in global coord
            R = inputs['dir_cosines'][i,:,:]
            RR = scipy.linalg.block_diag(R,R,R,R)
            Me[i,:,:] = np.transpose(RR).dot(mel[i,:,:]).dot(RR)
            Ke[i,:,:] = np.transpose(RR).dot(kel[i,:,:]).dot(RR)

        outputs['mel'] = Me

    def compute_partials(self, inputs, partials):
        nElem = self.nodal_data['nElem']
                
        partials['mel', 'mel_loc'] = np.eye((nElem*12*12))
        partials['kel', 'kel_loc'] = np.eye((nElem*12*12))
        partials['mel', 'dir_cosines'] = np.zeros(((nElem*12*12), (nElem*3*3)))
        partials['kel', 'dir_cosines'] = np.zeros(((nElem*12*12), (nElem*3*3)))
