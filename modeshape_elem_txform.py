from telnetlib import DM
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
    
        self.add_output('mel', val=np.zeros((nElem, 12, 12)), units='kg')
        self.add_output('kel', val=np.zeros((nElem, 12, 12)), units='N/m')

    def setup_partials(self):
        nElem = self.nodal_data['nElem']

        self.declare_partials('kel', 'kel_loc')
        self.declare_partials('mel', 'mel_loc')

    def compute(self, inputs, outputs):
        nElem = self.nodal_data['nElem']

        mel = inputs['mel_loc']
        kel = inputs['kel_loc']
        Me = np.zeros((nElem, 12, 12))
        Ke = np.zeros((nElem, 12, 12))
        
        for i in range(nElem):
            ## Element in global coord
            RR = self.nodal_data['RR'][i,:,:]
            # # Original
            # Me[i,:,:] = np.transpose(RR).dot(mel[i,:,:]).dot(RR)
            # Ke[i,:,:] = np.transpose(RR).dot(kel[i,:,:]).dot(RR)
            # Cleaner implementation
            Me[i,:,:] = RR.T @ mel[i,:,:] @ RR
            Ke[i,:,:] = RR.T @ kel[i,:,:] @ RR

        outputs['mel'] = Me
        outputs['kel'] = Ke

    def compute_partials(self, inputs, partials):
        nElem = self.nodal_data['nElem']
        
        blocks = []
        for k in range(nElem):
            RR = self.nodal_data['RR'][k,:,:]
            blocks.append(np.kron(RR.T,RR.T))

        partials['mel', 'mel_loc'] = scipy.linalg.block_diag(*blocks)
        partials['kel', 'kel_loc'] = scipy.linalg.block_diag(*blocks)