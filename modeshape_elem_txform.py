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
        self.add_input('block_rot_mat', val=np.zeros((nElem,12,12)))     
    
        self.add_output('mel', val=np.zeros((nElem, 12, 12)), units='kg')
        self.add_output('kel', val=np.zeros((nElem, 12, 12)), units='N/m')

    def setup_partials(self):
        nElem = self.nodal_data['nElem']
        nPart = nElem * 12 * 12

        Hcols = np.repeat(np.arange(nElem),(3*3))

        self.declare_partials('kel', 'kel_loc')
        self.declare_partials('mel', 'mel_loc')
        self.declare_partials('mel', 'block_rot_mat')
        self.declare_partials('kel', 'block_rot_mat')

    def compute(self, inputs, outputs):
        nElem = self.nodal_data['nElem']

        mel = inputs['mel_loc']
        kel = inputs['kel_loc']
        Me = np.zeros((nElem, 12, 12))
        Ke = np.zeros((nElem, 12, 12))
        
        for i in range(nElem):
            ## Element in global coord
            RR = inputs['block_rot_mat'][i,:,:]
            # # Original
            # Me[i,:,:] = np.transpose(RR).dot(mel[i,:,:]).dot(RR)
            # Ke[i,:,:] = np.transpose(RR).dot(kel[i,:,:]).dot(RR)
            # Cleaner implementation
            Me[i,:,:] = RR.T @ mel[i,:,:] @ RR
            Ke[i,:,:] = RR.T @ kel[i,:,:] @ RR

        outputs['mel'] = Me
        outputs['kel'] = Ke

    def compute_partials(self, inputs, partials):
        ## TODO these partials are roughly correct
        nElem = self.nodal_data['nElem']
        nPart = nElem * 12 * 12
        
        # --- Somewhat hacky solution
        partials['mel', 'block_rot_mat'] = np.zeros((nPart,nPart))
        partials['kel', 'block_rot_mat'] = np.zeros((nPart,nPart))

        blocks_M = []
        blocks_K = []
        blocks = []
        for k in range(nElem):
            mel = inputs['mel_loc'][k,:,:]
            kel = inputs['kel_loc'][k,:,:]
            RR = inputs['block_rot_mat'][k,:,:]

            blocks.append(np.kron(RR.T,RR.T))
            
            rows_m = []
            rows_k = []
            for i in range(12):
                row_m = []
                row_k = []
                for j in range(12):
                    block_m = np.zeros((12,12))
                    block_k = np.zeros((12,12))
                    E = np.zeros_like(RR)
                    E[j,i] += 1.
                    block_m += (RR.T @ mel @ E)
                    block_k += (RR.T @ kel @ E)
                    row_m.append(block_m)
                    row_k.append(block_k)
                row_m_concat = np.concatenate(row_m,axis=1)
                row_k_concat = np.concatenate(row_k,axis=1)
                rows_m.append(row_m_concat)
                rows_k.append(row_k_concat)
            blocked_m = np.concatenate(rows_m,axis=0)
            blocked_k = np.concatenate(rows_k,axis=0)
            blocked_m += np.kron((mel@RR).T,np.eye(12))
            blocked_k += np.kron((kel@RR).T,np.eye(12))
            
            blocks_M.append(blocked_m)
            blocks_K.append(blocked_k)

        partials['mel', 'block_rot_mat'] = scipy.linalg.block_diag(*blocks_M)
        partials['kel', 'block_rot_mat'] = scipy.linalg.block_diag(*blocks_K)
        partials['mel', 'mel_loc'] = scipy.linalg.block_diag(*blocks)
        partials['kel', 'kel_loc'] = scipy.linalg.block_diag(*blocks)

        # # --- Theoretical Solution
        # partials['mel', 'block_rot_mat'] = np.zeros((nPart,nPart))
        # partials['kel', 'block_rot_mat'] = np.zeros((nPart,nPart))

        # blocks_M = []
        # blocks_K = []
        # for k in range(nElem):
        #     block_m = np.zeros((144,144))
        #     block_k = np.zeros((144,144))
        #     for i in range(12):
        #         for j in range(12):
        #             idx = (i*12) + j
        #             dM_dR = np.zeros((12,12))
        #             dK_dR = np.zeros((12,12))
        #             for m in range(12):
        #                 for n in range(12):
        #                     dM_dR[m,n] += 2. * mel[k,i,j] * inputs['block_rot_mat'][k,m,n] 
        #                     dK_dR[m,n] += 2. * kel[k,i,j] * inputs['block_rot_mat'][k,m,n] 
        #                     # dM_dR[m,n] += mel[k,i,j] * inputs['block_rot_mat'][k,m,n] 
        #                     # dK_dR[m,n] += kel[k,i,j] * inputs['block_rot_mat'][k,m,n] 
        #                     # dM_dR[m,n] += inputs['block_rot_mat'][k,n,m] * mel[k,i,j] 
        #                     # dK_dR[m,n] += inputs['block_rot_mat'][k,n,m] * kel[k,i,j] 
        #             ix0 = i * 12
        #             ix1 = (i+1) * 12
        #             ix2 = j * 12
        #             ix3 = (j+1) *12
        #             block_m[ix0:ix1,ix2:ix3] += dM_dR
        #             block_k[ix0:ix1,ix2:ix3] += dK_dR

        #     blocks_M.append(block_m)
        #     blocks_K.append(block_k)

        # partials['mel', 'block_rot_mat'] = scipy.linalg.block_diag(*blocks_M)
        # partials['kel', 'block_rot_mat'] = scipy.linalg.block_diag(*blocks_K)