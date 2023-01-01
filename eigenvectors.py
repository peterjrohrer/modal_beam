import numpy as np
import scipy.linalg
import openmdao.api as om

class Eigenvecs(om.ExplicitComponent):
    # Solution to eigenvalue problem with eigenvectors for manipulation

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nDOF_r = self.nodal_data['nDOF_r']
        nDOF_tot = self.nodal_data['nDOF_tot']

        self.add_input('Ar_eig', val=np.ones((nDOF_r, nDOF_r)))

        self.add_output('Q_raw', val=np.ones((nDOF_r, nDOF_r)))
        self.add_output('sort_idx', val=np.zeros(nDOF_r))

    # def setup_partials(self):
    #     self.declare_partials('Q_raw', 'Ar_eig')

    def compute(self, inputs, outputs):
        nDOF = self.nodal_data['nDOF_r']
        A = inputs['Ar_eig']
                
        # --- To match Giles (2008) paper
        # https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
        D, Q = scipy.linalg.eig(a=A, b=None, left=False, right=True)

        # if np.imag(Q).any() > 0.:
        #     raise om.AnalysisError('Imaginary eigvectors')
        # elif np.imag(D).any() > 0.:
        #     raise om.AnalysisError('Imaginary eigvalue')
        
        # Sort 
        I = np.argsort(D)
        # Q = Q[:,I]
        # D = D[I]

        outputs['Q_raw'] = Q
        outputs['sort_idx'] = I

        self.Q = Q
        self.D = D # Must keep complex eigenvalues to have correct derivatives    

    # def compute_partials(self, inputs, partials, discrete_inputs=None):
    #     nDOF = self.nodal_data['nDOF_r']
    #     A = inputs['Ar_eig']

    #     Q = self.Q
    #     D = self.D

    #     # dv_dQ = np.zeros((nDOF,nDOF,nDOF,nDOF),dtype=complex)
    #     # dB_db = np.einsum('lj,ki->klij',np.eye(nDOF),np.eye(nDOF))

    #     # for i in range(nDOF):
    #     #     for j in range(nDOF):
    #     #         for k in range(nDOF):
    #     #             for l in range(nDOF):
    #     #                 dv_dQ[i,j,k,l] = np.sum(self.F_matrix[i,j] * np.inner(dB_db[k,l]@Q[:,i],Q[:,j]) * Q[:,j])
        
    #     dQ_dA = np.zeros((nDOF,nDOF,nDOF,nDOF), dtype=complex)
    #     for i in range(nDOF):
    #         for j in range(nDOF):
    #             dA = np.zeros_like(A)
    #             dA[i, j] = 1.

    #             P = scipy.linalg.solve(Q, np.dot(dA, Q))
    #             dU = np.dot(Q, (self.F_matrix * P))
    #             dQ_dA

    #     partials['Q_raw', 'Ar_eig'] = np.reshape(dQ_dA, (nDOF*nDOF, nDOF*nDOF))

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        N_DOF = self.nodal_data['nDOF_r']
        Q = self.Q
        D = self.D
        A = inputs['Ar_eig']

        E = (np.tile(D,(N_DOF,1)) - np.tile(D,(N_DOF,1)).T)
        E[np.arange(N_DOF),np.arange(N_DOF)] = 1. # avoid divide by zero
        F = 1./E
        F[np.arange(N_DOF),np.arange(N_DOF)] = 0.

        ## Based on Giles (2008), https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
        ##TODO look into if sorting affects this?
        if mode == 'rev':    
            if 'Q_raw' in d_outputs:
                if 'Ar_eig' in d_inputs:
                    d_inputs['Ar_eig'] = scipy.linalg.inv(Q).T @ (np.multiply(F, (Q.T @ d_outputs['Q_raw']))) @ Q.T

        elif mode == 'fwd':
            if 'Q_raw' in d_outputs:
                if 'Ar_eig' in d_inputs:
                    d_outputs['Q_raw'] = Q @ np.multiply(F, (np.linalg.inv(Q) @ d_inputs['Ar_eig'] @ Q))