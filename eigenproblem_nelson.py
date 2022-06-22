import numpy as np
import scipy.linalg
import openmdao.api as om

class EigenproblemNelson(om.ExplicitComponent):
    # Full solution to eigenvalue problem

    def initialize(self):
        self.options.declare('nDOF', types=int)

    def setup(self):
        nDOF = self.options['nDOF']

        self.add_input('A_eig', val=np.zeros((nDOF, nDOF)))

        self.add_output('eig_vectors', val=np.zeros((nDOF, nDOF)))
        self.add_output('eig_vals', val=np.zeros((nDOF, nDOF)))

    def compute(self, inputs, outputs):
        A = inputs['A_eig']

        vals, lvecs, rvecs = scipy.linalg.eig(A, left=True, right=True)

        # if any(np.imag(vals) != 0.) :
        #     raise om.AnalysisError('Imaginary eigenvalues')
        # if any(np.real(vals) < 0.) :
        #     raise om.AnalysisError('Negative eigenvalues')

        outputs['eig_vectors'] = rvecs
        outputs['eig_vals'] = np.diag(np.real(vals))

        self.lvecs = lvecs
        self.rvecs = rvecs
        self.vals = np.diag(np.real(vals))
    
    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        nDOF = self.options['nDOF']        
        lvecs= self.lvecs
        vecs = self.rvecs
        vals = self.vals
        
        F = np.zeros((nDOF, nDOF))
        C = np.zeros((nDOF, nDOF))
        V = np.zeros((nDOF, nDOF))
        M = nDOF*np.eye(nDOF)

        if mode == 'rev':    
            if 'eig_vectors' in d_outputs:
                if 'A_eig' in d_inputs:
                    d_inputs['A_eig'] += vecs @ ((vals.T @ d_outputs['eig_vals']) + np.multiply(F,(vecs.T @ d_outputs['eig_vectors'])) - np.multiply((0.5*np.eye(nDOF)), (vecs.T @ d_outputs['eig_vectors']))) @ vecs.T
            if 'eig_vals' in d_outputs:
                if 'A_eig' in d_inputs:
                    d_inputs['A_eig'] += vecs @ ((vals.T @ d_outputs['eig_vals']) + np.multiply(F,(vecs.T @ d_outputs['eig_vectors'])) - np.multiply((0.5*np.eye(nDOF)), (vecs.T @ d_outputs['eig_vectors']))) @ vecs.T

        elif mode == 'fwd':
            if 'eig_vectors' in d_outputs:
                if 'A_eig' in d_inputs:
                    for i in range(nDOF):
                        F[:,i] += (vecs[:,i] * (lvecs[:,i].T @ d_inputs['A_eig'] @ vecs[:,i])) - (d_inputs['A_eig'] @ vecs[:,i])
                        for k in range(nDOF):
                            if k != i:
                                C[k,i] += (lvecs[:,i] @ F[:,i])/(vals[k,k]-vals[i,i])
                    for i in range(nDOF):
                        for k in range(nDOF):
                            V[:,i] += C[k,i] * vecs[:,k]
                    
                    print('a')

                    # d_outputs['eig_vectors'] += (vals @ np.multiply(F, ((-1. * vecs.T @ vecs @ vals) + (vecs.T @ d_inputs['A_eig'] @ vecs)))) - (0.5 * vecs @ np.multiply(np.eye(nDOF), (vecs.T @ d_inputs['A_eig'] @ vecs)))
            if 'eig_vals' in d_outputs:
                if 'A_eig' in d_inputs:
                    d_outputs['eig_vals'] += lvecs.T @ d_inputs['A_eig'] @ vecs