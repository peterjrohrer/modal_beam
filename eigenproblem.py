import numpy as np
import scipy.linalg
import openmdao.api as om

class Eigenproblem(om.ExplicitComponent):
    # Full solution to eigenvalue problem

    def initialize(self):
        self.options.declare('nDOF', types=int)

    def setup(self):
        self.N_DOF = self.options['nDOF']

        self.add_input('M_mode', val=np.ones((self.N_DOF, self.N_DOF)), units='kg')
        self.add_input('K_mode', val=np.ones((self.N_DOF, self.N_DOF)), units='N/m')

        self.add_output('eig_vectors', val=np.ones((self.N_DOF, self.N_DOF)))
        self.add_output('eig_vals', val=np.eye(self.N_DOF))

    # def setup_partials(self):
    #     self.declare_partials('eig_vectors', ['M_mode', 'K_mode'])
    #     self.declare_partials('eig_vals', ['M_mode', 'K_mode'])

    def compute(self, inputs, outputs):
        K = inputs['K_mode']
        M = inputs['M_mode']

        vals, vecs = scipy.linalg.eig(M,K, left=False, right=True) # To match He (2022) paper - note must invert eigenvalues to get to natural frequencies
        # # Combining to A?
        # A = scipy.linalg.inv(K) @ M
        # avals, avecs = scipy.linalg.eig(A, left=False, right=True)
        # # Direct LAPACK call
        # wr, wi, _, vr, _ = scipy.linalg.lapack.sgeev(A)
        # avals = wr + (1j * wi)
        # avecs = vr
        
        # Normalize eigenvectors with M matrix
        norm_fac = np.zeros((1, self.N_DOF))
        vecs_mortho = np.zeros((self.N_DOF, self.N_DOF))
        for i in range(self.N_DOF):
            norm_fac[0,i] = np.sqrt(1./(vecs[:,i].T @ M @ vecs[:,i]))
            if not np.isnan(norm_fac[0,i]) : # To avoid ending up with an entire eigenvector of NaNs 
                vecs_mortho[:,i] = norm_fac[0,i] * vecs[:,i]
        
        # # Throw errors for unexpected eigenvalues
        # if any(np.imag(vals) != 0.) :
        #     raise om.AnalysisError('Imaginary eigenvalues')
        # if any(np.real(vals) < -1.e-03) :
        #     raise om.AnalysisError('Negative eigenvalues')

        # Check solution
        if not np.allclose((M @ vecs) - (K @ vecs @ np.diag(vals)), np.zeros((self.N_DOF,self.N_DOF)), atol=1.0) :
            raise om.AnalysisError('Eigenvalue problem looks wrong')
        if not np.allclose((vecs_mortho.T @ M @ vecs_mortho) - np.eye(self.N_DOF), np.zeros((self.N_DOF,self.N_DOF)), atol=1.0) :
            raise om.AnalysisError('Eigenvectors not scaled properly')

        # Calculate modal expansion
        u_vec = np.ones(self.N_DOF)
        q_vec = np.zeros(self.N_DOF)
        for i in range(self.N_DOF):
            q_vec[i] = (vecs[:,i].T @ M @ u_vec)/(vecs[:,i].T @ M @ vecs[:,i])

        outputs['eig_vectors'] = vecs_mortho
        outputs['eig_vals'] = np.diag(np.real(vals))

        self.vecs = vecs_mortho
        self.vecs_unsc = vecs
        self.vals = np.diag(vals) # Must keep complex eigenvalues to have correct derivatives
    
        ## Based on He, Jonsson, Martins (2022) - Section C. "Modal Method"
        F = np.zeros((self.N_DOF, self.N_DOF), dtype=complex)
        for i in range(self.N_DOF):
            for j in range(self.N_DOF):
                if i == j:
                    F[i,j] = 0.
                else:
                    F[i,j] = vals[i]/(vals[j]-vals[i])
        self.F_matrix = F

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        vecs = self.vecs
        vecs_unsc = self.vecs_unsc
        vals = np.real(self.vals)
        K = inputs['K_mode']
        M = inputs['M_mode']
        F = self.F_matrix

        ## Based on He, Jonsson, Martins (2022) - Section C. "Modal Method"
        if mode == 'rev':    
            if 'eig_vectors' in d_outputs:
                if 'M_mode' in d_inputs:
                    d_inputs['M_mode'] += np.real(vecs @ (np.multiply(F,(vecs.T @ d_outputs['eig_vectors'])) - np.multiply((0.5*np.eye(self.N_DOF)), (vecs.T @ d_outputs['eig_vectors']))) @ vecs.T)
                if 'K_mode' in d_inputs:
                    d_inputs['K_mode'] += np.real(-1. * vecs @ np.multiply(F,(vecs.T @ d_outputs['eig_vectors'])) @ vals @ vecs.T)
            if 'eig_vals' in d_outputs:
                if 'M_mode' in d_inputs:
                    d_inputs['M_mode'] += np.real(vecs @ (vals @ d_outputs['eig_vals']) @ vecs.T)
                if 'K_mode' in d_inputs:
                    d_inputs['K_mode'] += np.real(-1. * vecs @ (vals @ d_outputs['eig_vals']) @ vals @ vecs.T)

        elif mode == 'fwd':
            if 'eig_vectors' in d_outputs:
                if 'M_mode' in d_inputs:
                    d_outputs['eig_vectors'] += np.real((vecs @ np.multiply(F, (vecs.T @ d_inputs['M_mode'] @ vecs))) - (0.5 * vecs @ np.multiply(np.eye(self.N_DOF), (vecs.T @ d_inputs['M_mode'] @ vecs))))
                if 'K_mode' in d_inputs:
                    d_outputs['eig_vectors'] += np.real((vecs @ np.multiply(F, (-1. * vecs.T @ d_inputs['K_mode'] @ vecs @ vals))))
            if 'eig_vals' in d_outputs:
                if 'M_mode' in d_inputs:
                    d_outputs['eig_vals'] += np.real(vals @ np.multiply(np.eye(self.N_DOF), (vecs.T @ d_inputs['M_mode'] @ vecs)))
                if 'K_mode' in d_inputs:
                    d_outputs['eig_vals'] += np.real(vals @ np.multiply(np.eye(self.N_DOF), (-1. * vecs.T @ d_inputs['K_mode'] @ vecs @ vals)))