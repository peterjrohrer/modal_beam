import numpy as np
import scipy.linalg
import openmdao.api as om

class Eigenproblem(om.ExplicitComponent):
    # Full solution to eigenvalue problem

    def initialize(self):
        self.options.declare('nDOF', types=int)

    def setup(self):
        nDOF = self.options['nDOF']
        self.add_input('M_mode', val=np.zeros((nDOF, nDOF)), units='kg')
        self.add_input('K_mode', val=np.zeros((nDOF, nDOF)), units='N/m')

        self.add_output('eig_vectors', val=np.zeros((nDOF, nDOF)))
        self.add_output('eig_vals', val=np.zeros((nDOF, nDOF)))

        # self.declare_partials('*','*', method='fd', form='central', step_calc='rel_avg', step=1.e-8)

    def compute(self, inputs, outputs):
        nDOF = self.options['nDOF']
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
        norm_fac = np.zeros((1, nDOF))
        vecs_mortho = np.zeros((nDOF, nDOF))
        for i in range(nDOF):
            norm_fac[0,i] = np.sqrt(1./(vecs[:,i].T @ M @ vecs[:,i]))
            vecs_mortho[:,i] = norm_fac[0,i] * vecs[:,i]
        
        # Throw errors for unexpected eigenvalues
        if any(np.imag(vals) != 0.) :
            raise om.AnalysisError('Imaginary eigenvalues')
        if any(np.real(vals) < 0.) :
            raise om.AnalysisError('Negative eigenvalues')

        # Check solution
        if not np.allclose((M @ vecs_mortho) - (K @ vecs_mortho @ np.diag(vals)), np.zeros((nDOF,nDOF)), atol=1.e-03) :
            raise om.AnalysisError('Eigenvalue problem looks wrong')
        if not np.allclose((vecs_mortho.T @ M @ vecs_mortho) - np.eye(nDOF), np.zeros((nDOF,nDOF)), atol=1.e-03) :
            raise om.AnalysisError('Eigenvectors not scaled properly')

        outputs['eig_vectors'] = vecs_mortho
        outputs['eig_vals'] = np.diag(np.real(vals))

        self.vecs = vecs_mortho
        self.vals = np.diag(vals) # Must keep complex eigenvalues to have correct derivatives
    
    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        nDOF = self.options['nDOF']        
        vecs = self.vecs
        vals = self.vals
        K = inputs['K_mode']
        M = inputs['M_mode']

        ## Based on He, Jonsson, Martins (2022) - Section C. "Modal Method"
        F = np.zeros((nDOF, nDOF), dtype=complex)
        for i in range(nDOF):
            for j in range(nDOF):
                if i == j:
                    F[i,j] = 0.
                else:
                    F[i,j] = vals[i,i]/(vals[j,j]-vals[i,i])

        if mode == 'rev':    
            if 'eig_vectors' in d_outputs:
                if 'M_mode' in d_inputs:
                    d_inputs['M_mode'] += np.real(vecs @ (np.multiply(F,(vecs.T @ d_outputs['eig_vectors'])) - np.multiply((0.5*np.eye(nDOF)), (vecs.T @ d_outputs['eig_vectors']))) @ vecs.T)
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
                    d_outputs['eig_vectors'] += np.real((vecs @ np.multiply(F, (vecs.T @ d_inputs['M_mode'] @ vecs))) - (0.5 * vecs @ np.multiply(np.eye(nDOF), (vecs.T @ d_inputs['M_mode'] @ vecs))))
                if 'K_mode' in d_inputs:
                    d_outputs['eig_vectors'] += np.real((vecs @ np.multiply(F, (-1. * vecs.T @ d_inputs['K_mode'] @ vecs @ vals))))
            if 'eig_vals' in d_outputs:
                if 'M_mode' in d_inputs:
                    d_outputs['eig_vals'] += np.real(vals @ np.multiply(np.eye(nDOF), (vecs.T @ d_inputs['M_mode'] @ vecs)))
                if 'K_mode' in d_inputs:
                    d_outputs['eig_vals'] += np.real(vals @ np.multiply(np.eye(nDOF), (-1. * vecs.T @ d_inputs['K_mode'] @ vecs @ vals)))

        ## -------
        # ## Based on Fox and Kapoor (1968)
        #     # FWD mode only!
        # a = np.zeros((nDOF,nDOF), dtype=complex)
        # F = np.zeros((nDOF, nDOF), dtype=complex)
        # G = np.zeros((nDOF, nDOF), dtype=complex)
        # # for i in range(nDOF):
        # #     for j in range(nDOF):
        # #         if i == j:
        # #             F[i,j] = 0.
        # #         else:
        # #             F[i,j] = vals[i,i]/(vals[j,j]-vals[i,i])
        # #         G[i,j] = vals[i,i]/vals[j,j]

        # if mode == 'rev':
        #     for i in range(nDOF):
        #         for j in range(nDOF):
        #             if i == j:
        #                 F[i,j] = 0.
        #             else:
        #                 F[i,j] = vals[i,i]/(vals[j,j]-vals[i,i])
        #             G[i,j] = vals[i,i]/vals[j,j]    
        #     # if 'eig_vectors' in d_outputs:
        #     #     if 'M_mode' in d_inputs:
        #     #         d_inputs['M_mode'] += (vecs @ ((vals.T @ d_outputs['eig_vals']) + np.multiply((F-G),(vecs.T @ d_outputs['eig_vectors'])) - np.multiply((0.5*np.eye(nDOF)), (vecs.T @ d_outputs['eig_vectors']))) @ vecs.T) + ((np.linalg.inv(K) @ d_outputs['eig_vectors']) @ np.linalg.inv(vals) @ vecs.T)
        #     #     if 'K_mode' in d_inputs:
        #     #         d_inputs['K_mode'] += (-1. * vecs @ ((vals.T @ d_outputs['eig_vals']) + np.multiply((F-G),(vecs.T @ d_outputs['eig_vectors']))) @ vals @ vecs.T) - ((np.linalg.inv(K) @ d_outputs['eig_vectors']) @ vecs.T)
        #     # if 'eig_vals' in d_outputs:
        #     #     if 'M_mode' in d_inputs:
        #     #         d_inputs['M_mode'] += (vecs @ ((vals.T @ d_outputs['eig_vals']) + np.multiply((F-G),(vecs.T @ d_outputs['eig_vectors'])) - np.multiply((0.5*np.eye(nDOF)), (vecs.T @ d_outputs['eig_vectors']))) @ vecs.T) + ((np.linalg.inv(K) @ d_outputs['eig_vectors']) @ np.linalg.inv(vals) @ vecs.T)
        #     #     if 'K_mode' in d_inputs:
        #     #         d_inputs['K_mode'] += (-1. * vecs @ ((vals.T @ d_outputs['eig_vals']) + np.multiply((F-G),(vecs.T @ d_outputs['eig_vectors']))) @ vals @ vecs.T) - ((np.linalg.inv(K) @ d_outputs['eig_vectors']) @ vecs.T)

        # elif mode == 'fwd':
        #     if 'eig_vectors' in d_outputs:
        #         # # --- Using Formulation 1
        #         # dF_dj = np.zeros((nDOF,nDOF), dtype=complex)
        #         # dX_dj = np.zeros((nDOF,nDOF), dtype=complex)
        #         # for i in range(nDOF):
        #         #     # Each F matrix should be singular - they are not??
        #         #     F = K - (vals[i,i]*M)
        #         #     dF_dj = d_inputs['K_mode'] - (vals[i,i] * d_inputs['M_mode']) - (d_outputs['eig_vals'] @ M)
        #         #     X = np.reshape(vecs[:,i],(nDOF,1))
        #         #     dX_dj[:,i] += np.reshape(-1. * np.linalg.inv((F @ F) + (2. * M @ X @ X.T @ M)) @ ((F @ dF_dj) + (M @ X @ X.T @ d_inputs['M_mode'])) @ X,(nDOF))

        #         # if 'M_mode' in d_inputs or 'K_mode' in d_inputs:
        #         #     d_outputs['eig_vectors'] += np.real(dX_dj)
                
        #         # --- Using Formulation 2
        #         dX_dj = np.zeros((nDOF,nDOF), dtype=complex)
        #         for i in range(nDOF):
        #             for k in range(nDOF):
        #                 if i != k :
        #                     a[i,k] += (vecs[:,k] @ (d_inputs['K_mode'] - (vals[i,i]*d_inputs['M_mode'])) @ vecs[:,i])/(vals[i,i]-vals[k,k])
        #                 elif i == k :
        #                     a[i,k] += (-0.5) * (vecs[:,i].T @ d_inputs['M_mode'] @ vecs[:,i])
        #             # for k in range(int(nDOF/2),nDOF) :
        #                 dX_dj[:,i] += a[i,k] * vecs[:,k]
        #         if 'M_mode' in d_inputs or 'K_mode' in d_inputs:
        #             d_outputs['eig_vectors'] += np.abs(dX_dj)
            
        #     if 'eig_vals' in d_outputs:
        #         if 'M_mode' in d_inputs or 'K_mode' in d_inputs:
        #             for i in range(nDOF):
        #                 d_outputs['eig_vals'][i,i] += np.real(vecs[:,i].T @ (d_inputs['K_mode'] - (vals[i,i] * d_inputs['M_mode'])) @ vecs[:,i])
