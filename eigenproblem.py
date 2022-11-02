import numpy as np
import scipy.linalg
import openmdao.api as om

class Eigenproblem(om.ExplicitComponent):
    # Full solution to eigenvalue problem

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nDOF_r = self.nodal_data['nDOF_r']
        nDOF_tot = self.nodal_data['nDOF_tot']

        self.add_input('Mr_glob', val=np.ones((nDOF_r, nDOF_r)), units='kg')
        self.add_input('Kr_glob', val=np.ones((nDOF_r, nDOF_r)), units='N/m')

        self.add_output('Q', val=np.ones((nDOF_tot, nDOF_r)))
        self.add_output('eig_freqs', val=np.zeros(nDOF_r), units='1/s')


    # def setup_partials(self):
    #     self.declare_partials('Q', ['Mr_glob', 'Kr_glob'])
    #     self.declare_partials('eig_freqs', ['Mr_glob', 'Kr_glob'])

    def compute(self, inputs, outputs):
        nDOF = self.nodal_data['nDOF_r']
        Tr = self.nodal_data['Tr']
        K = inputs['Kr_glob']
        M = inputs['Mr_glob']

        # --- To match He (2022) paper - note must invert eigenvalues to get to natural frequencies
        # D, Q = scipy.linalg.eig(M,K, left=False, right=True) 
        # Normalize eigenvectors with M matrix
        # Q_og = Q # preserve original eigvectors
        # for j in range(nDOF):
        #     norm_fac = np.sqrt(1./(Q[:,j].T @ M @ Q[:,j]))
        #     if not np.isnan(norm_fac) : # To avoid ending up with an entire eigenvector of NaNs 
        #         Q[:,j] = norm_fac * Q[:,j]
        # lambdaDiag = 1. / np.real(D) # Note lambda might have off diagonal values due to numerics

        D, Q = scipy.linalg.eig(K,M)
        Q_og = Q
        # Normalize eigenvectors
        for j in range(M.shape[1]):
            q_j = Q[:,j]
            modalmass_j = np.dot(q_j.T,M).dot(q_j)
            Q[:,j]= Q[:,j]/np.sqrt(modalmass_j)

        # Sort and diagonalize
        Lambda = (np.dot(Q.T,K).dot(Q)) 
        # inversion because of Eigenproblem definition
        lambdaDiag = np.diag(Lambda) # Note lambda might have off diagonal values due to numerics
        I = np.argsort(lambdaDiag)
        Q = Q[:,I]
        lambdaDiag = lambdaDiag[I]
        # Export frequencies
        Lambda = np.sqrt(lambdaDiag)/(2*np.pi) # frequencies [Hz]

        # --- Renormalize modes 
        for j in range(Q.shape[1]):
            q_j = Q[:,j]
            iMax = np.argmax(np.abs(q_j))
            scale = q_j[iMax] # not using abs to normalize to "1" and not "+/-1"
            Q[:,j]= Q[:,j]/scale

        # --- Add removed DOF back into eigenvectors
        Qr = Q
        Q = Tr.dot(Qr)

        self.vecs = Q
        self.vecs_unsc = Q_og
        self.vals = D # Must keep complex eigenvalues to have correct derivatives
        
        # --- Sanitization, ensure real values (for export, not for derivatives!)
        Q_im = np.imag(Q)
        Q = np.real(Q)
        imm = np.mean(np.abs(Q_im),axis = 0)
        bb = imm>0
        if sum(bb)>0:
            W=list(np.where(bb)[0])
            print('[WARN] Found {:d} complex eigenvectors at positions {}/{}'.format(sum(bb),W,Q.shape[0]))
        Lambda = np.real(Lambda)

        outputs['Q'] = Q
        outputs['eig_freqs'] = Lambda
    
        ## Based on He, Jonsson, Martins (2022) - Section C. "Modal Method"
        F = np.zeros((nDOF, nDOF), dtype=complex)
        for i in range(nDOF):
            for j in range(nDOF):
                if i == j:
                    F[i,j] = 0.
                else:
                    F[i,j] = Lambda[i]/(Lambda[j]-Lambda[i])
        self.F_matrix = F

        # # Throw errors for unexpected eigenvalues
        # if any(np.imag(vals) != 0.) :
        #     raise om.AnalysisError('Imaginary eigenvalues')
        # if any(np.real(vals) < -1.e-03) :
        #     raise om.AnalysisError('Negative eigenvalues')

        # # Check solution
        # if not np.allclose((M @ vecs) - (K @ vecs @ np.diag(vals)), np.zeros((nDOF,nDOF)), atol=1.0) :
        #     raise om.AnalysisError('Eigenvalue problem looks wrong')
        # if not np.allclose((vecs_mortho.T @ M @ vecs_mortho) - np.eye(nDOF), np.zeros((nDOF,nDOF)), atol=1.0) :
        #     raise om.AnalysisError('Eigenvectors not scaled properly')

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        nDOF = self.nodal_data['nDOF_r']
        vecs = self.vecs
        vecs_unsc = self.vecs_unsc
        vals = np.real(self.vals)
        K = inputs['Kr_glob']
        M = inputs['Mr_glob']
        F = self.F_matrix

        ## Based on He, Jonsson, Martins (2022) - Section C. "Modal Method"
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