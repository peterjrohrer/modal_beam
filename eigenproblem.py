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

        vals, vecs = scipy.linalg.eig(K,M)

        # if any(np.imag(vals) != 0.) :
        #     raise om.AnalysisError('Imaginary eigenvalues')
        # if any(np.real(vals) < 0.) :
        #     raise om.AnalysisError('Negative eigenvalues')

        # Normalize eigenvectors with M matrix
        norm_fac = np.zeros((1, nDOF))
        vecs_mortho = np.zeros((nDOF, nDOF))
        for i in range(nDOF):
            norm_fac[0,i] = np.sqrt(1./(vecs[:,i].T @ M @ vecs[:,i]))
            vecs_mortho[:,i] = norm_fac[0,i] * vecs[:,i]

        outputs['eig_vectors'] = vecs_mortho
        outputs['eig_vals'] = np.diag(np.real(vals))

        self.vecs = vecs_mortho
        self.vals = np.diag(np.real(vals))
    
    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        nDOF = self.options['nDOF']        
        vecs = self.vecs
        vals = self.vals
        K = inputs['K_mode']
        M = inputs['M_mode']

        # ## Based on He, Jonsson, Martins (2022)
        # F = np.zeros((nDOF, nDOF))
        # G = np.zeros((nDOF, nDOF))
        # for i in range(nDOF):
        #     for j in range(nDOF):
        #         if i == j:
        #             F[i,j] = 0.
        #         else:
        #             F[i,j] = vals[i,i]/(vals[j,j]-vals[i,i])
        #         G[i,j] = vals[i,i]/vals[j,j]

        # if mode == 'rev':    
        #     if 'eig_vectors' in d_outputs:
        #         if 'M_mode' in d_inputs:
        #             d_inputs['M_mode'] += (vecs @ ((vals.T @ d_outputs['eig_vals']) + np.multiply((F-G),(vecs.T @ d_outputs['eig_vectors'])) - np.multiply((0.5*np.eye(nDOF)), (vecs.T @ d_outputs['eig_vectors']))) @ vecs.T) + ((np.linalg.inv(K) @ d_outputs['eig_vectors']) @ np.linalg.inv(vals) @ vecs.T)
        #         if 'K_mode' in d_inputs:
        #             d_inputs['K_mode'] += (-1. * vecs @ ((vals.T @ d_outputs['eig_vals']) + np.multiply((F-G),(vecs.T @ d_outputs['eig_vectors']))) @ vals @ vecs.T) - ((np.linalg.inv(K) @ d_outputs['eig_vectors']) @ vecs.T)
        #     if 'eig_vals' in d_outputs:
        #         if 'M_mode' in d_inputs:
        #             d_inputs['M_mode'] += (vecs @ ((vals.T @ d_outputs['eig_vals']) + np.multiply((F-G),(vecs.T @ d_outputs['eig_vectors'])) - np.multiply((0.5*np.eye(nDOF)), (vecs.T @ d_outputs['eig_vectors']))) @ vecs.T) + ((np.linalg.inv(K) @ d_outputs['eig_vectors']) @ np.linalg.inv(vals) @ vecs.T)
        #         if 'K_mode' in d_inputs:
        #             d_inputs['K_mode'] += (-1. * vecs @ ((vals.T @ d_outputs['eig_vals']) + np.multiply((F-G),(vecs.T @ d_outputs['eig_vectors']))) @ vals @ vecs.T) - ((np.linalg.inv(K) @ d_outputs['eig_vectors']) @ vecs.T)

        # elif mode == 'fwd':
        #     if 'eig_vectors' in d_outputs:
        #         if 'M_mode' in d_inputs:
        #             d_outputs['eig_vectors'] += (vals @ np.multiply(F, ((-1. * vecs.T @ d_inputs['K_mode'] @ vecs @ vals) + (vecs.T @ d_inputs['M_mode'] @ vecs)))) - (0.5 * vecs @ np.multiply(np.eye(nDOF), (vecs.T @ d_inputs['M_mode'] @ vecs)))
        #         if 'K_mode' in d_inputs:
        #             d_outputs['eig_vectors'] += (vals @ np.multiply(F, ((-1. * vecs.T @ d_inputs['K_mode'] @ vecs @ vals) + (vecs.T @ d_inputs['M_mode'] @ vecs)))) - (0.5 * vecs @ np.multiply(np.eye(nDOF), (vecs.T @ d_inputs['M_mode'] @ vecs)))
        #     if 'eig_vals' in d_outputs:
        #         if 'M_mode' in d_inputs:
        #             d_outputs['eig_vals'] += vals @ np.multiply(np.eye(nDOF), ((-1. * vecs.T @ d_inputs['K_mode'] @ vecs @ vals) + (vecs.T @ d_inputs['M_mode'] @ vecs)))
        #         if 'K_mode' in d_inputs:
        #             d_outputs['eig_vals'] += vals @ np.multiply(np.eye(nDOF), ((-1. * vecs.T @ d_inputs['K_mode'] @ vecs @ vals) + (vecs.T @ d_inputs['M_mode'] @ vecs)))
        # ## -------
        # ## Based on Mesquita (1989)
        # F = np.zeros((nDOF, nDOF))
        # for i in range(nDOF):
        #     for j in range(nDOF):
        #         if i == j:
        #             F[i,j] = 0.
        #         else:
        #             F[i,j] = vals[i,i]/(vals[j,j]-vals[i,i])

        # if mode == 'rev':    
        #     if 'eig_vectors' in d_outputs:
        #         if 'M_mode' in d_inputs:
        #             d_inputs['M_mode'] += vecs @ ((vals.T @ d_outputs['eig_vals']) + np.multiply(F,(vecs.T @ d_outputs['eig_vectors'])) - np.multiply((0.5*np.eye(nDOF)), (vecs.T @ d_outputs['eig_vectors']))) @ vecs.T
        #         if 'K_mode' in d_inputs:
        #             d_inputs['K_mode'] += -1. * vecs @ ((vals.T @ d_outputs['eig_vals']) + np.multiply(F,(vecs.T @ d_outputs['eig_vectors']))) @ vals @ vecs.T
        #     if 'eig_vals' in d_outputs:
        #         if 'M_mode' in d_inputs:
        #             d_inputs['M_mode'] += vecs @ ((vals.T @ d_outputs['eig_vals']) + np.multiply(F,(vecs.T @ d_outputs['eig_vectors'])) - np.multiply((0.5*np.eye(nDOF)), (vecs.T @ d_outputs['eig_vectors']))) @ vecs.T
        #         if 'K_mode' in d_inputs:
        #             d_inputs['K_mode'] += -1. * vecs @ ((vals.T @ d_outputs['eig_vals']) + np.multiply(F,(vecs.T @ d_outputs['eig_vectors']))) @ vals @ vecs.T

        # elif mode == 'fwd':
        #     if 'eig_vectors' in d_outputs:
        #         print('a')


        #         if 'M_mode' in d_inputs:
        #             d_outputs['eig_vectors'] += (vals @ np.multiply(F, ((-1. * vecs.T @ d_inputs['K_mode'] @ vecs @ vals) + (vecs.T @ d_inputs['M_mode'] @ vecs)))) - (0.5 * vecs @ np.multiply(np.eye(nDOF), (vecs.T @ d_inputs['M_mode'] @ vecs)))
        #         if 'K_mode' in d_inputs:
        #             d_outputs['eig_vectors'] += (vals @ np.multiply(F, ((-1. * vecs.T @ d_inputs['K_mode'] @ vecs @ vals) + (vecs.T @ d_inputs['M_mode'] @ vecs)))) - (0.5 * vecs @ np.multiply(np.eye(nDOF), (vecs.T @ d_inputs['M_mode'] @ vecs)))
        #     if 'eig_vals' in d_outputs:
        #         if 'M_mode' in d_inputs:
        #             d_outputs['eig_vals'] += vecs.T @ (d_inputs['K_mode'] - (vals @ d_inputs['M_mode'])) @ vecs
        #         if 'K_mode' in d_inputs:
        #             d_outputs['eig_vals'] += vecs.T @ (d_inputs['K_mode'] - (vals @ d_inputs['M_mode'])) @ vecs
        # ## -------
        ## Based on Fox and Kapoor (1968)
        a = np.zeros((nDOF,nDOF))
        F = np.zeros((nDOF, nDOF))
        G = np.zeros((nDOF, nDOF))
        for i in range(nDOF):
            for j in range(nDOF):
                if i == j:
                    F[i,j] = 0.
                else:
                    F[i,j] = vals[i,i]/(vals[j,j]-vals[i,i])
                G[i,j] = vals[i,i]/vals[j,j]

        if mode == 'rev':    
            if 'eig_vectors' in d_outputs:
                if 'M_mode' in d_inputs:
                    d_inputs['M_mode'] += (vecs @ ((vals.T @ d_outputs['eig_vals']) + np.multiply((F-G),(vecs.T @ d_outputs['eig_vectors'])) - np.multiply((0.5*np.eye(nDOF)), (vecs.T @ d_outputs['eig_vectors']))) @ vecs.T) + ((np.linalg.inv(K) @ d_outputs['eig_vectors']) @ np.linalg.inv(vals) @ vecs.T)
                if 'K_mode' in d_inputs:
                    d_inputs['K_mode'] += (-1. * vecs @ ((vals.T @ d_outputs['eig_vals']) + np.multiply((F-G),(vecs.T @ d_outputs['eig_vectors']))) @ vals @ vecs.T) - ((np.linalg.inv(K) @ d_outputs['eig_vectors']) @ vecs.T)
            if 'eig_vals' in d_outputs:
                if 'M_mode' in d_inputs:
                    d_inputs['M_mode'] += (vecs @ ((vals.T @ d_outputs['eig_vals']) + np.multiply((F-G),(vecs.T @ d_outputs['eig_vectors'])) - np.multiply((0.5*np.eye(nDOF)), (vecs.T @ d_outputs['eig_vectors']))) @ vecs.T) + ((np.linalg.inv(K) @ d_outputs['eig_vectors']) @ np.linalg.inv(vals) @ vecs.T)
                if 'K_mode' in d_inputs:
                    d_inputs['K_mode'] += (-1. * vecs @ ((vals.T @ d_outputs['eig_vals']) + np.multiply((F-G),(vecs.T @ d_outputs['eig_vectors']))) @ vals @ vecs.T) - ((np.linalg.inv(K) @ d_outputs['eig_vectors']) @ vecs.T)

        elif mode == 'fwd':
            if 'eig_vectors' in d_outputs:
                dX_dj = np.zeros((nDOF,nDOF))

                for i in range(nDOF):
                    for k in range(nDOF):
                        if i != k :
                            a[i,k] += (vecs[:,k] @ (d_inputs['K_mode'] - (vals[i,i]*d_inputs['M_mode'])) @ vecs[:,i])/(vals[i,i]-vals[k,k])
                        elif i == k :
                            a[i,k] += (-0.5) * (vecs[:,i].T @ d_inputs['M_mode'] @ vecs[:,i])

                        dX_dj[:,i] += a[i,k] * vecs[:,k]
                # This is off by a factor of two??
                if 'M_mode' in d_inputs:
                    d_outputs['eig_vectors'] += dX_dj
                if 'K_mode' in d_inputs:
                    d_outputs['eig_vectors'] += dX_dj

                print('a')

            if 'eig_vals' in d_outputs:
                if 'M_mode' in d_inputs:
                    d_outputs['eig_vals'] += vecs.T @ (d_inputs['K_mode'] - (vals @ d_inputs['M_mode'])) @ vecs
                if 'K_mode' in d_inputs:
                    d_outputs['eig_vals'] += vecs.T @ (d_inputs['K_mode'] - (vals @ d_inputs['M_mode'])) @ vecs