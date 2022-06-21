import numpy as np
import scipy.linalg
import openmdao.api as om

class ModeshapeEigen(om.ExplicitComponent):
    # Full solution to eigenvalue problem

    def initialize(self):
        self.options.declare('nNode', types=int)
        self.options.declare('nElem', types=int)
        self.options.declare('nDOF', types=int)

    def setup(self):
        nNode = self.options['nNode']
        nElem = self.options['nElem']
        nDOF = self.options['nDOF']

        self.add_input('M_mode', val=np.zeros((nDOF, nDOF)), units='kg')
        self.add_input('K_mode', val=np.zeros((nDOF, nDOF)), units='N/m')

        self.add_output('eig_vectors', val=np.zeros((nDOF, nDOF)))
        self.add_output('eig_vals', val=np.zeros((nDOF, nDOF)))

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        K = inputs['K_mode']
        M = inputs['M_mode']

        vals, vecs = scipy.linalg.eig(K,M)

        if any(np.imag(vals) != 0.) :
            raise om.AnalysisError('Imaginary eigenvalues')
        if any(np.real(vals) < 0.) :
            raise om.AnalysisError('Negative eigenvalues')

        outputs['eig_vectors'] = vecs
        outputs['eig_vals'] = np.diag(np.real(vals))

        self.vecs = vecs
        self.vals = np.diag(np.real(vals))

    # ## Adjoint method
    # def compute_partials(self, inputs, partials):
    #     nDOF = self.options['nDOF']
    #     vecs = self.vecs
    #     vals = self.vals

    #     K = inputs['K_mode']
    #     M = inputs['M_mode']

    #     for i in range(nDOF):


    # ## Fox and Kapoor analytical? 
    # def compute_partials(self, inputs, partials):
    #     nDOF = self.options['nDOF']

    #     K = inputs['K_mode']
    #     M = inputs['M_mode']

    #     partials['eig_vectors','K_mode'] = np.zeros((nDOF*nDOF, nDOF*nDOF))
    #     partials['eig_vectors','M_mode'] = np.zeros((nDOF*nDOF, nDOF*nDOF))
    #     partials['eig_vals','K_mode'] = np.zeros((nDOF*nDOF, nDOF*nDOF))
    #     partials['eig_vals','M_mode'] = np.zeros((nDOF*nDOF, nDOF*nDOF))

    #     vals_img, vecs = scipy.linalg.eig(K,M)

    #     # if any(np.imag(vals) != 0.) :
    #     #     raise om.AnalysisError('Imaginary eigenvalues')
    #     # if any(np.real(vals) < 0.) :
    #     #     raise om.AnalysisError('Negative eigenvalues')

    #     vals = np.diag(np.real(vals_img))

    #     ## Values
    #     for i in range(nDOF):
    #         partials['eig_vals','K_mode'][i,:] += np.ones(nDOF*nDOF)
    #         partials['eig_vals','M_mode'][i,:] += ((vecs[:,i].T)@(vals[i]*vecs[:,i]))*np.ones(nDOF*nDOF)

    #     ## Vectors
    #     F = np.zeros((nDOF,nDOF,nDOF))
    #     for i in range(nDOF):
    #         F[:,:,i] += K - vals[i]*M

    #     # F = np.zeros((nDOF, nDOF))
    #     # ## What should diagonal values be??
    #     # for i in range(nDOF):
    #     #     for j in range(nDOF):
    #     #         F[i,j] = vals[i]/(vals[j]-vals[i])
    #     #         if i == j:
    #     #             F[i,j] = 0.
    
    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        nDOF = self.options['nDOF']

        K = inputs['K_mode']
        M = inputs['M_mode']

        vals_col, vecs = scipy.linalg.eig(K,M)
        vals = np.diag(np.real(vals_col))
        
        # vecs = self.vecs
        # vals = self.vals

        if mode == 'rev':
            F = np.zeros((nDOF, nDOF))
            for i in range(nDOF):
                for j in range(nDOF):
                    if i == j:
                        F[i,j] = 0.
                    else:
                        F[i,j] = vals[i,i]/(vals[j,j]-vals[i,i])
            
            if 'eig_vectors' in d_outputs:
                if 'M_mode' in d_inputs:
                    d_inputs['M_mode'] += vecs @ ((vals.T @ d_outputs['eig_vals']) + np.multiply(F,(vecs.T @ d_outputs['eig_vectors'])) - np.multiply((0.5*np.eye(nDOF)), (vecs.T @ d_outputs['eig_vectors']))) @ vecs.T
                if 'K_mode' in d_inputs:
                    d_inputs['K_mode'] += -1. * vecs @ ((vals.T @ d_outputs['eig_vals']) + np.multiply(F,(vecs.T @ d_outputs['eig_vectors']))) @ vals @ vecs.T
            if 'eig_vals' in d_outputs:
                if 'M_mode' in d_inputs:
                    d_inputs['M_mode'] += vecs @ ((vals.T @ d_outputs['eig_vals']) + np.multiply(F,(vecs.T @ d_outputs['eig_vectors'])) - np.multiply((0.5*np.eye(nDOF)), (vecs.T @ d_outputs['eig_vectors']))) @ vecs.T
                if 'K_mode' in d_inputs:
                    d_inputs['K_mode'] += -1. * vecs @ ((vals.T @ d_outputs['eig_vals']) + np.multiply(F,(vecs.T @ d_outputs['eig_vectors']))) @ vals @ vecs.T

            # d_inputs['M_mode'] = vecs @ ((vals.T @ d_outputs['eig_vals']) + np.multiply(F,(vecs.T @ d_outputs['eig_vectors'])) - np.multiply((0.5*np.eye(nDOF)), (vecs.T @ d_outputs['eig_vectors']))) @ vecs.T
            # d_inputs['K_mode'] = -1. * vecs @ ((vals.T @ d_outputs['eig_vals']) + np.multiply(F,(vecs.T @ d_outputs['eig_vectors']))) @ vals @ vecs.T

        elif mode == 'fwd':
            raise Exception('Forward mode partial derivatives not defined!')
