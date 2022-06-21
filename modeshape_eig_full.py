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

    def compute_partials(self, inputs, partials):
        nDOF = self.options['nDOF']

        K = inputs['K_mode']
        M = inputs['M_mode']

        partials['eig_vectors','K_mode'] = np.zeros((nDOF*nDOF, nDOF*nDOF))
        partials['eig_vectors','M_mode'] = np.zeros((nDOF*nDOF, nDOF*nDOF))
        partials['eig_vals','K_mode'] = np.zeros((nDOF*nDOF, nDOF*nDOF))
        partials['eig_vals','M_mode'] = np.zeros((nDOF*nDOF, nDOF*nDOF))

        vals, vecs = scipy.linalg.eig(K,M)

        ## Values
        for i in range(nDOF):
            partials['eig_vals','K_mode'][i,:] += np.ones(nDOF*nDOF)
            partials['eig_vals','M_mode'][i,:] += ((vecs[:,i].T)@(vals[i]*vecs[:,i]))*np.ones(nDOF*nDOF)

        ## Vectors
        F = np.zeros((nDOF,nDOF,nDOF))
        for i in range(nDOF):
            F[:,:,i] += K - vals[i]*M

        # F = np.zeros((nDOF, nDOF))
        # ## What should diagonal values be??
        # for i in range(nDOF):
        #     for j in range(nDOF):
        #         F[i,j] = vals[i]/(vals[j]-vals[i])
        #         if i == j:
        #             F[i,j] = 0.

        print('help!')
