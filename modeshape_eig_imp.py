import numpy as np
import scipy.linalg
import openmdao.api as om


class ModeshapeEigen(om.ImplicitComponent):
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

        self.add_output('Phi', val=np.zeros((nDOF, nDOF)))
        self.add_output('Lambda', val=np.zeros(nDOF))

    def setup_partials(self):
        self.declare_partials('*', '*')

    def apply_nonlinear(self, inputs, outputs, residuals):
        K = inputs['K_mode']
        M = inputs['M_mode']
        Phi = outputs['Phi']
        Lambda = outputs['Lambda']

        np.matmul(inputs['M_mode'],outputs)


    def compute(self, inputs, outputs):
        K = inputs['K_mode']
        M = inputs['M_mode']

        vals, vecs = scipy.linalg.eig(K,M)

        outputs['Phi'] = vecs
        outputs['Lambda'] = vals

    def compute_partials(self, inputs, partials):
        nDOF = self.options['nDOF']

        K = inputs['K_mode']
        M = inputs['M_mode']

        partials['Phi','K_mode'] = np.zeros((nDOF*nDOF, nDOF*nDOF))
        partials['Phi','M_mode'] = np.zeros((nDOF*nDOF, nDOF*nDOF))
        partials['Lambda','K_mode'] = np.zeros((nDOF, nDOF*nDOF))
        partials['Lambda','M_mode'] = np.zeros((nDOF, nDOF*nDOF))

        vals, vecs = scipy.linalg.eig(K,M)

        ## Values
        for i in range(nDOF):
            partials['Lambda','K_mode'][i,:] += np.ones(nDOF*nDOF)
            partials['Lambda','M_mode'][i,:] += ((vecs[:,i].T)@(vals[i]*vecs[:,i]))*np.ones(nDOF*nDOF)

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
