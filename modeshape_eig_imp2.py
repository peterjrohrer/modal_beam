import numpy as np
import scipy
import openmdao.api as om


class ModeshapeEigenImp2(om.ImplicitComponent):
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
        self.add_input('eig_vectors', val=np.zeros((nDOF, nDOF)))

        self.add_output('eig_vals', val=np.zeros((nDOF, nDOF)))

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')

    def apply_nonlinear(self, inputs, outputs, residuals):
        nDOF = self.options['nDOF']
        K = inputs['K_mode']
        M = inputs['M_mode']
        PHI = inputs['eig_vectors']
        LAM = outputs['eig_vals']

        residuals['eig_vals'] = np.matmul(M,PHI) - np.matmul(K,np.matmul(PHI,LAM))

    # def solve_nonlinear(self, inputs, outputs):
    #     nDOF = self.options['nDOF']
    #     K = inputs['K_mode']
    #     M = inputs['M_mode']

    #     w,v = scipy.linalg.eig(K,M)

    #     outputs['eig_vals'] = np.diag(w)

    # def linearize(self, inputs, outputs, partials):
    #     nDOF = self.options['nDOF']
    #     K = inputs['K_mode']
    #     M = inputs['M_mode']
    #     PHI = outputs['eig_vectors']
    #     LAM = outputs['eig_vals']

    #     partials['eig_vals','M_mode'] = 0.
    #     partials['eig_vals','K_mode'] = 0.
    #     partials['eig_vals','eig_vectors'] = 0.
    #     partials['eig_vals','eig_vals'] = -1. * np.matmul(K,PHI)

    def guess_nonlinear(self, inputs, outputs):
        nDOF = self.options['nDOF']
        K = inputs['K_mode']
        M = inputs['M_mode']

        w,v = scipy.linalg.eig(K,M)

        outputs['eig_vals'] = np.diag(w)
