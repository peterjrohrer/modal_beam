import numpy as np
from scipy import linalg, sparse
from openmdao.api import ImplicitComponent

class ModeshapeEigmatrixImp(ImplicitComponent):
    # Assemble global eigenmatrix
    # Based on OpenMDAO LinearSystemComp 

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nDOF = self.nodal_data['nDOF_r']

        self.add_input('Mr_glob', val=np.zeros((nDOF, nDOF)), units='kg')
        self.add_input('Kr_glob', val=np.zeros((nDOF, nDOF)), units='N/m')

        self.add_output('Ar_eig', val=np.zeros((nDOF, nDOF)), ref=1.e7)

    def setup_partials(self):
        nDOF = self.nodal_data['nDOF_r']

        row_col = np.arange(nDOF*nDOF, dtype=int)
        self.declare_partials('Ar_eig', 'Kr_glob', val=np.full(nDOF*nDOF, -1.0), rows=row_col, cols=row_col)

        rows = np.repeat(np.arange(nDOF*nDOF, dtype=int), nDOF)
        cols = np.array([])
        for i in range(nDOF):
            cols_add = np.tile(np.arange(nDOF, dtype=int), nDOF) + (nDOF*i)
            cols = np.concatenate((cols,cols_add))

        self.declare_partials('Ar_eig', 'Mr_glob', rows=rows, cols=cols)

        cols = np.array([])
        for i in range(nDOF):
            cols_add = np.arange(i, nDOF*nDOF, nDOF, dtype=int)
            cols = np.concatenate((cols,cols_add))
            
        self.declare_partials(of='Ar_eig', wrt='Ar_eig', rows=rows, cols=np.tile(cols,nDOF))

    def apply_nonlinear(self, inputs, outputs, residuals):
        ## R = Ax - b.
        residuals['Ar_eig'] = (inputs['Mr_glob'] @ outputs['Ar_eig']) - inputs['Kr_glob']

    def solve_nonlinear(self, inputs, outputs):
        ## Use numpy to solve Ax=b for x.
        nDOF = self.nodal_data['nDOF_r']

        # lu factorization for use with solve_linear
        self._lup  = linalg.lu_factor(inputs['Mr_glob'])
        outputs['Ar_eig'] = linalg.lu_solve(self._lup, inputs['Kr_glob'])

    def linearize(self, inputs, outputs, partials):
        ## Compute the non-constant partial derivatives.
        nDOF = self.nodal_data['nDOF_r']
        x = outputs['Ar_eig']
        vec_size = nDOF
        size = nDOF

        partials['Ar_eig', 'Mr_glob'] = np.tile(outputs['Ar_eig'], nDOF).flatten(order='F')
        partials['Ar_eig', 'Ar_eig'] = np.tile(inputs['Mr_glob'], nDOF).flatten()

    def solve_linear(self, d_outputs, d_residuals, mode):
        ## Back-substitution to solve the derivatives of the linear system.
        
        if mode == 'fwd':
            d_outputs['Ar_eig'] = linalg.lu_solve(self._lup, d_residuals['Ar_eig'], trans=0)
        elif mode == 'rev': 
            d_residuals['Ar_eig'] = linalg.lu_solve(self._lup, d_outputs['Ar_eig'], trans=1)
        