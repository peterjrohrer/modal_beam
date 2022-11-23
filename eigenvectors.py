import numpy as np
import scipy.linalg
import openmdao.api as om

class Eigenvecs(om.ExplicitComponent):
    # Solution to eigenvalue problem with eigenvectors for manipulation

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nDOF_r = self.nodal_data['nDOF_r']
        nDOF_tot = self.nodal_data['nDOF_tot']

        self.add_input('Ar_eig', val=np.ones((nDOF_r, nDOF_r)))

        self.add_output('Q_raw', val=np.ones((nDOF_r, nDOF_r)))

    def compute(self, inputs, outputs):
        nDOF = self.nodal_data['nDOF_r']
        A = inputs['Ar_eig']
                
        # --- To match Giles (2008) paper
        # https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
        D, Q = scipy.linalg.eig(a=A, b=None, left=False, right=True)

        outputs['Q_raw'] = Q

        self.Q = Q
        self.D = np.diag(D) # Must keep complex eigenvalues to have correct derivatives    

        ## TODO handle repeated eigenvalues!
        E = np.zeros((nDOF, nDOF), dtype=complex)
        F = np.zeros((nDOF, nDOF), dtype=complex)
        for i in range(nDOF):
            for j in range(nDOF):
                if i == j:
                    pass
                else:
                    E[i,j] = (D[j]-D[i])
                    F[i,j] = 1./(D[j]-D[i])
        self.E_matrix = E
        self.F_matrix = F

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        ##TODO persistent questionable partials??
        nDOF = self.nodal_data['nDOF_r']
        Q = self.Q
        D = self.D
        A = inputs['Ar_eig']
        E = self.E_matrix
        F = self.F_matrix

        ## Based on Giles (2008), https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
        if mode == 'rev':    
            if 'Q_raw' in d_outputs:
                if 'Ar_eig' in d_inputs:
                    d_inputs['Ar_eig'] += np.real(scipy.linalg.inv(Q).T @ (np.multiply(F, (Q.T @ d_outputs['Q_raw']))) @ Q.T)

        elif mode == 'fwd':
            if 'Q_raw' in d_outputs:
                if 'Ar_eig' in d_inputs:
                    d_outputs['Q_raw'] += np.real(Q @ np.multiply(F, (np.linalg.inv(Q) @ d_inputs['Ar_eig'] @ Q)))