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

        self.add_input('Ar_eig', val=np.ones((nDOF_r, nDOF_r)))

        self.add_output('Q_raw', val=np.ones((nDOF_r, nDOF_r)))
        self.add_output('eig_freqs_raw', val=np.zeros((nDOF_r, nDOF_r)), units='1/s')

    # def setup_partials(self):
    #     self.declare_partials('Q_raw', ['Mr_glob', 'Kr_glob'])
    #     self.declare_partials('eig_freqs_raw', ['Mr_glob', 'Kr_glob'])

    def compute(self, inputs, outputs):
        nDOF = self.nodal_data['nDOF_r']
        A = inputs['Ar_eig']
                
        # --- To match Giles (2008) paper
        # https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
        D, Q = scipy.linalg.eig(a=A, b=None, left=False, right=True)

        outputs['Q_raw'] = Q
        outputs['eig_freqs_raw'] = np.diag(D)

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
            if 'eig_freqs_raw' in d_outputs:
                if 'Ar_eig' in d_inputs:
                    d_inputs['Ar_eig'] += np.real(scipy.linalg.inv(Q).T @ (d_outputs['eig_freqs_raw']) @ Q.T)

        elif mode == 'fwd':
            if 'Q_raw' in d_outputs:
                if 'Ar_eig' in d_inputs:
                    d_outputs['Q_raw'] += np.real(Q @ np.multiply(F, (np.linalg.inv(Q) @ d_inputs['Ar_eig'] @ Q)))
            if 'eig_freqs_raw' in d_outputs:
                if 'Ar_eig' in d_inputs:
                    d_outputs['eig_freqs_raw'] += np.real(np.multiply(np.eye(nDOF), (np.linalg.inv(Q) @ d_inputs['Ar_eig'] @ Q)))