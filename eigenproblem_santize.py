import numpy as np
from openmdao.api import ExplicitComponent

class EigenSantize(ExplicitComponent):

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nDOF_tot = self.nodal_data['nDOF_tot']
        nDOF_r = self.nodal_data['nDOF_r']
        nMode = self.nodal_data['nMode']

        self.add_input('Q_mass_norm', val=np.zeros((nDOF_r, nDOF_r)))
        self.add_input('eigenvals_raw', val=np.zeros((nDOF_r, nDOF_r)))
    
        self.add_output('Q_full', val=np.zeros((nDOF_tot, nDOF_r)))
        self.add_output('eig_freqs_full', val=np.zeros(nDOF_r), units='1/s')

    def setup_partials(self):
        nDOF_tot = self.nodal_data['nDOF_tot']
        nDOF_r = self.nodal_data['nDOF_r']
        nMode = self.nodal_data['nMode']
        
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        nDOF = self.nodal_data['nDOF_r']
        Tr = self.nodal_data['Tr']
        Q = inputs['Q_mass_norm']
        D = inputs['eigenvals_raw']

        # Sort and diagonalize
        lambdaDiag = np.diag(D)
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
        
        # --- Sanitization, ensure real values (for export, not for derivatives!)
        Q_im = np.imag(Q)
        Q = np.real(Q)
        imm = np.mean(np.abs(Q_im),axis = 0)
        bb = imm>0
        if sum(bb)>0:
            W=list(np.where(bb)[0])
            print('[WARN] Found {:d} complex eigenvectors at positions {}/{}'.format(sum(bb),W,Q.shape[0]))
        Lambda = np.real(Lambda)

        outputs['Q_full'] = Q
        outputs['eig_freqs_full'] = Lambda
    
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


    def compute_partials(self, inputs, partials):
        nDOF_tot = self.nodal_data['nDOF_tot']
        nMode = self.nodal_data['nMode']

        partials['Q', 'Q_full'] = np.ones(nDOF_tot * nMode)        
        partials['eig_freqs', 'eig_freqs_full'] = np.ones(nMode)        
