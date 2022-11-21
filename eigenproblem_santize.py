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

        self.add_input('Q_all', val=np.zeros((nDOF_tot, nDOF_r)))
        self.add_input('eigenvals_sorted', val=np.zeros(nDOF_r))
    
        self.add_output('Q_full', val=np.zeros((nDOF_tot, nDOF_r)))
        self.add_output('eig_freqs_full', val=np.zeros(nDOF_r), units='1/s')

    def setup_partials(self):
        nDOF_tot = self.nodal_data['nDOF_tot']
        nDOF_r = self.nodal_data['nDOF_r']
        
        self.declare_partials('Q_full', 'Q_all')
        self.declare_partials('eig_freqs_full', 'eigenvals_sorted', rows=np.arange(nDOF_r), cols=np.arange(nDOF_r))

    def compute(self, inputs, outputs):
        nDOF_tot = self.nodal_data['nDOF_tot']
        nDOF_r = self.nodal_data['nDOF_r']
        Tr = self.nodal_data['Tr']
        Q = inputs['Q_all']
        LambdaDiag = inputs['eigenvals_sorted']

        # Export frequencies
        Lambda = np.sqrt(np.real(LambdaDiag))/(2*np.pi) # frequencies [Hz]
        outputs['eig_freqs_full'] = Lambda

        # --- Renormalize modes 
        self.argmax_idx = np.zeros(nDOF_r)
        self.scales = np.zeros(nDOF_r)
        for j in range(nDOF_r):
            q_j = Q[:,j]
            iMax = np.argmax(np.abs(q_j))
            scale = q_j[iMax] # not using abs to normalize to "1" and not "+/-1"
            Q[:,j]= Q[:,j]/scale
            self.argmax_idx[j] = iMax
            self.scales[j] = scale
        
        # --- Sanitization, ensure real values (for export, not for derivatives!)
        Q_im = np.imag(Q)
        Q = np.real(Q)
        imm = np.mean(np.abs(Q_im), axis=0)
        bb = imm>0
        if sum(bb)>0:
            W=list(np.where(bb)[0])
            print('[WARN] Found {:d} complex eigenvectors at positions {}/{}'.format(sum(bb),W,Q.shape[0]))

        outputs['Q_full'] = Q
    
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
        nDOF_r = self.nodal_data['nDOF_r']
        Q = inputs['Q_all']
        LambdaDiag = inputs['eigenvals_sorted']

        partials['Q_full', 'Q_all'] = np.eye((nDOF_tot * nDOF_r))  

        partials['eig_freqs_full', 'eigenvals_sorted'] = np.zeros(nDOF_r)
        for i in range(nDOF_r):
            partials['eig_freqs_full', 'eigenvals_sorted'][i] += 1. / ((4.*np.pi)*np.sqrt(np.real(LambdaDiag[i])))

        ##TODO need to revisit this partial
        # Q_part = np.zeros_like(partials['Q_full', 'Q_all'])  
        # scale_idx = np.zeros(nDOF_r, dtype=int)
        # for i in range(nDOF_r):
        #     scale_idx[i] = self.argmax_idx[i] + (i * nDOF_tot)
        
        # for j in range(nDOF_r):
        #     for i in range(nDOF_tot):
        #         ix1 = (j * nDOF_tot) + i
        #         Q_part[ix1,scale_idx[j]] += -1. / (self.scales[j] * self.scales[j])
        
        # partials['Q_full', 'Q_all'] += Q_part
            
        # # --- Renormalize modes 
        # for j in range(nDOF_r):
        #     q_j = Q[:,j]
        #     iMax = np.argmax(np.abs(q_j))
        #     scale = q_j[iMax] # not using abs to normalize to "1" and not "+/-1"
        #     Q[:,j]= Q[:,j]/scale
        
        # Q_part = np.zeros((nDOF_tot, nDOF_r, nDOF_tot, nDOF_r))
        # for i in range(nDOF_r):
        #     for j in range(nDOF_tot):
        #         if j == self.argmax_idx[i]:
        #             Q_part[:,i,j,i] += -1. / (self.scales[i] * self.scales[i])

        # partials['Q_full', 'Q_all'] += np.reshape(Q_part, ((nDOF_tot * nDOF_r),(nDOF_tot * nDOF_r)))