import numpy as np
from openmdao.api import ExplicitComponent

class EigenSantize(ExplicitComponent):

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nDOF = self.nodal_data['nDOF_tot']
        nMode = self.nodal_data['nMode']

        self.add_input('Q_unnorm', val=np.zeros((nDOF, nMode)))
    
        self.add_output('Q', val=np.zeros((nDOF, nMode)))

    def setup_partials(self):
        nDOF = self.nodal_data['nDOF_tot']
        nMode = self.nodal_data['nMode']
        
        self.declare_partials('Q', 'Q_unnorm')

    def compute(self, inputs, outputs):
        nDOF = self.nodal_data['nDOF_tot']
        nMode = self.nodal_data['nMode']
        Tr = self.nodal_data['Tr']
        Q = inputs['Q_unnorm']

        # --- Renormalize modes 
        self.argmax_idx = np.zeros(nMode)
        self.scales = np.zeros(nMode)
        for j in range(nMode):
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

        outputs['Q'] = Q
    
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
        Q = inputs['Q_unnorm']

        partials['Q', 'Q_unnorm'] = np.eye((nDOF_tot * nMode))  


        ##TODO need to revisit this partial
        # Q_part = np.zeros_like(partials['Q', 'Q_unnorm'])  
        # scale_idx = np.zeros(nMode, dtype=int)
        # for i in range(nMode):
        #     scale_idx[i] = self.argmax_idx[i] + (i * nDOF_tot)
        
        # for j in range(nMode):
        #     for i in range(nDOF_tot):
        #         ix1 = (j * nDOF_tot) + i
        #         Q_part[ix1,scale_idx[j]] += -1. / (self.scales[j] * self.scales[j])
        
        # partials['Q', 'Q_unnorm'] += Q_part
            
        # # --- Renormalize modes 
        # for j in range(nMode):
        #     q_j = Q[:,j]
        #     iMax = np.argmax(np.abs(q_j))
        #     scale = q_j[iMax] # not using abs to normalize to "1" and not "+/-1"
        #     Q[:,j]= Q[:,j]/scale
        
        # Q_part = np.zeros((nDOF_tot, nMode, nDOF_tot, nMode))
        # for i in range(nMode):
        #     for j in range(nDOF_tot):
        #         if j == self.argmax_idx[i]:
        #             Q_part[:,i,j,i] += -1. / (self.scales[i] * self.scales[i])

        # partials['Q', 'Q_unnorm'] += np.reshape(Q_part, ((nDOF_tot * nMode),(nDOF_tot * nMode)))