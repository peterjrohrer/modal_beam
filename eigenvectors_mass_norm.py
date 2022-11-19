import numpy as np
import openmdao.api as om

class EigenvecsMassNorm(om.ExplicitComponent):
    # Mass normalize eigenvectors

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nDOF_r = self.nodal_data['nDOF_r']

        self.add_input('M_mode_eig', val=np.zeros((nDOF_r, nDOF_r)), units='kg')
        self.add_input('Q_raw', val=np.zeros((nDOF_r, nDOF_r)))

        self.add_output('Q_mass_norm', val=np.zeros((nDOF_r, nDOF_r)))

    def setup_partials(self):
        nDOF = self.nodal_data['nDOF_r']
        nPart = nDOF*nDOF
        Hrows = np.arange(0,nPart,nDOF)
        Hcols = np.repeat(0,nDOF)
        for i in range(1,nDOF):
            Hrow_add = np.arange(i,nPart,nDOF)
            Hrows = np.concatenate((Hrows,Hrow_add))
            Hcol_add = np.repeat(np.arange(i,nPart,nDOF)[i],nDOF)
            Hcols = np.concatenate((Hcols,Hcol_add))

        self.declare_partials('Q_mass_norm','M_mode_eig', rows=Hrows, cols=Hcols)#, cols=Hcols)
        self.declare_partials('Q_mass_norm','Q_raw', rows=np.arange(nPart), cols=np.arange(nPart))
        
    def compute(self, inputs, outputs):
        nDOF = self.nodal_data['nDOF_r']
        Q = inputs['Q_raw']
        M_modal = np.diag(inputs['M_mode_eig'])
        Q_norm = np.zeros_like(Q)
                
        for j in range(nDOF):
            modalmass_j = M_modal[j]
            # Quick fix to avoid problems with NaN
            if modalmass_j < 0. :
                print('[WARN] Found negative modal mass at position %d' %j)
                modalmass_j = 1.0e-6

            Q_norm[:,j] = Q[:,j]/np.sqrt(modalmass_j)

        outputs['Q_mass_norm'] = Q_norm

    def compute_partials(self, inputs, partials):
        nDOF = self.nodal_data['nDOF_r']
        nPart = nDOF * nDOF
        Q = inputs['Q_raw']
        M_modal = np.diag(inputs['M_mode_eig'])

        partials['Q_mass_norm', 'M_mode_eig'] = np.zeros((nDOF*nDOF))
        partials['Q_mass_norm', 'Q_raw'] = np.zeros((nDOF*nDOF))

        for j in range(nDOF):
            idx1 = j*nDOF
            idx2 = (j+1)*nDOF
            modalmass_j = M_modal[j]
            if modalmass_j > 0. :
                partials['Q_mass_norm', 'M_mode_eig'][idx1:idx2] += Q[:,j] / (-2. * (modalmass_j ** (1.5)))
            
            for k in range(nDOF):
                modalmass_k = M_modal[k]
                # Quick fix to avoid problems with NaN
                if modalmass_k < 0. :
                    modalmass_k = 1.0e-6
                partials['Q_mass_norm', 'Q_raw'][(j*nDOF)+k] += 1./np.sqrt(modalmass_k)