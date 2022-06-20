import numpy as np
import openmdao.api as om
from modeshape_eig_lhs_1 import ModeshapeEigenLHS1
from modeshape_eig_lhs_2 import ModeshapeEigenLHS2
from modeshape_eig_select import ModeshapeEigSelect

class EigenBal(om.Group):

    def initialize(self):
        self.options.declare('nNode', types=int)
        self.options.declare('nElem', types=int)
        self.options.declare('nDOF', types=int)

    def setup(self):
        nNode = self.options['nNode']        
        nElem = self.options['nElem']        
        nDOF = self.options['nDOF']        
        
        ## Maybe try balance component too?
        #https://openmdao.org/newdocs/versions/latest/features/building_blocks/components/balance_comp.html
        # M@PHI - K@PHI@LAM = 0
        # PHI.T@M@PHI - I = 0

        def guess_phi(inputs, outputs, residuals) :
            A = np.matmul(np.linalg.inv(inputs['M_mode']), inputs['K_mode'])
            w, v = np.linalg.eig(A)
            outputs['phi'] = v

        def guess_lam(inputs, outputs, residuals) :
            A = np.matmul(np.linalg.inv(inputs['M_mode']), inputs['K_mode'])
            w, v = np.linalg.eig(A)
            outputs['lam'] = np.diag(w)

        # lam_init = np.diag(np.arange(1,nDOF+1))
        # phi_init = np.matmul(np.reshape(np.arange(1,nDOF+1),(nDOF,1)), np.reshape(np.arange(1,nDOF+1),(1,nDOF)))

        lam_init = np.ones((nDOF,nDOF))
        phi_init = np.ones((nDOF,nDOF))

        self.add_subsystem('modeshape_eig_lhs_2', 
            ModeshapeEigenLHS2(nDOF=nDOF),
            promotes_inputs=['M_mode', 'K_mode'])

        bal_2 = om.BalanceComp()
        bal_2.add_balance(name= 'phi', val=phi_init)
        # bal_2.options['guess_func'] = guess_phi
        self.add_subsystem(name='bal_2', subsys=bal_2)

        self.connect('bal_2.phi', 'modeshape_eig_lhs_2.eig_vectors')
        self.connect('modeshape_eig_lhs_2.lhs', 'bal_2.lhs:phi')
        
        self.add_subsystem('modeshape_eig_lhs_1', 
            ModeshapeEigenLHS1(nDOF=nDOF),
            promotes_inputs=['M_mode', 'K_mode'])

        bal_1 = om.BalanceComp()
        bal_1.add_balance(name= 'lam', val=lam_init)
        # bal_1.options['guess_func'] = guess_lam
        self.add_subsystem(name='bal_1', subsys=bal_1)

        self.connect('bal_1.lam', 'modeshape_eig_lhs_1.eig_vals')
        self.connect('modeshape_eig_lhs_1.lhs', 'bal_1.lhs:lam')
        self.connect('bal_2.phi', 'modeshape_eig_lhs_1.eig_vectors')

        # self.add_subsystem('modeshape_eigvector', 
        #     ModeshapeEigvector(nNode=nNode,nElem=nElem,nDOF=nDOF), 
        #     promotes_inputs=['A_eig'], 
        #     promotes_outputs=['eig_vector_1', 'eig_freq_1', 'eig_vector_2', 'eig_freq_2', 'eig_vector_3', 'eig_freq_3'])
        self.add_subsystem('modeshape_eig_select', 
            ModeshapeEigSelect(nNode=nNode,nElem=nElem,nDOF=nDOF), 
            promotes_outputs=['eig_vector_1', 'eig_freq_1', 'eig_vector_2', 'eig_freq_2', 'eig_vector_3', 'eig_freq_3'])
        
        self.connect('bal_2.phi', 'modeshape_eig_select.eig_vectors')
        self.connect('bal_1.lam', 'modeshape_eig_select.eig_vals')