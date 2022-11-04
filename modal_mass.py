import numpy as np
from openmdao.api import ExplicitComponent

class ModalMass(ExplicitComponent):
    # Calculate modal mass for TLPWT 
    
    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nDOF_tot = self.nodal_data['nDOF_tot']
        nMode = self.nodal_data['nMode']

        self.add_input('M_glob', val=np.zeros((nDOF_tot, nDOF_tot)), units='kg')
        self.add_input('Q', val=np.ones((nDOF_tot, nMode)))

        self.add_output('M_modal', val=np.zeros((nMode,nMode)), units='kg')

    def	setup_partials(self):
        self.declare_partials('M_modal', ['M_glob', 'Q'])

    def compute(self, inputs, outputs):
        nMode = self.nodal_data['nMode']

        Q = inputs['Q']
        M_glob = inputs['M_glob']
        M_modal = Q.T @ M_glob @ Q
        
        outputs['M_modal'] = M_modal

    def compute_partials(self, inputs, partials):
        nDOF_tot = self.nodal_data['nDOF_tot']
        nMode = self.nodal_data['nMode']

        Q = inputs['Q']
        M_glob = inputs['M_glob']

        partials['M_modal', 'M_glob'] = np.kron(Q.T,Q.T)

        # --- Somewhat hacky solution         
        partials['M_modal', 'Q'] = np.kron((M_glob@Q).T,np.eye(nMode))
        rows = []
        for i in range(nMode):
            row = []
            for j in range(nDOF_tot):
                block = np.zeros((nMode,nMode))
                E = np.zeros_like(Q)
                E[j,i] += 1.
                block += (Q.T @ M_glob @ E)
                row.append(block)
            row_concat = np.concatenate(row,axis=1)
            rows.append(row_concat)
        blocked = np.concatenate(rows,axis=0)
        partials['M_modal', 'Q'] += blocked

        # # --- Theoretical Solution         
        # rows = []
        # for i in range(nMode):
        #     row = []
        #     for j in range(nDOF_tot):
        #         block = np.zeros((nMode,nMode))
        #         E = np.zeros_like(Q)
        #         E[j,i] += 1.
        #         block += (Q.T @ M_glob @ E)
        #         block += (E.T @ (M_glob @ Q))
        #         row.append(block)
        #     row_concat = np.concatenate(row,axis=1)
        #     rows.append(row_concat)
        # blocked = np.concatenate(rows,axis=0)
        # partials['M_modal', 'Q'] += blocked