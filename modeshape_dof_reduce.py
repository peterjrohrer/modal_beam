import numpy as np

from openmdao.api import ExplicitComponent


class ModeshapeDOFReduce(ExplicitComponent):

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nDOF_tot = self.nodal_data['nDOF_tot']
        nDOF_r = self.nodal_data['nDOF_r']

        self.add_input('Tr', val=np.zeros((nDOF_tot, nDOF_r)))
        self.add_input('M_glob', val=np.zeros((nDOF_tot, nDOF_tot)), units='kg')
        self.add_input('K_glob', val=np.zeros((nDOF_tot, nDOF_tot)), units='N/m')
    
        self.add_output('Mr_glob', val=np.zeros((nDOF_r, nDOF_r)), units='kg')
        self.add_output('Kr_glob', val=np.zeros((nDOF_r, nDOF_r)), units='N/m')
        self.add_output('A_glob', val=np.zeros((nDOF_tot, nDOF_tot)), units='N/m')

    def setup_partials(self):
        nDOF_tot = self.nodal_data['nDOF_tot']
        nDOF_r = self.nodal_data['nDOF_r']
        
        Hrows = np.arange(nDOF_r * nDOF_r)
        Hcols = np.arange(nDOF_tot * nDOF_tot)
        IDOF_removed = np.array(self.nodal_data['IDOF_removed'])

        for i in range(len(IDOF_removed)):
            removed_nodes = np.arange((IDOF_removed[i]*nDOF_tot),((IDOF_removed[i]+1)*nDOF_tot))
            Hcols = np.setdiff1d(Hcols,removed_nodes)
        for i in range(nDOF_tot):
            removed_DOF = (i*nDOF_tot) + IDOF_removed 
            Hcols = np.setdiff1d(Hcols,removed_DOF)

        self.declare_partials('Mr_glob', 'M_glob', rows=Hrows, cols=Hcols, val=np.ones(nDOF_r * nDOF_r))
        self.declare_partials('Kr_glob', 'K_glob', rows=Hrows, cols=Hcols, val=np.ones(nDOF_r * nDOF_r))

        y1 = np.einsum('lj,ki->klij',np.eye(nDOF_r),np.ones((nDOF_r,nDOF_tot)))
        y2 = np.einsum('kj,il->klij',np.eye(nDOF_r),np.ones((nDOF_tot,nDOF_r)))
        b = np.reshape((y1 + y2), (nDOF_r * nDOF_r, nDOF_tot * nDOF_r))
        Grows = np.nonzero(b)[0]
        Gcols = np.nonzero(b)[1]

        self.declare_partials('Mr_glob', 'Tr', rows=Grows, cols=Gcols)
        self.declare_partials('Kr_glob', 'Tr', rows=Grows, cols=Gcols)
        self.declare_partials('A_glob', 'M_glob')
        self.declare_partials('A_glob', 'K_glob')

    def compute(self, inputs, outputs):
        Tr = inputs['Tr']

        M_glob = inputs['M_glob']
        K_glob = inputs['K_glob']

        Mr = (Tr.T) @ (M_glob) @ (Tr)
        Kr = (Tr.T) @ (K_glob) @ (Tr)       
       
        outputs['Mr_glob'] = Mr
        outputs['Kr_glob'] = Kr
        outputs['A_glob'] = M_glob @ K_glob

    def compute_partials(self, inputs, partials):
        nDOF_tot = self.nodal_data['nDOF_tot']
        nDOF_r = self.nodal_data['nDOF_r']

        Tr = inputs['Tr']

        M_glob = inputs['M_glob']
        K_glob = inputs['K_glob']

        dMr_dTr = np.zeros((nDOF_r,nDOF_r,nDOF_tot,nDOF_r))
        dKr_dTr = np.zeros((nDOF_r,nDOF_r,nDOF_tot,nDOF_r))
        for i in range(nDOF_tot):
            for j in range(nDOF_r):
                J_ij = np.zeros_like(Tr)
                J_ij[i,j] += 1.
                J_ji = J_ij.T
                dMr_dTr[:,:,i,j] += (Tr.T @ M_glob @ J_ij) + (J_ji @ M_glob @ Tr)
                dKr_dTr[:,:,i,j] += (Tr.T @ K_glob @ J_ij) + (J_ji @ K_glob @ Tr)

        a = np.reshape(dMr_dTr, (nDOF_r * nDOF_r, nDOF_tot * nDOF_r))
    
        y1 = np.einsum('lj,ki->klij',np.ones((nDOF_r,1)),(Tr.T @ M_glob))
        y2 = np.einsum('kj,il->klij',np.ones((nDOF_r,1)),(M_glob @ Tr))
        b = np.reshape(y1+y2, (nDOF_r * nDOF_r * nDOF_tot ))

        partials['Mr_glob', 'Tr'] = b
        partials['Kr_glob', 'Tr'] = np.reshape(dKr_dTr, (nDOF_r * nDOF_r, nDOF_tot * nDOF_r))
        partials['A_glob', 'M_glob'] = np.kron(np.eye(nDOF_tot),K_glob)
        partials['A_glob', 'K_glob'] = np.kron(M_glob, np.eye(nDOF_tot))