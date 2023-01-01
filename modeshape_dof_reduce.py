import numpy as np
import scipy.signal
import scipy.sparse
import opt_einsum as oe
import sparse as sps
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
        Tr_mask = self.nodal_data['Tr_mask']
        
        sTr = scipy.sparse.eye(nDOF_tot, nDOF_r)
        sTr_offD = scipy.sparse.coo_array((np.ones(len(Tr_mask[0])), (Tr_mask[0],Tr_mask[1])))
        sTr += sTr_offD
        sparse_dr_d = scipy.sparse.kron(sTr.T,sTr.T)
        self.declare_partials('Mr_glob', 'M_glob',val=sparse_dr_d)
        self.declare_partials('Kr_glob', 'K_glob',val=sparse_dr_d)
        self.declare_partials('Mr_glob', 'K_glob', dependent=False)
        self.declare_partials('Kr_glob', 'M_glob', dependent=False)

        s_glob = scipy.sparse.diags(diagonals=np.repeat(1.,13), offsets=np.arange(-6,7), shape=(nDOF_tot,nDOF_tot), dtype=bool)
        dTr_shape = (nDOF_r * nDOF_r, nDOF_tot * nDOF_r)
        dr1 = oe.contract('lj,ki->klij', sps.eye(nDOF_r, dtype=bool), sps.COO((sTr.T @ s_glob))).reshape(dTr_shape)
        dr2 = oe.contract('kj,il->klij', sps.eye(nDOF_r, dtype=bool), sps.COO((s_glob @ sTr))).reshape(dTr_shape)
        self.declare_partials('Mr_glob', 'Tr', val=(dr1 + dr2).to_scipy_sparse())
        self.declare_partials('Kr_glob', 'Tr', val=(dr1 + dr2).to_scipy_sparse())

        self.declare_partials('A_glob', 'M_glob')
        self.declare_partials('A_glob', 'K_glob')

    def compute(self, inputs, outputs):
        Tr = inputs['Tr']

        M_glob = inputs['M_glob']
        K_glob = inputs['K_glob']
       
        outputs['Mr_glob'] = (Tr.T) @ (M_glob) @ (Tr)
        outputs['Kr_glob'] = (Tr.T) @ (K_glob) @ (Tr)
        outputs['A_glob'] = M_glob @ K_glob

    def compute_partials(self, inputs, partials):
        nDOF_tot = self.nodal_data['nDOF_tot']
        nDOF_r = self.nodal_data['nDOF_r']

        sTr = scipy.sparse.coo_array(inputs['Tr'])
        sM_glob = scipy.sparse.coo_array(inputs['M_glob'])
        sK_glob = scipy.sparse.coo_array(inputs['K_glob'])

        partials['Mr_glob', 'M_glob'] = scipy.sparse.kron(sTr.T,sTr.T)
        partials['Kr_glob', 'K_glob'] = scipy.sparse.kron(sTr.T,sTr.T)

        partials['A_glob', 'M_glob'] = scipy.sparse.kron(scipy.sparse.eye(nDOF_tot),sK_glob)
        partials['A_glob', 'K_glob'] = scipy.sparse.kron(sM_glob, scipy.sparse.eye(nDOF_tot))

        # Tr = inputs['Tr']
        # M_glob = inputs['M_glob']
        # K_glob = inputs['K_glob']
        
        # dMr_dTr = np.empty(Tr.shape, dtype=object)
        # for i in range(nDOF_tot):
        #     for j in range(nDOF_r):
        #         J_ij = scipy.sparse.coo_array(scipy.signal.unit_impulse(Tr.shape,(i,j),dtype=bool))
        #         J_ji = J_ij.T
        #         dMr_dTr[i,j] = ((sTr.T @ sM_glob @ J_ij) + (J_ji @ sM_glob @ sTr))

        dTr_shape = (nDOF_r * nDOF_r, nDOF_tot * nDOF_r)
        dMr1 = oe.contract('lj,ki->klij',sps.COO(np.eye(nDOF_r)),sps.COO((sTr.T @ sM_glob))).reshape(dTr_shape)
        dMr2 = oe.contract('kj,il->klij',sps.COO(np.eye(nDOF_r)),sps.COO((sM_glob @ sTr))).reshape(dTr_shape)
        dMr_dTr = (dMr1 + dMr2).to_scipy_sparse()
        dKr1 = oe.contract('lj,ki->klij',sps.COO(np.eye(nDOF_r)),sps.COO((sTr.T @ sK_glob))).reshape(dTr_shape)
        dKr2 = oe.contract('kj,il->klij',sps.COO(np.eye(nDOF_r)),sps.COO((sK_glob @ sTr))).reshape(dTr_shape)
        dKr_dTr = (dKr1 + dKr2).to_scipy_sparse()
        
        # dMr1 = np.einsum('kj,il->klij',np.eye(nDOF_r),(Tr.T @ M_glob))
        # dMr2 = np.einsum('kj,il->klij',np.eye(nDOF_r),(M_glob @ Tr))
        # dMr_dTr = np.reshape(dMr1+dMr2, (nDOF_r * nDOF_r, nDOF_tot * nDOF_r))
        
        # dMr1 = (Tr.T @ M_glob)[:,None,:,None] * np.eye(nDOF_r)[None,:,None,:]
        # dMr2 = np.eye(nDOF_r)[:,None,None,:] * (M_glob @ Tr).T[None,:,:,None]
        # dKr1 = (Tr.T @ K_glob)[:,None,:,None] * np.eye(nDOF_r)[None,:,None,:]
        # dKr2 = np.eye(nDOF_r)[:,None,None,:] * (K_glob @ Tr).T[None,:,:,None]

        # sMr1 = scipy.sparse.coo_array(dMr1.reshape(nDOF_r * nDOF_r, nDOF_tot * nDOF_r))
        # sMr2 = scipy.sparse.coo_array(dMr2.reshape(nDOF_r * nDOF_r, nDOF_tot * nDOF_r))
        # sKr1 = scipy.sparse.coo_array(dKr1.reshape(nDOF_r * nDOF_r, nDOF_tot * nDOF_r))
        # sKr2 = scipy.sparse.coo_array(dKr2.reshape(nDOF_r * nDOF_r, nDOF_tot * nDOF_r))

        # dMr_dTr = dMr1.reshape(nDOF_r * nDOF_r, nDOF_tot * nDOF_r) + dMr2.reshape(nDOF_r * nDOF_r, nDOF_tot * nDOF_r)

        # dMr1 = np.kron(np.eye(nDOF_r),(Tr.T @ M_glob))
        # dMr2 = np.kron((M_glob @ Tr), np.eye(nDOF_r)).T
        # dMr_dTr = dMr1 + dMr2 

        partials['Mr_glob', 'Tr'] = dMr_dTr
        partials['Kr_glob', 'Tr'] = dKr_dTr