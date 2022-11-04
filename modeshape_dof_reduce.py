import numpy as np

from openmdao.api import ExplicitComponent


class ModeshapeDOFReduce(ExplicitComponent):

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nDOF_tot = self.nodal_data['nDOF_tot']
        nDOF_r = self.nodal_data['nDOF_r']

        self.add_input('M_glob', val=np.zeros((nDOF_tot, nDOF_tot)), units='kg')
        self.add_input('K_glob', val=np.zeros((nDOF_tot, nDOF_tot)), units='N/m')
    
        self.add_output('Mr_glob', val=np.zeros((nDOF_r, nDOF_r)), units='kg')
        self.add_output('Kr_glob', val=np.zeros((nDOF_r, nDOF_r)), units='N/m')

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

        self.declare_partials('Mr_glob', 'M_glob', rows=Hrows, cols=Hcols)
        self.declare_partials('Kr_glob', 'K_glob', rows=Hrows, cols=Hcols)

    def compute(self, inputs, outputs):
        Tr = self.nodal_data['Tr']

        M_glob = inputs['M_glob']
        K_glob = inputs['K_glob']

        Mr = (Tr.T).dot(M_glob).dot(Tr)
        Kr = (Tr.T).dot(K_glob).dot(Tr)       
       
        outputs['Mr_glob'] = Mr
        outputs['Kr_glob'] = Kr

    def compute_partials(self, inputs, partials):
        nDOF_r = self.nodal_data['nDOF_r']

        partials['Mr_glob', 'M_glob'] = np.ones(nDOF_r * nDOF_r)
        partials['Kr_glob', 'K_glob'] = np.ones(nDOF_r * nDOF_r)