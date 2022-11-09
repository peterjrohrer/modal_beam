import numpy as np

from openmdao.api import ExplicitComponent

class ModeshapeGlobStiff(ExplicitComponent):

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nElem = self.nodal_data['nElem']
        nDOF = self.nodal_data['nDOF_tot']

        self.add_input('kel', val=np.zeros((nElem, 12, 12)), units='N/m')
    
        self.add_output('K_glob_pre', val=np.zeros((nDOF, nDOF)), units='N/m')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        nElem = self.nodal_data['nElem']
        nDOF = self.nodal_data['nDOF_tot']
        Elem2DOF = self.nodal_data['Elem2DOF']

        kel = inputs['kel']
       
        K_glob = np.zeros((nDOF, nDOF))
        for k in range(nElem):
            DOFindex=Elem2DOF[k,:]

            for i,ii in enumerate(DOFindex):
                for j,jj in enumerate(DOFindex):
                    K_glob[ii,jj] += kel[k,i,j]

        outputs['K_glob_pre'] = K_glob

    def compute_partials(self, inputs, partials):
        nElem = self.nodal_data['nElem']
        nDOF = self.nodal_data['nDOF_tot']
        Elem2DOF = self.nodal_data['Elem2DOF']

        partials['K_glob_pre', 'kel'] = np.zeros(((nDOF*nDOF),(nElem*12*12)))

        for k in range(nElem):
            DOFindex=Elem2DOF[k,:]
            for i,ii in enumerate(DOFindex):
                for j,jj in enumerate(DOFindex):
                    glob_idx = (ii*36)+jj
                    loc_idx = (144*k) + (i*12) +j

                    partials['K_glob_pre', 'kel'][glob_idx,loc_idx] += 1