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
        nElem = self.nodal_data['nElem']
        nDOF = self.nodal_data['nDOF_tot']
        nPart = nElem * 12 * 12

        Hrows1 = Hrows0 = np.arange(12)
        for i in range(1,12):
            Hrows_add = (i * nDOF) + Hrows0
            Hrows1 = np.concatenate((Hrows1,Hrows_add),axis=0)
        
        Hrows = np.array([])
        for i in range(nElem):
            Hrows_add = (i * ((nDOF+1) * 6)) + Hrows1
            Hrows = np.concatenate((Hrows,Hrows_add),axis=0)

        self.declare_partials('K_glob_pre', 'kel', rows=Hrows, cols=np.arange(nPart), val=np.ones(nPart))

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