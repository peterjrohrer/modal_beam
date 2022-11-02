import numpy as np

from openmdao.api import ExplicitComponent


class ModeshapeGlobMass(ExplicitComponent):

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nElem = self.nodal_data['nElem']
        nDOF = self.nodal_data['nDOF_tot']

        self.add_input('mel', val=np.zeros((nElem, 12, 12)), units='kg')
    
        self.add_output('M_glob', val=np.zeros((nDOF, nDOF)), units='kg')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        nElem = self.nodal_data['nElem']
        nDOF = self.nodal_data['nDOF_tot']
        Elem2DOF = self.nodal_data['Elem2DOF']

        mel = inputs['mel']
       
        M_glob = np.zeros((nDOF, nDOF))
        for k in range(nElem):
            DOFindex=Elem2DOF[k,:]

            for i,ii in enumerate(DOFindex):
                for j,jj in enumerate(DOFindex):
                    M_glob[ii,jj] += mel[k,i,j]

        outputs['M_glob'] = M_glob

    def compute_partials(self, inputs, partials):
        nElem = self.nodal_data['nElem']
        nDOF = self.nodal_data['nDOF_tot']
        Elem2DOF = self.nodal_data['Elem2DOF']

        partials['M_glob', 'mel'] = np.zeros(((nDOF*nDOF),(nElem*12*12)))
