import numpy as np

from openmdao.api import ExplicitComponent


class ModeshapeDOFReduce(ExplicitComponent):

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nDOF_tot = self.nodal_data['nDOF_tot']
        nDOF_r = self.nodal_data['nDOF_r']
        Tr = self.nodal_data['Tr']

        self.add_input('M_glob', val=np.zeros((nDOF_tot, nDOF_tot)), units='kg')
        self.add_input('K_glob', val=np.zeros((nDOF_tot, nDOF_tot)), units='N/m')
    
        self.add_output('Mr_glob', val=np.zeros((nDOF_tot, nDOF_r)), units='kg')
        self.add_output('Kr_glob', val=np.zeros((nDOF_tot, nDOF_r)), units='N/m')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        nElem = self.nodal_data['nElem']
        nDOF_tot = self.nodal_data['nDOF_tot']
        nDOF_r = self.nodal_data['nDOF_r']
        Tr = self.nodal_data['Tr']

        M_glob = inputs['M_glob']
        K_glob = inputs['K_glob']

        Mr = (Tr.T).dot(M_glob).dot(Tr)
        Kr = (Tr.T).dot(K_glob).dot(Tr)       
       
        outputs['Mr_glob'] = Mr
        outputs['Kr_glob'] = Kr

    def compute_partials(self, inputs, partials):
        nElem = self.nodal_data['nElem']
        nDOF_tot = self.nodal_data['nDOF_tot']
        nDOF_r = self.nodal_data['nDOF_r']
        Tr = self.nodal_data['Tr']
