import numpy as np
import myconstants as myconst

from openmdao.api import ExplicitComponent


class ModeshapeElemEI(ExplicitComponent):

    def initialize(self):
        self.options.declare('nNode', types=int)
        self.options.declare('nElem', types=int)

    def setup(self):
        nNode = self.options['nNode']        
        nElem = self.options['nElem']
    
        self.add_input('D_tower', val=np.zeros(nElem), units='m')
        self.add_input('wt_tower', val=np.zeros(nElem), units='m')

        self.add_output('EI_mode_elem', val=np.zeros(nElem), units='N*m**2')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        D_tower = inputs['D_tower']
        wt_tower = inputs['wt_tower']

        N_towerelem = len(D_tower)

        outputs['EI_mode_elem'] = np.zeros(N_towerelem)

        for i in range(N_towerelem):
            outputs['EI_mode_elem'][i] = (np.pi / 64.) * (D_tower[i]**4. - (D_tower[i] - 2. * wt_tower[i])**4.) * myconst.E_STL

        a = 1.

    def compute_partials(self, inputs, partials):
        nElem = self.options['nElem']

        D_tower = inputs['D_tower']
        wt_tower = inputs['wt_tower']

        N_towerelem = len(D_tower)

        partials['EI_mode_elem', 'D_tower'] = np.zeros((nElem, nElem))
        partials['EI_mode_elem', 'wt_tower'] = np.zeros((nElem, nElem))

        for i in range(N_towerelem):
            partials['EI_mode_elem', 'D_tower'][i, i] = np.pi / 64. * (4. * D_tower[i]**3. - 4. * (D_tower[i] - 2. * wt_tower[i])**3.) * myconst.E_STL
            partials['EI_mode_elem', 'wt_tower'][i, i] = np.pi / 64. * 8. * (D_tower[i] - 2. * wt_tower[i])**3. * myconst.E_STL
