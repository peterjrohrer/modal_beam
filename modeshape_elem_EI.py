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
    
        self.add_input('D_beam', val=np.zeros(nElem), units='m')
        self.add_input('wt_beam', val=np.zeros(nElem), units='m')

        self.add_output('EI_mode_elem', val=np.zeros(nElem), units='N*m**2')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        D_beam = inputs['D_beam']
        wt_beam = inputs['wt_beam']

        N_beamelem = len(D_beam)

        outputs['EI_mode_elem'] = np.zeros(N_beamelem)

        for i in range(N_beamelem):
            outputs['EI_mode_elem'][i] = (np.pi / 64.) * (D_beam[i]**4. - (D_beam[i] - 2. * wt_beam[i])**4.) * myconst.E_STL

        a = 1.

    def compute_partials(self, inputs, partials):
        nElem = self.options['nElem']

        D_beam = inputs['D_beam']
        wt_beam = inputs['wt_beam']

        N_beamelem = len(D_beam)

        partials['EI_mode_elem', 'D_beam'] = np.zeros((nElem, nElem))
        partials['EI_mode_elem', 'wt_beam'] = np.zeros((nElem, nElem))

        for i in range(N_beamelem):
            partials['EI_mode_elem', 'D_beam'][i, i] = np.pi / 64. * (4. * D_beam[i]**3. - 4. * (D_beam[i] - 2. * wt_beam[i])**3.) * myconst.E_STL
            partials['EI_mode_elem', 'wt_beam'][i, i] = np.pi / 64. * 8. * (D_beam[i] - 2. * wt_beam[i])**3. * myconst.E_STL
