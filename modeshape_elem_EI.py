import numpy as np
import myconstants as myconst

from openmdao.api import ExplicitComponent


class ModeshapeElemEI(ExplicitComponent):

    def initialize(self):
        self.options.declare('nElem', types=int)

    def setup(self):
        nElem = self.options['nElem']
    
        self.add_input('D_beam', val=np.zeros(nElem), units='m')

        self.add_output('EI_mode_elem', val=np.zeros(nElem), units='N*m**2')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        nElem = self.options['nElem']
        D_beam = inputs['D_beam']

        outputs['EI_mode_elem'] = np.zeros(nElem)

        for i in range(nElem):
            outputs['EI_mode_elem'][i] = (np.pi / 64.) * D_beam[i]**4. * myconst.E_STL

    def compute_partials(self, inputs, partials):
        nElem = self.options['nElem']
        D_beam = inputs['D_beam']

        partials['EI_mode_elem', 'D_beam'] = np.zeros((nElem, nElem))
        for i in range(nElem):
            partials['EI_mode_elem', 'D_beam'][i, i] = np.pi / 64. * (4. * D_beam[i]**3.) * myconst.E_STL
