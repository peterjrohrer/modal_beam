import numpy as np

from openmdao.api import ExplicitComponent


class TransformMatrix(ExplicitComponent):

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nDOF_tot = self.nodal_data['nDOF_tot']
        nDOF_r = self.nodal_data['nDOF_r']

        self.add_output('Tr', val=np.zeros((nDOF_tot, nDOF_r)))

    def compute(self, inputs, outputs):
        Tr = self.nodal_data['Tr']
        outputs['Tr'] = Tr

    def compute_partials(self, inputs, partials):
        pass