import numpy as np
import openmdao.api as om

class FirstTangentVector(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nElem = self.nodal_data['nElem']

        self.add_input('tangent_vecs', val=np.zeros((nElem,3,1)))

        self.add_output('first_tangent_vec', val=np.zeros((3,1)))

    def setup_partials(self):
        self.declare_partials('first_tangent_vec', 'tangent_vecs', rows=np.arange(3), cols=np.arange(3), val=np.ones(3))

    def compute(self, inputs, outputs):            
        outputs['first_tangent_vec'] = inputs['tangent_vecs'][0,:,:].T

