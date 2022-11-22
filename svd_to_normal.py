import numpy as np
import openmdao.api as om

class SVD2Normal(om.ExplicitComponent):
    '''
    This is simplification that assumes straight lines, ie beams have a single primary axis that is set by the first element
    '''
    ##TODO make this more generic?

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nElem = self.nodal_data['nElem']

        self.add_input('tangent_v', val=np.zeros((3,3)))

        self.add_output('normal_vecs', val=np.zeros((nElem,3,1)))

    def setup_partials(self):
        nElem = self.nodal_data['nElem']
        nPart = nElem * 3
        Hcols = np.tile(np.array([0,3,6]),nElem)

        self.declare_partials('normal_vecs', 'tangent_v', rows=np.arange(nPart), cols=Hcols, val=np.ones(nPart))

    def compute(self, inputs, outputs):
        nElem = self.nodal_data['nElem']

        e2 = np.zeros((nElem,3,1))

        for i in range(nElem):
            e2[i,:,:] = inputs['tangent_v'][:,0].reshape(3,1) 
            
        outputs['normal_vecs'] = e2