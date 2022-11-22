import numpy as np
import openmdao.api as om

class TangentVector(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nElem = self.nodal_data['nElem']

        self.add_input('d_node', val=np.zeros((nElem,3,1)))
        self.add_input('elem_norm', val=np.zeros(nElem))

        self.add_output('tangent_vecs', val=np.zeros((nElem,3,1)))

    def setup_partials(self):
        nElem = self.nodal_data['nElem']
        nPart = nElem * 3

        self.declare_partials('tangent_vecs', 'd_node', rows=np.arange(nPart), cols=np.arange(nPart))
        self.declare_partials('tangent_vecs', 'elem_norm', rows=np.arange(nPart), cols=np.repeat(np.arange(nElem),3))

    def compute(self, inputs, outputs):
        nElem = self.nodal_data['nElem']
        dx = inputs['d_node']
        le = inputs['elem_norm']

        e1 = np.zeros((nElem,3,1))
        
        for i in range(nElem):
            e1[i,:,:] = dx[i,:,:]/le[i]
            
        outputs['tangent_vecs'] = e1

    def compute_partials(self, inputs, partials):
        nElem = self.nodal_data['nElem']
        dx = inputs['d_node']
        le = inputs['elem_norm']

        partials['tangent_vecs', 'd_node'] = 1./np.repeat(le,3)
        partials['tangent_vecs', 'elem_norm'] = -1. * np.reshape(dx,(1,nElem*3)) / (np.repeat(le**2.,3)) 