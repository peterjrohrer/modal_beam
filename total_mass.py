import numpy as np
import myconstants as myconst
import openmdao.api as om

class TotalMass(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nElem = self.nodal_data['nElem']

        self.add_input('tip_mass', val=0., units='kg')
        self.add_input('M_beam', val=np.zeros(nElem), units='kg')

        self.add_output('tot_M', val=0., units='kg')

    def setup_partials(self):
        nElem = self.nodal_data['nElem']

        self.declare_partials('tot_M', ['tip_mass', 'M_beam'])

    def compute(self, inputs, outputs):
        outputs['tot_M'] = sum(inputs['M_beam']) + inputs['tip_mass']
        
    def compute_partials(self, inputs, partials):
        nElem = self.nodal_data['nElem']      
        partials['tot_M', 'tip_mass'] = 1.
        partials['tot_M', 'M_beam'] = np.ones(nElem)