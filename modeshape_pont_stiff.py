import numpy as np
import myconstants as myconst
from openmdao.api import ExplicitComponent


class ModeshapePontStiff(ExplicitComponent):

    def initialize(self):
        self.options.declare('pont_data', types=dict)

    def setup(self):

        pont_data = self.options['pont_data']
        self.N_pont = pont_data['N_pont']
        self.theta = pont_data['theta']

        self.add_input('r_pont', val=0., units='m')
        self.add_input('k11', val=0., units='N/m')
        self.add_input('k33', val=0., units='N/m')

        self.add_output('mode_K11_pont', val=0., units='N/m')
        self.add_output('mode_K22_pont', val=0., units='N*m')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):

        N_pont = self.N_pont
        theta = self.theta

        r_pont = inputs['r_pont']
        k11 = inputs['k11']
        k33 = inputs['k33']

        outputs['mode_K11_pont'] = N_pont*k11
        outputs['mode_K22_pont'] = sum(k33*r_pont*r_pont*np.cos(theta)**2)
    
    ##TODO these partials can be cleaned up
    def compute_partials(self, inputs, partials):
        
        N_pont = self.N_pont
        theta = self.theta

        r_pont = inputs['r_pont']
        k11 = inputs['k11']
        k33 = inputs['k33']

        partials['mode_K11_pont', 'r_pont'] = 0.
        partials['mode_K11_pont', 'k11'] = N_pont
        partials['mode_K11_pont', 'k33'] = 0.

        partials['mode_K22_pont', 'r_pont'] = 0.
        partials['mode_K22_pont', 'k11'] = 0.
        partials['mode_K22_pont', 'k33'] = 0.

        for i in range(N_pont):
            partials['mode_K22_pont', 'r_pont'] += k33 * (2*r_pont) * (np.cos(theta[i])**2)
            partials['mode_K22_pont', 'k33'] += (r_pont**2) * (np.cos(theta[i])**2)
